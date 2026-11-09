"""Compare observed phase times of qualified runs; never infer net speed from gains."""
from pathlib import Path
import argparse,hashlib,json
r=Path(__file__).resolve().parents[2]/"artifacts/realman/coffee_service_20261102"

def load_qualified(trial,scope):
    result=json.loads((trial/'result.json').read_text());audit=json.loads((trial/'audit.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if not result['experiment_success'] or result['error'] is not None or audit['passed'] is not True or not all(audit['checks'].values()):
        raise ValueError('timing report requires a completed qualified run')
    if audit['scope']!=scope or audit['result_sha256']!=sha(trial/'result.json') or audit['trace_sha256']!=sha(trial/'trace.json'):
        raise ValueError('qualification scope or evidence hash mismatch')
    if scope == 'continuous_simulation_service_with_assumed_staff_pickup':
        extra = ['delivery_order_audit.json']
        if result['parameters'].get('release_force_segments'):
            extra.append('release_force_audit.json')
        for name in extra:
            evidence=json.loads((trial/name).read_text())
            if evidence.get('passed') is not True or evidence.get('result_sha256')!=sha(trial/'result.json'):
                raise ValueError('missing or stale supplementary qualification: '+name)
            if name=='delivery_order_audit.json' and evidence.get('trace_sha256')!=sha(trial/'trace.json'):
                raise ValueError('stale pickup-order trace')
            if name=='release_force_audit.json' and evidence.get('release_load_sha256')!=sha(trial/'release_load.json'):
                raise ValueError('stale release load trace')
    trace=json.loads((trial/'trace.json').read_text());groups=[]
    for row in trace:
        if not groups or groups[-1]['stage']!=row['stage']:groups.append({'stage':row['stage'],'start_s':row['time_s']})
    for i,g in enumerate(groups):g['end_s']=groups[i+1]['start_s'] if i+1<len(groups) else result['simulation_seconds']
    return result,audit,groups

def windows(groups,start,end):
    found=[]
    for i,g in enumerate(groups):
        if g['stage']!=start:continue
        stop=None
        for other in groups[i+1:]:
            if other['stage']==start:break
            if other['stage']==end:stop=other['start_s'];break
        if stop is None:raise ValueError('missing endpoint before next repeated phase: '+start+' -> '+end)
        found.append({'start_s':g['start_s'],'end_s':stop,'duration_s':stop-g['start_s']})
    if not found:raise ValueError('missing phase: '+start)
    return found

def cup_pickup_windows(groups):
    """End at cup_hold completion, before any intervening pot operation.

    Cup-first and pot-first workflows have different next actions. Using the
    later cup_raise boundary incorrectly credits the pot take as cup speedup.
    Existing cup-stage settling recoveries are included in the measured time.
    """
    found=[]
    for i,g in enumerate(groups):
        if g['stage']!='cup_approach':
            continue
        end=None
        for j,h in enumerate(groups[i+1:],i+1):
            if h['stage']=='cup_approach' or not h['stage'].startswith('cup_'):
                break
            if h['stage']=='cup_hold':
                end=h['end_s']
                for continuation in groups[j+1:]:
                    if continuation['stage'] not in {'cup_hold','cup_hold_rack_settle'}:
                        break
                    end=continuation['end_s']
                break
        if end is None or end<=g['start_s']:
            raise ValueError('cup pickup lacks a completed hold before another action')
        found.append({'start_s':g['start_s'],'end_s':end,'duration_s':end-g['start_s']})
    if not found:
        raise ValueError('missing cup pickup')
    return found


def report(baseline,service):
    br,ba,bg=load_qualified(baseline,'continuous_baseline_lifecycle')
    sr,sa,sg=load_qualified(service,'continuous_simulation_service_with_assumed_staff_pickup')
    phases={
      'cup_pickup':('cup_approach','cup_hold','cup_hold'),
      'pot_extraction':('pot_extract_010','cup_raise','cup_rejoin_pregrasp'),
      'cup_free_transport':('cup_raise','pot_retreat','pot_retreat'),
      'pot_free_outbound':('pot_retreat','pour_tilt_5','pour_tilt_5'),
      'tilt_up':('pour_tilt_5','pour_geometry_hold','pour_geometry_hold'),
      'prescribed_payload_transfer':('prescribed_payload_transfer','pour_restore_40','pour_restore_40'),
      'loaded_restore':('pour_restore_40','pot_restore_approach','pot_restore_approach'),
      'pot_free_return':('pot_restore_approach','cup_table_clearance_lift','cup_table_clearance_lift'),
      'cup_place_release_retract':('cup_table_clearance_lift','pot_return_0.22','staff_collection_wait'),
      'pot_supported_return_and_release':('pot_return_0.22','exit_empty_gripper_aperture',None),
    }
    comparisons={}
    for name,(start,bend,send) in phases.items():
        bw=cup_pickup_windows(bg) if name=='cup_pickup' else windows(bg,start,bend)
        if name=='cup_pickup':
            sw=cup_pickup_windows(sg)
        elif send is None:
            # There are two returns: one before the refill wait, one before exit.
            sw=[]
            for i,g in enumerate(sg):
                if g['stage']!=start:continue
                after=next((h for h in sg[i+1:] if h['stage'] in {'wait_for_external_refill','exit_empty_gripper_aperture'}),None)
                if after is None:raise ValueError('return endpoint absent')
                sw.append({'start_s':g['start_s'],'end_s':after['start_s'],'duration_s':after['start_s']-g['start_s']})
        else:sw=windows(sg,start,send)
        if len(bw)!=1:raise ValueError('single baseline serving required')
        comparisons[name]={'baseline':bw[0],'service_repetitions':sw,'observed_duration_ratios':[bw[0]['duration_s']/x['duration_s'] for x in sw]}
    checks={'three_cup_pickups':len(comparisons['cup_pickup']['service_repetitions'])==3,'three_pours':len(comparisons['prescribed_payload_transfer']['service_repetitions'])==3,'two_pot_takes':len(comparisons['pot_extraction']['service_repetitions'])==2,'two_pot_returns':len(comparisons['pot_supported_return_and_release']['service_repetitions'])==2}
    args=sr['parameters']['arguments'];waits=[g for g in sg if g['stage']=='wait_for_external_refill']
    return {'passed':all(checks.values()),'checks':checks,'scope':'observed phase durations from two independently qualified runs; route/order/load differences prevent a controlled speed-only comparison','configured_factors':{'free_transport':args['free_speed_scale'],'pot_contact':args['pot_contact_speed_scale'],'cup_contact':args['contact_speed_scale'],'postpour_return':sr['parameters']['postpour_return_speed_scale'],'pour':1},'phase_comparisons':comparisons,'baseline_one_cup_total_s':br['simulation_seconds'],'service_three_cups_one_refill_total_s':sr['simulation_seconds'],'refill_wait_windows':waits,'whole_workflow_speed_ratio':None,'whole_workflow_speed_ratio_reason':'Different cup counts, pot-first order, external pickup and refill; no equal-work comparison','boundary_sampling_uncertainty_s':.101,'cup_pickup_boundary':'cup_approach through end of cup_hold; excludes intervening pot actions','baseline_trace_sha256':ba['trace_sha256'],'service_trace_sha256':sa['trace_sha256'],'physical_motion_allowed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('service',type=Path);p.add_argument('--baseline',type=Path,default=r/'lifecycle_coffee/baseline_lifecycle_exit_reference_v5');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior timing evidence')
    out=report(a.baseline,a.service);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));raise SystemExit(0 if out['passed'] else 1)
