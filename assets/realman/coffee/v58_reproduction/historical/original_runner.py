"""Simulation coffee service with explicit staff pickup at one fixed delivery point.
Only the declared staff event moves a released cup out of the workspace.
No hardware IO; no robot/object reset is used to complete robot actions.
"""

import argparse
from cap_harness.realman.rack_recovery import rack_settle_members
from cap_harness.realman.service_release import release_retry_allowed, RETRY_OPENING
from cap_harness.realman.held_object import stationary_grasp_target
from cap_harness.realman.service_lifecycle import ServiceLifecycle
from dataclasses import asdict
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from local_ik_constrained import solve_local_mount as solve_posture
from local_ik_return import solve_local_mount as solve_return
from local_ik import solve_local_mount
from local_ik_clearance_strict import solve_local_mount as solve_clearance
from cap_harness.realman.service_contacts import prohibited_pot_contacts
from cap_harness.realman.service_payload import ServicePayload
from cap_harness.realman.service_collection import StaffCupCollection, fixed_delivery_slot
from cap_harness.realman.service_evaluation import ServiceGeometryEvaluator, selected_cup_withdraw_clearance, next_table_slot
from cap_harness.realman.coffee_service import CoffeeService, CoffeeAction, ServiceEvidence
from cap_harness.realman.joint_planning import plan_joint_path
from cap_harness.realman.service_planning import plan_service_joint_path
import hashlib
import inspect
from cap_harness.realman.contact_rollout import ContactRollout, cup_withdraw_clearance
from cap_harness.realman.upper_handle import (
    upper_handle_target,
    upper_bar_contacts,
    upper_handle_loads,
    handle_axis_to_seat_error,
    OutletUnilateralAllowance,
)
from cap_harness.realman.grasp_geometry import (
    grasp_geometry,
    mount_pose_for_grasp,
    pot_handle_grasp_pose,
    gripper_linkage_errors,
)
from cap_harness.realman.grasp_monitor import HandleGraspMonitor, HandleGraspLimits
from cap_harness.realman.tracking import bounded_rotation_integral, bumpless_rotation_limit
from cap_harness.realman.platform import (
    is_machine_support_geom,
    support_height,
    nominal_seated_height,
)
from cap_harness.realman.supported_slide import supported_slide_settled, supported_yaw_integral
from cap_harness.realman.diagnostics import PadLoadLimits, PadLoadMonitor
from cap_harness.realman.return_geometry import clearance_entry_centres
from cap_harness.realman.workspace import robot_contacts

parser = argparse.ArgumentParser()
parser.add_argument("--pot-contact-speed-scale",type=float,choices=[1,1.5,2],default=2)
parser.add_argument("--extraction-timestep",type=float,choices=[.001,.0005,.00025,.0001])
parser.add_argument("--requested-cups", type=int, default=3)
parser.add_argument("--demo-refills", type=int, default=1)
parser.add_argument("--stop-after-service-take-pot", action="store_true")
parser.add_argument("--extraction-contact-timestep",type=float,choices=[.0005,.00025,.0001])
parser.add_argument("--transport-contact-timestep",type=float,choices=[.0005,.00025,.0001])
parser.add_argument("--stop-after-smooth-transport",action="store_true")
parser.add_argument("--free-speed-scale", type=float, default=1)
parser.add_argument("--contact-speed-scale", type=float, default=1)
parser.add_argument("--nominal-yaw-deg", type=float, default=135)
parser.add_argument("--travel-heading-deg", type=float)
parser.add_argument("--slide-yaw-tolerance-deg", type=float, default=5)
parser.add_argument("--yaw-tolerance-deg", type=float, default=10)
parser.add_argument("--grasp-depth-mm", type=float, default=0)
parser.add_argument("--sustained-load-n", type=float, default=30)
parser.add_argument("--peak-load-n", type=float, default=35)
parser.add_argument("--hold-seconds", type=float, default=30)
parser.add_argument("--resume-tilt", type=int, choices=[40,45], default=40)
parser.add_argument("--return-timestep", type=float, choices=[0.001,0.0005,0.00025,0.0001,0.00005])
parser.add_argument("--pour-timestep", type=float, choices=[0.001,0.0005,0.00025,0.0001,0.00005])
parser.add_argument("--transport-timestep", type=float, choices=[0.001,0.0005,0.00025,0.0001,0.00005])
parser.add_argument("--stop-after-pour-geometry", action="store_true")
parser.add_argument("--pour-angular-rate", type=float, default=0.04)
parser.add_argument("--gripper-damping", type=float, choices=[0.2,0.8,2.0], default=0.2)
parser.add_argument("--left-contact-margin", type=float, choices=[0,1e-7,1e-6,5e-6], default=0)
parser.add_argument("--collision-backend", choices=["native", "libccd"], default="native")
parser.add_argument("--physics-timestep", type=float, choices=[0.001,0.0005,0.00025,0.0001,0.00005])
parser.add_argument("--hold-source-stage", choices=["load_transfer", "pot_transfer_rise"])
parser.add_argument("--grasp-inplane-deg", type=float, default=0)
parser.add_argument("--grasp-radial-mm", type=float, default=0)
parser.add_argument("--allow-outlet-unilateral-ms", type=float, default=200)
parser.add_argument("--allow-return-regrasp", action="store_true")
parser.add_argument("--return-grasp-roll-deg", type=float, default=20)
parser.add_argument("--supported-level-correction-deg", type=float, default=0)
parser.add_argument("--extraction-yaw-max-deg", type=float, default=2.5)
parser.add_argument("--pad-load-limit", type=float, default=25.0)
parser.add_argument("--loaded-pd", action="store_true")
parser.add_argument("--prelift-mm", type=float, default=0.5)
parser.add_argument("--prelift", action="store_true")
parser.add_argument("--payload-feedforward", action="store_true")
parser.add_argument("--case", default="stock_two_cups")
parser.add_argument("--stop-after-withdrawal", action="store_true")
parser.add_argument("--name", default="run")
parser.add_argument("--checkpoint")
parser.add_argument("--pre-pour-offset-mm", type=float, default=55)
parser.add_argument("--pour-offset-mm", type=float, default=26)
parser.add_argument("--pour-gap-mm", type=float, default=25)
parser.add_argument("--adaptive-settle", action="store_true")
parser.add_argument("--settle", type=float, default=3)
parser.add_argument("--release-opening", type=float, default=0.205)
parser.add_argument("--return-speed", type=float, default=0.008)
parser.add_argument("--return-lift-mm", type=float, default=1.428)
parser.add_argument("--extract-speed", type=float, default=0.008)
parser.add_argument("--extraction-lift-mm", type=float, default=1.428)
parser.add_argument("--support-torsion", action="store_true")
parser.add_argument(
    "--phase",
    choices=[
        "start",
        "tilted",
        "hold",
        "pinched",
        "pot",
        "extraction",
        "egress",
        "far",
        "return",
        "extended",
        "regrasp",
        "pour",
        "approach",
        "transfer",
        "release",
        "loaded_return",
        "seated",
        "regrasped",
        "cup_extract",
    ],
    default="start",
)
parser.add_argument("--platform-roundtrip", action="store_true")
parser.add_argument("--supported-slide-tilt-deg", type=float, default=2.0)
parser.add_argument("--return-yaw-max-deg", type=float, default=1.0)
parser.add_argument("--return-start-mm", type=int, default=180)
parser.add_argument("--return-clearance", action="store_true")
parser.add_argument("--return-entry-offset-mm", type=float, default=2.8284271247461903)
parser.add_argument("--suspend-height-mm", type=float, default=0)
parser.add_argument("--air-correction-deg", type=float, default=2)
parser.add_argument("--grip-scale", type=float, default=1)
parser.add_argument("--grip-effort-scale", type=float, default=1)
parser.add_argument("--diagnostic-grasp-angle-deg", type=float, default=5)
parser.add_argument("--regrasp-offset-mm", type=float, default=-12)
parser.add_argument("--initial-grasp-roll-deg", type=float, default=0)
parser.add_argument("--grasp-height-offset-mm", type=float, default=0)
parser.add_argument("--regrasp-roll-deg", type=float, default=0)
parser.add_argument("--cup-pour-height-m", type=float, default=0.84)
parser.add_argument("--pour-route", choices=["original", "central"], default="original")
parser.add_argument("--constrained-all", action="store_true")
parser.add_argument("--cold-regrasp", action="store_true")
parser.add_argument("--regrasp-radius-mm", type=int, choices=[60, 120, 180], default=60)
parser.add_argument("--held-contact-timestep",type=float,choices=[.00025,.0001],required=True)
parser.add_argument("--low-payload-restore-correction-deg", type=float, default=8.0)
parser.add_argument("--cup-endpoint-max-seconds", type=float, default=8.0)
parser.add_argument("--prescribed-rate-ml-s", type=float, default=10.0)
parser.add_argument("--transfer-timestep", type=float, choices=[.0001,.00025,.0005,.001], default=.0001)
parser.add_argument("--pour-tilt-speed-scale", type=float, choices=[1.0,2.0], default=1.0)
parser.add_argument("--pour-angular-limit-deg-s", type=float, choices=[5.0,6.0,8.0], default=5.0)
args = parser.parse_args()
args.delivery_mode = "staff_collect_fixed_slot"
if not np.isfinite(args.prescribed_rate_ml_s) or not 1.0 <= args.prescribed_rate_ml_s <= 60.0:
    raise ValueError("prescribed transfer rate must be in [1, 60] ml/s")
if not 3.0 <= args.cup_endpoint_max_seconds <= 8.0:
    raise ValueError("cup endpoint correction budget must be3..8seconds")
if not 5.5 <= args.low_payload_restore_correction_deg <= 8.0:
    raise ValueError("bounded low-payload restoration authority must be5.5..8degrees")
if not np.isfinite(args.sustained_load_n) or not 30 <= args.sustained_load_n <= 40:
    raise ValueError("sustained load exploration limited to 30..40 N")
if not np.isfinite(args.peak_load_n) or not args.sustained_load_n < args.peak_load_n <= 50:
    raise ValueError("peak limit must exceed sustained and be at most 50 N")
if not np.isfinite(args.hold_seconds) or not 1 <= args.hold_seconds <= 60:
    raise ValueError("hold must be 1..60 s")
if args.phase == "hold" and args.hold_source_stage is None:
    raise ValueError("hold requires explicit source stage")
if not 0 <= args.supported_level_correction_deg <= 3:
    raise ValueError("invalid supported leveling authority")
if not 0 < args.extraction_yaw_max_deg <= 5:
    raise ValueError("invalid extraction yaw authority")
if not 40 <= args.pre_pour_offset_mm <= 70:
    raise ValueError("invalid pre-pour radial offset")
if not 0 <= args.pour_offset_mm <= 29 or not 15 <= args.pour_gap_mm <= 30:
    raise ValueError("invalid spout target")
if not 0 <= args.return_grasp_roll_deg <= 45:
    raise ValueError("invalid return grasp roll")
if not 0 <= args.initial_grasp_roll_deg <= 45:
    raise ValueError("invalid initial grasp roll")
if not -18 <= args.regrasp_offset_mm <= 18 or not -45 <= args.regrasp_roll_deg <= 60:
    raise ValueError("invalid supported regrasp")
if not 5 <= args.diagnostic_grasp_angle_deg <= 8:
    raise ValueError("invalid diagnostic bound")
if not .7 <= args.grip_effort_scale <= 1:
    raise ValueError("invalid effort-only reduction")
if not 0.185 <= args.release_opening <= 0.3:
    raise ValueError("invalid exploratory release aperture")
if not 1 <= args.grip_scale <= 1.25:
    raise ValueError("invalid grip scale")
if not 0 <= args.suspend_height_mm <= 20 or not 2 <= args.air_correction_deg <= 6:
    raise ValueError("invalid suspension control")
if not 0 <= args.return_entry_offset_mm <= 10:
    raise ValueError("entry offset requires 0..10 mm")
if not np.isfinite(args.return_yaw_max_deg) or not 0 < args.return_yaw_max_deg <= 5:
    raise ValueError("invalid return yaw authority")
if args.return_start_mm not in [10, 20, 30, 40, 60, 100, 120, 140, 180] or (
    args.return_start_mm != 180 and args.phase not in {"return", "loaded_return", "seated"}
):
    raise ValueError("invalid return resume radius")
checkpoint_stage = {
    "seated": "seat_contact",
    "tilted": f"pour_tilt_{args.resume_tilt}",
    "hold": args.hold_source_stage,
    "loaded_return": f"pot_return_{args.return_start_mm / 1000}",
    "approach": "pot_transfer_rise",
    "transfer": "pot_retreat",
    "release": "cup_support_probe",
    "pinched": "pot_close",
    "cup_extract": "cup_extract_down",
    "regrasped": "supported_regrasp_close",
    "pot": "cup_hold",
    "extended": "pot_extract_180",
    "regrasp": f"pot_extract_{args.regrasp_radius_mm:03d}",
    "pour": "pot_extract_240",
    "far": "pot_extract_100",
    "egress": "pot_extract_060",
    "extraction": "pot_extract_010",
}.get(args.phase)
resume_stage = (
    "pot_extract_180" if args.return_start_mm == 180 else f"pot_return_{args.return_start_mm:03d}"
)
if not np.isfinite(args.supported_slide_tilt_deg) or not 0 < args.supported_slide_tilt_deg <= 8:
    raise ValueError("supported slide tilt must be in (0,8] degrees")
if args.phase == "return":
    checkpoint_stage = resume_stage
if args.phase == "start" and args.checkpoint:
    raise ValueError("fresh start cannot restore checkpoint")
if args.phase != "start" and not args.checkpoint:
    raise ValueError("extraction resume needs checkpoint")
root = Path(__file__).parent / args.case
if (root / args.name).exists():
    raise FileExistsError("use a new trial name; never overwrite recorded evidence")
e = ContactRollout(
    root / "review",
    root / args.name,
    checkpoint=None if args.phase != "start" else root / "initial_standing.npz",
)
e.source_bytes = Path(__file__).read_bytes()
(e.output / "return_elbow_impl.py").write_bytes((Path(__file__).parent / "local_ik_return_elbow_v2.py").read_bytes())
(e.output / "upper_handle_impl.py").write_bytes(
    Path(inspect.getfile(upper_handle_target)).read_bytes()
)
(e.output / "tracking_impl.py").write_bytes(Path(inspect.getfile(bumpless_rotation_limit)).read_bytes())
(e.output / "supported_slide_impl.py").write_bytes(
    Path(inspect.getfile(supported_slide_settled)).read_bytes()
)
(e.output / "pour_ik_impl.py").write_bytes(Path(inspect.getfile(solve_posture)).read_bytes())
(e.output / "return_ik_impl.py").write_bytes(Path(inspect.getfile(solve_return)).read_bytes())
active_cup = "cup_0"
a = e.adapter
m, d = a.model, a.data
if args.phase not in {"start", "loaded_return"} or (args.phase == "start" and args.checkpoint) or args.allow_return_regrasp or args.platform_roundtrip:
    raise ValueError("only fresh start or verified loaded-return continuation is supported")
a.task_evaluator = ServiceGeometryEvaluator(a,e.config)
a.task_evidence = a.task_evaluator.update()
lifecycle=ServiceLifecycle(e,Path(__file__).parent/"lifecycle/plan.json")
ready_right_q=np.asarray(lifecycle.points[-1]["qpos"])[a.qadr["right"]].copy()
load_limits = PadLoadLimits(**{**e.config["simulation_diagnostics"]["pad_load"], "sustained_n": args.sustained_load_n, "clear_n": args.sustained_load_n - 1, "peak_n": args.peak_load_n})
load_monitor = PadLoadMonitor(load_limits)
release_load_monitor = None
release_load_rows = []
release_guard_active = False
release_last_load = None

def check_release_load():
    global release_load_monitor, release_guard_active, release_last_load
    active = e.stage in {'pot_release', 'left_gripper_retract'}
    if not active:
        if release_guard_active:
            if release_load_rows[-1]['time_s'] != release_last_load['time_s']:
                release_load_rows.append(release_last_load)
            e.parameters['release_force_segments'][-1]['end_s'] = release_last_load['time_s']
        release_guard_active = False
        return
    if not release_guard_active:
        release_load_monitor = PadLoadMonitor(load_limits)
        e.parameters.setdefault('release_force_segments', []).append(dict(
            start_s=float(d.time-m.opt.timestep), end_s=None, physics_samples=0,
            timestep_s=float(m.opt.timestep), peak_finger_n=0.0))
        release_guard_active = True
    segment=e.parameters['release_force_segments'][-1]
    if segment['timestep_s'] != float(m.opt.timestep):
        raise RuntimeError('unexpected release guard timestep change')
    loads=upper_handle_loads(m,d,e.config['pot'])
    metrics=release_load_monitor.update(float(d.time),loads['finger_normal_resultants_n'])
    failures=list(metrics['failures'])
    if loads['maximum_contact_normal_n']>load_limits.peak_n:
        failures.append('single_contact_peak_load')
    segment['physics_samples']+=1
    segment['end_s']=float(d.time)
    segment['peak_finger_n']=max(segment['peak_finger_n'],max(loads['finger_normal_resultants_n']))
    release_last_load=dict(time_s=float(d.time),stage=e.stage,loads=loads,diagnostic=metrics,failures=failures,segment=len(e.parameters['release_force_segments'])-1)
    if not release_load_rows or release_load_rows[-1]['segment'] != release_last_load['segment'] or d.time-release_load_rows[-1]['time_s']>=.001-1e-10 or failures:
        release_load_rows.append(release_last_load)
    if failures:raise RuntimeError('release contact load: '+str(failures))
outlet_allowance = OutletUnilateralAllowance(args.allow_outlet_unilateral_ms / 1000)
airborne_since = None
low_payload_restore_since = None
m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_MULTICCD)
transfer = ServicePayload(m,d,e.config)
collector = StaffCupCollection(a, e.config, transfer, simulation_only=True)
if args.phase != "start":
    cp = Path(args.checkpoint)
    meta = json.loads(cp.with_suffix(".json").read_text())
    if (
        meta["stage"] != checkpoint_stage
        or hashlib.sha256(cp.read_bytes()).hexdigest() != meta["npz_sha256"]
        or meta["model_sha256"] != e.model_hash
    ):
        raise ValueError("matching stage and model checkpoint required")
    state = meta["runtime_payload"]
    expected = transfer.state()
    if (
        hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        != meta["runtime_payload_sha256"]
    ):
        raise ValueError("payload checksum mismatch")
    if args.phase in {"release", "loaded_return", "seated"}:
        transfer.restore_payload_state(state)
        expected = transfer.state()
        if abs(transfer.cup_volume_m3 - 0.8 * transfer.cavities[active_cup].capacity_m3) > 1e-12:
            raise ValueError("release resume requires completed prescribed target")
    for key in ["mode", "pot_volume_m3", "cup_volume_m3", "initial_volume_m3"]:
        if state[key] != expected[key]:
            raise ValueError("requires untransferred payload")
    for name, properties in expected["body_properties"].items():
        if any(
            not np.allclose(state["body_properties"][name][k], v, atol=1e-10, rtol=0)
            for k, v in properties.items()
        ):
            raise ValueError("payload/model mismatch")
    source_options = meta["runtime_options"]
    if (source_options["enable_flags"] != int(m.opt.enableflags)
        or source_options["timestep_s"] not in [0.001,0.0005,0.00025,0.0001,0.00005]
        or source_options.get("disable_flags",0) not in [0,int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)]):
        raise ValueError("runtime options mismatch")
    m.opt.timestep = source_options["timestep_s"]
    m.opt.disableflags = source_options.get("disable_flags",0)
    mujoco.mj_setState(m, d, np.load(cp)["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(m, d)
    e.checkpoint = str(cp)
    e.allowed = {tuple(x) for x in meta["allowed_contacts"]}
    e.trace = []
    e.last_log_s = -1
    resume_meta = meta
    if args.phase == "loaded_return":
        events = meta["staff_collection"]["events"]
        for event in events:
            cup = event["cup_id"]
            a.task_evaluator.served[cup] = np.asarray(event["before_position_m"], dtype=float)
            a.task_evaluator.collected_cups[cup] = np.asarray(event["after_position_m"], dtype=float)
        collector.events = events
        for cup in a.task_evaluator.all_cups:
            if cup not in a.task_evaluator.served:
                a.task_evaluator.initial[cup] = d.body(cup).xpos.copy()
        a.task_evidence = a.task_evaluator.update()
if args.physics_timestep is not None:
    m.opt.timestep = args.physics_timestep
if args.collision_backend == "libccd":
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
for gn in ["left_left_pad_contact", "left_right_pad_contact", "ltool_Left_Support_Link_collision_0", "ltool_Right_Support_Link_collision_0"]:
    m.geom_margin[m.geom(gn).id] = args.left_contact_margin
grip_id = m.actuator("left_gripper_servo").id
gain_reference = {
    "kp": float(m.actuator_gainprm[grip_id, 0]),
    "forcerange": m.actuator_forcerange[grip_id].tolist(),
}
m.actuator_biasprm[grip_id, 2] = -args.gripper_damping
m.actuator_gainprm[grip_id, 0] *= args.grip_scale
m.actuator_biasprm[grip_id, 1] *= args.grip_scale
m.actuator_forcerange[grip_id] *= args.grip_scale * args.grip_effort_scale
runtime_grip = {
    "velocity_damping": args.gripper_damping,
    "scale": args.grip_scale,
    "effort_scale": args.grip_effort_scale,
    "kp": float(m.actuator_gainprm[grip_id, 0]),
    "forcerange": m.actuator_forcerange[grip_id].tolist(),
    "xml_reference": gain_reference,
}
(e.output / "payload_impl.py").write_bytes(
    Path(inspect.getfile(ServicePayload)).read_bytes()
)
right_load_on = transfer.cup_volume_m3 > 0 and args.phase not in {"loaded_return", "seated"}
torso_geom_id = m.geom("xb_link_collision_0").id
cup_separation_monitor = False
cup_surface_pairs = []
def update_cup_pairs():
    global cup_surface_pairs
    remaining = set(a.task_evaluator.cups[1:])
    cup_surface_pairs = [(i,j) for i in range(m.ngeom)
        if m.body(m.geom_bodyid[i]).name==active_cup and (m.geom_contype[i] or m.geom_conaffinity[i])
        for j in range(m.ngeom) if m.body(m.geom_bodyid[j]).name in remaining
        and (m.geom_contype[j] or m.geom_conaffinity[j])]


def cup_surface_gap():
    gap = 0.01
    for i, j in cup_surface_pairs:
        if (
            np.linalg.norm(d.geom_xpos[i] - d.geom_xpos[j]) - m.geom_rbound[i] - m.geom_rbound[j]
            > gap
        ):
            continue
        gap = min(gap, float(mujoco.mj_geomDistance(m, d, i, j, gap, None)))
    return gap


original_save = e.save_checkpoint
original_record = e.record


def payload_save(path):
    original_save(path)
    if True:
        meta = json.loads(path.with_suffix(".json").read_text())
        meta["runtime_left_contact_margin_m"] = args.left_contact_margin
        meta["runtime_grip"] = runtime_grip
        meta["runtime_payload"] = transfer.state()
        meta["staff_collection"] = collector.state()
        meta["staff_collection_sha256"] = hashlib.sha256(json.dumps(meta["staff_collection"], sort_keys=True).encode()).hexdigest()
        meta["runtime_options"] = {
            "timestep_s": float(m.opt.timestep),
            "enable_flags": int(m.opt.enableflags),
            "disable_flags": int(m.opt.disableflags),
        }
        meta["runtime_payload_sha256"] = hashlib.sha256(
            json.dumps(meta["runtime_payload"], sort_keys=True).encode()
        ).hexdigest()
        if monitor_on and monitor is not None:
            context = {
                "version": 1,
                "grasp_epoch": grasp_epoch,
                "held_relative": held.tolist(),
                "seat_pose": reference_seat.tolist(),
                "handle_local_m": local_h.tolist(),
                "feedback_m": feedback.tolist(),
                "angular_feedback_rad": angular_feedback.tolist(),
                "load_ff_start": load_ff_start,
                "last_planned_pose": None
                if last_planned_pose is None
                else last_planned_pose.tolist(),
                "extra_suspension_m": extra_suspension_m,
                "use_posture": use_posture,
                "grasp_limits": asdict(monitor.limits),
                "load_limits": asdict(load_monitor.limits),
                "grasp_timers": {
                    k: getattr(monitor, k)
                    for k in ["last_time", "unilateral_since", "bilateral_since"]
                },
                "outlet_disengaged": outlet_allowance.disengaged,
                "airborne_since_s": airborne_since,
                "low_payload_restore_since_s": low_payload_restore_since,
                "load_timers": {
                    k: getattr(load_monitor, k)
                    for k in ["last_time", "over_since", "asymmetric_since"]
                },
            }
            meta["grasp_context"] = context
            meta["grasp_context_sha256"] = hashlib.sha256(
                json.dumps(context, sort_keys=True).encode()
            ).hexdigest()
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")


def payload_record(*, force=False):
    old = len(e.trace)
    original_record(force=force)
    if len(e.trace) > old:
        e.trace[-1]["runtime_payload"] = transfer.state()
        e.trace[-1]["staff_collection"] = collector.state()
        e.trace[-1]["upper_bar_bearing"] = upper_handle_loads(m, d, e.config["pot"])
        e.trace[-1]["rack_support_diagnostics"] = a.task_evidence.get("rack_support_diagnostics")


e.save_checkpoint = payload_save
e.record = payload_record
cup_keys = ["remaining_cups_retained"]
grasp_epoch = 0
use_posture = args.constrained_all
allow_pour_tilt = args.phase == "tilted"
planned_pot_z = None
load_ff_start = None
monitor = None
monitor_on = False
guard_rows = []
control_rows = []
command_rows = []
velocity_rows = []
force_rows = []
support_rows = []
last_guard_log = -1.0
reference_seat = None


def pose(name="pot"):
    T = np.eye(4)
    T[:3, :3] = d.body(name).xmat.reshape(3, 3)
    T[:3, 3] = d.body(name).xpos
    return T


def relative():
    return np.linalg.inv(np.array(grasp_geometry(m, d, "left")["T_world_grasp"])) @ pose()


left_commanded_this_tick = False


def command(arm, G):
    global left_commanded_this_tick
    if cup_separation_monitor:
        gap = cup_surface_gap()
        e.parameters["cup_min_surface_gap_m"] = min(
            e.parameters.get("cup_min_surface_gap_m", 0.01), gap
        )
        if gap < 0.003:
            raise RuntimeError("cup remaining-surface clearance below 3 mm")
    T = mount_pose_for_grasp(G, grasp_geometry(m, d, arm)["T_mount_grasp"])
    fn = (
        (solve_posture if use_posture else solve_clearance)
        if arm == "left"
        else (
            solve_posture
            if args.constrained_all or (use_posture and args.pour_route == "central")
            else solve_local_mount
        )
    )
    if (
        args.return_clearance
        and arm == "left"
        and (e.stage.startswith("pot_return_") or e.stage == "seat_contact")
    ):
        fn = solve_return
    if arm == "left" and e.stage in {
        "pot_restore_approach", "pot_back_to_front", "pot_back_to_height",
        "pot_lateral_return", "pot_back_to_retreat", "pot_back_to_staging",
    }:
        from local_ik_return_elbow_v2 import solve_local_mount as return_elbow_solver
        fn = return_elbow_solver
    r = fn(a, arm, T)
    if not r["pose_solved"]:
        # Bounded Cartesian correction: keep the joint interval and endpoint tests.
        # A recovered command is an intermediate target, never endpoint success.
        sid = m.site(arm + "_tool_mount").id
        current = np.eye(4)
        current[:3, 3] = d.site_xpos[sid]
        current[:3, :3] = d.site_xmat[sid].reshape(3, 3)
        rv = Rotation.from_matrix(T[:3, :3] @ current[:3, :3].T).as_rotvec()
        for fraction in [0.5, 0.25, 0.125, 0.0625]:
            intermediate = current.copy()
            intermediate[:3, 3] += (T[:3, 3] - current[:3, 3]) * fraction
            intermediate[:3, :3] = Rotation.from_rotvec(rv * fraction).as_matrix() @ current[:3, :3]
            candidate = fn(a, arm, intermediate)
            if candidate["pose_solved"]:
                e.parameters.setdefault("bounded_ik_corrections", []).append(
                    {
                        "time_s": float(d.time),
                        "stage": e.stage,
                        "fraction": fraction,
                        "original_error_m": r["position_error_m"],
                        "arm": arm,
                        "intermediate_result": candidate,
                    }
                )
                r = candidate
                break
    if not r["pose_solved"]:
        e.parameters.setdefault("ik_failures", []).append(
            {
                "time_s": float(d.time),
                "stage": e.stage,
                "arm": arm,
                "grasp_target": G.tolist(),
                "mount_target": T.tolist(),
                "result": r,
            }
        )
        raise RuntimeError("local IK failed " + json.dumps(r))
    target_joints = np.array(r["joint_positions"])
    previous_control = d.ctrl[a.actuators[arm]].copy()
    if args.payload_feedforward and (
        (arm == "left" and load_ff_start is not None) or (arm == "right" and right_load_on)
    ):
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        bodyid = m.site_bodyid[m.site(arm + "_tool_mount").id]
        vessel = "pot" if arm == "left" else active_cup
        mujoco.mj_jac(m, d, jacp, jacr, d.body(vessel).xipos, bodyid)
        compensation = jacp[:, a.vadr[arm]].T @ (-m.body(vessel).mass[0] * m.opt.gravity)
        compensation *= min(1.0, max(0.0, (d.time - load_ff_start) / 2.0)) if arm == "left" else 1.0
        target_joints += compensation / m.actuator_gainprm[a.actuators[arm], 0]
        e.parameters["max_feedforward_nm"] = max(
            e.parameters.get("max_feedforward_nm", 0), float(np.max(np.abs(compensation)))
        )
    command_rows.append({
        "time_s": float(d.time), "stage": e.stage, "arm": arm,
        "grasp_target": G.tolist(), "mount_target": T.tolist(), "ik": r,
        "previous_ctrl": previous_control.tolist(), "target_ctrl": target_joints.tolist(),
        "measured_qpos": d.qpos[a.qadr[arm]].tolist(),
        "measured_qvel": d.qvel[a.vadr[arm]].tolist(),
        "ctrl_delta_rad": (target_joints-previous_control).tolist(),
    })
    d.ctrl[a.actuators[arm]] = target_joints
    if arm == "left":
        left_commanded_this_tick = True


def check_upper_handle_pinch():
    evidence = upper_handle_loads(m, d, e.config["pot"])
    if not evidence["opposed_contact"]:
        raise RuntimeError("missing opposing finger contact: " + json.dumps(evidence))


e.check_pot_pinch = check_upper_handle_pinch
e.command_grasp = command


from cap_harness.realman.coffee_service import MotionSpeedProfile, MotionTiming
speed_profile = MotionSpeedProfile(args.free_speed_scale, args.contact_speed_scale)
def base_stage_speed():
    cup_free = e.stage in {"cup_placement_clearance_raise", "cup_raise", "cup_position", "cup_table_clearance_lift", "cup_table_inset", "right_retract"}
    pot_free = e.stage in {"pot_retreat", "pot_transfer_height", "pot_lateral_transfer", "pot_transfer_rise", "pot_above_cup", "pot_restore_approach", "pot_back_to_front", "pot_back_to_height", "pot_lateral_return", "pot_back_to_retreat", "pot_back_to_staging", "left_gripper_retract"}
    clear = cup_free or (pot_free and a.task_evidence["evidence"]["pot_clear_of_machine"].value is True and upward_platform_support() <= .02)
    return speed_profile.factor(independently_clear=clear)

from cap_harness.realman.coffee_speed import pot_transport_speed_factor

def pot_speed_decision():
    independently_clear = a.task_evidence["evidence"]["pot_clear_of_machine"].value is True
    support_n = float(upward_platform_support())
    factor = pot_transport_speed_factor(
        args.free_speed_scale,
        args.pot_contact_speed_scale,
        independently_clear=independently_clear,
        upward_support_n=support_n,
    )
    return factor, {
        "pot_clear_of_machine": independently_clear,
        "upward_platform_support_n": support_n,
        "independent_clearance_threshold_n": 0.02,
        "factor": factor,
    }

def stage_speed():
    # Keep liquid tilt/restore angular trajectories at their validated reference rate;
    # apply 2x only to the other contact stages in this speed candidate.
    if allow_pour_tilt or e.stage.startswith("pour_restore_"):
        return args.pour_tilt_speed_scale
    if (e.stage.startswith(("pot_extract_", "pot_return_"))
        or e.stage in {"pot_restore_approach", "pot_back_to_front", "pot_back_to_height",
                       "pot_lateral_return", "pot_back_to_retreat", "pot_back_to_staging"}):
        return pot_speed_decision()[0]
    if e.stage == "seat_contact":
        return args.pot_contact_speed_scale
    return base_stage_speed()

def scaled_gripper(arm, fraction, *, compensate=True):
    target = np.asarray(grasp_geometry(m, d, arm)["T_world_grasp"])
    if e.stage == "pot_release" and arm == "left":
        # Locally validated 2 mm downward tool retreat preserves the seated pot
        # while opening, creating clearance from the support link and handle.
        target = target.copy()
        target[2, 3] -= 0.002
    start = a.get_robot_state().gripper_positions[arm]
    count = max(1, round(100/args.contact_speed_scale))
    for i in range(1, count+1):
        a._grippers({arm: fraction if i==count else start+(fraction-start)*(i/count)})
        if compensate:command(arm,target)
        e.tick()
    e.hold(.6)
e.close_gripper = scaled_gripper

baseline_free_move = e.move_grasp

def slow_free_move(arm, target, *, speed=0.015):
    if arm == "right":
        # Preserve the validated cup extraction: supported contact on the rack
        # base precedes the separate geometric-separation/withdrawal checks.
        # Scale rotation as well as translation, retaining the original endpoint hold.
        first = np.asarray(grasp_geometry(m,d,arm)["T_world_grasp"])
        delta = Rotation.from_matrix(target[:3,:3] @ first[:3,:3].T).as_rotvec()
        factor = stage_speed()
        count = max(1,int(np.ceil(max(np.linalg.norm(target[:3,3]-first[:3,3])/speed,np.linalg.norm(delta)/.1)/factor/.02)))
        for i in range(1,count+1):
            G=first.copy();G[:3,3]+=(target[:3,3]-first[:3,3])*i/count
            G[:3,:3]=Rotation.from_rotvec(delta*i/count).as_matrix() @ first[:3,:3]
            command(arm,G);e.tick()
        e.hold(.4)
        actual=np.asarray(grasp_geometry(m,d,arm)["T_world_grasp"])
        e.parameters.setdefault("right_endpoint_diagnostics",[]).append({"stage":e.stage,"time_s":float(d.time),"position_error_m":float(np.linalg.norm(actual[:3,3]-target[:3,3])),"orientation_error_rad":float(Rotation.from_matrix(target[:3,:3] @ actual[:3,:3].T).magnitude()),"acceptance":"original baseline object predicates; cup_move retains2mm cup-position requirement"})
        return
    first = np.array(grasp_geometry(m, d, arm)["T_world_grasp"])
    delta = Rotation.from_matrix(target[:3, :3] @ first[:3, :3].T).as_rotvec()
    count = max(
        1,
        int(
            np.ceil(
                max(
                    np.linalg.norm(target[:3, 3] - first[:3, 3]) / (min(speed, 0.005)*stage_speed()),
                    np.linalg.norm(delta) / (0.04*stage_speed()),
                )
                / 0.02
            )
        ),
    )
    for i in range(1, count + 1):
        G = first.copy()
        G[:3, 3] += (target[:3, 3] - first[:3, 3]) * i / count
        G[:3, :3] = Rotation.from_rotvec(delta * i / count).as_matrix() @ first[:3, :3]
        command(arm, G)
        e.tick()
    stable = 0
    for _ in range(150):
        command(arm, target)
        e.tick()
        actual = np.array(grasp_geometry(m, d, arm)["T_world_grasp"])
        pe = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        re = float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude())
        stable = stable + 1 if pe < 0.002 and re < 0.02 else 0
        if stable >= 10:
            return
    raise RuntimeError(f"free grasp endpoint not reached: {pe=}, {re=}")


e.move_grasp = slow_free_move


pot_body_id = m.body("pot").id
geom_names = [m.geom(i).name for i in range(m.ngeom)]
geom_body_names = [m.body(int(m.geom_bodyid[i])).name for i in range(m.ngeom)]


def advance(seconds):
    global last_guard_log
    if m.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
        raise RuntimeError("fresh-stage optimization validated only for implicitfast")
    if mujoco.get_mjcb_control() is not None:
        raise RuntimeError("fresh-stage optimization forbids control callbacks")
    # Refresh dependencies after any control-boundary payload/option changes.
    mujoco.mj_step1(m, d)
    for _ in range(round(seconds / m.opt.timestep)):
        mujoco.mj_checkPos(m, d)
        mujoco.mj_checkVel(m, d)
        mujoco.mj_step2(m, d)
        mujoco.mj_forward(m, d)
        lifecycle.verify_physics_step()
        check_release_load()
        if (
            not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
            or any(w.number for w in d.warning)
        ):
            raise RuntimeError("nonfinite state or simulator warning")
        if d.time - a._last_evaluation_s >= 0.02 - 1e-9:
            a.task_evidence = a.task_evaluator.update()
            a._last_evaluation_s = float(d.time)
            support_rows.append(
                {
                    "time_s": float(d.time),
                    "stage": e.stage,
                    **a.task_evidence["rack_support_diagnostics"],
                    "inventory":a.task_evidence["rack_inventory_support_diagnostics"],
                }
            )
        if cup_separation_monitor:
            for i, c in enumerate(d.contact[: d.ncon]):
                bodies = [geom_body_names[g] for g in [c.geom1, c.geom2]]
                if active_cup in bodies and any(
                    n.startswith("cup_") and n not in {active_cup, "cup_dispenser"} for n in bodies
                ):
                    f = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, f)
                    if f[0] > 0.02:
                        raise RuntimeError("remaining cup contacted during withdrawal")
        if monitor_on:
            forces = {"left_left_pad_contact": 0.0, "left_right_pad_contact": 0.0}
            contacts = []
            for i, c in enumerate(d.contact[: d.ncon]):
                if pot_body_id not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                    continue
                ns = [geom_names[g] for g in [c.geom1, c.geom2]]
                bs = [geom_body_names[g] for g in [c.geom1, c.geom2]]
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, f)
                if any(n.startswith("handle_collision_") for n in ns):
                    for n in ns:
                        if n in forces:
                            forces[n] += max(0.0, float(f[0]))
                contacts.append(
                    dict(
                        body1=bs[0],
                        body2=bs[1],
                        geom1=ns[0],
                        geom2=ns[1],
                        normal_force_n=float(f[0]),
                    )
                )
            outlet_joint = m.joint("outlet_compression")
            outlet_q = float(d.qpos[outlet_joint.qposadr[0]])
            outlet_v = float(d.qvel[outlet_joint.dofadr[0]])
            force_entry = {
                    "time_s": float(d.time),
                    "stage": e.stage,
                    "pad_forces_n": list(forces.values()),
                    "outlet_compression_m": outlet_q,
                    "outlet_velocity_m_s": outlet_v,
                    "outlet_elastic_force_n": float(
                        m.jnt_stiffness[outlet_joint.id]
                        * (outlet_q - m.qpos_spring[outlet_joint.qposadr[0]])
                    ),
                    "outlet_damping_force_n": float(
                        m.dof_damping[outlet_joint.dofadr[0]] * outlet_v
                    ),
                    "left_drive_torque_nm": float(
                        d.actuator_force[m.actuator("left_gripper_servo").id]
                    ),
            }
            bearing_loads = upper_handle_loads(m, d, e.config["pot"])
            effective_loads = [
                load if inward > 0.02 else 0.0
                for load, inward in zip(
                    bearing_loads["finger_normal_resultants_n"],
                    bearing_loads["opposing_closing_forces_n"],
                )
            ]
            force_entry["upper_handle_loads"] = bearing_loads
            force_entry["effective_grasp_loads_n"] = effective_loads
            if not force_rows or d.time-force_rows[-1]["time_s"] >= 0.001-1e-10:
                force_rows.append(force_entry)
            e.parameters["maximum_finger_normal_resultant_n_all_steps"] = max(
                e.parameters.get("maximum_finger_normal_resultant_n_all_steps",0),
                max(bearing_loads["finger_normal_resultants_n"]))
            fixed = prohibited_pot_contacts(contacts)
            metrics = monitor.update(float(d.time), relative(), effective_loads)
            load_metrics = load_monitor.update(
                float(d.time), bearing_loads["finger_normal_resultants_n"]
            )
            metrics["load_diagnostic"] = load_metrics
            metrics["upper_handle_loads"] = bearing_loads
            metrics["failures"].extend(load_metrics["failures"])
            tilt = float(np.rad2deg(np.arccos(np.clip(d.body("pot").xmat[8], -1, 1))))
            linkage = max(map(abs, gripper_linkage_errors(m, d, "left").values()))
            failures = list(metrics["failures"])
            if bearing_loads["maximum_contact_normal_n"] > load_limits.peak_n:
                failures.append("single_contact_peak_load")
            raw_unilateral_failures = list(failures)
            outlet_touch = any(
                {c["body1"], c["body2"]} == {"pot", "spring_outlet"} and c["normal_force_n"] > 0.02
                for c in contacts
            )
            failures, permitted = outlet_allowance.apply(
                failures,
                extracting=e.stage.startswith("pot_extract_"),
                outlet_contact=outlet_touch,
                platform_support_n=upward_platform_support(),
                outlet_clear=a.task_evidence["evidence"]["pot_clear_of_outlet"].value is True,
                unilateral_s=metrics["unilateral_unload_s"],
                bilateral_s=metrics["bilateral_unload_s"],
            )
            metrics["operator_permitted_unilateral_flags"] = permitted
            metrics["raw_failures"] = raw_unilateral_failures
            metrics["outlet_contact"] = outlet_touch
            if permitted:
                e.parameters["permitted_unilateral_samples"] = (
                    e.parameters.get("permitted_unilateral_samples", 0) + 1
                )

            if fixed:
                failures.append("loaded_fixed_housing")
            if e.stage.startswith(("pot_extract_", "pot_return_")) and outlet_allowance.disengaged:
                axis_error = handle_axis_to_seat_error(pose(), reference_seat[:3, 3])
                metrics["axis_to_initial_center_error_deg"] = axis_error
                if axis_error is not None and axis_error > 12:
                    failures.append("handle_axis_not_toward_initial_center")
            if allow_pour_tilt:
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    m, d, mujoco.mjtObj.mjOBJ_BODY, pot_body_id, velocity, 0
                )
                angular_speed = float(np.rad2deg(np.linalg.norm(velocity[:3])))
                e.parameters["maximum_pour_angular_speed_deg_s"] = max(
                    e.parameters.get("maximum_pour_angular_speed_deg_s", 0), angular_speed
                )
                if (not velocity_rows or d.time-velocity_rows[-1]["time_s"] >= 0.001-1e-10
                    or angular_speed > args.pour_angular_limit_deg_s):
                    velocity_rows.append({"time_s": float(d.time), "stage": e.stage,
                                          "angular_speed_deg_s": angular_speed,
                                          "pot_velocity_world": velocity.tolist()})
                if angular_speed > args.pour_angular_limit_deg_s:
                    failures.append("pour_angular_speed")
            if tilt > (47 if allow_pour_tilt else 8):
                failures.append("pot_tilt_envelope")
            if linkage > 0.005:
                failures.append("gripper_linkage")
            if (
                reference_seat is not None
                and (planned_pot_z if planned_pot_z is not None else reference_seat[2, 3])
                - pose()[2, 3]
                > 0.003
            ):
                failures.append("pot_drop_3mm")
            # Full robot contact check every control tick; additionally catch torso contact every physics step.
            for i, c in enumerate(d.contact[: d.ncon]):
                if torso_geom_id not in (c.geom1, c.geom2):
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, force)
                if c.dist < -0.0001 or force[0] > 0.02:
                    failures.append("torso_contact")
                    break
            if failures or d.time - last_guard_log >= 0.01 - 1e-9:
                guard_rows.append(
                    {
                        "time_s": float(d.time),
                        "stage": e.stage,
                        "grasp_epoch": grasp_epoch,
                        "metrics": metrics,
                        "tilt_deg": tilt,
                        "upward_platform_support_n": upward_platform_support(),
                        "pot_position_m": d.body("pot").xpos.tolist(),
                        "linkage_error_rad": linkage,
                        "fixed_housing_contacts": fixed,
                        "failures": failures,
                    }
                )
                last_guard_log = float(d.time)
            if failures:
                if force_rows[-1] is not force_entry:
                    force_rows.append(force_entry)
                raise RuntimeError("pot guard " + json.dumps(guard_rows[-1]))
        if not monitor_on and reference_seat is not None:
            # Deliberate release ends holding-only requirements, not housing checks.
            for i, c in enumerate(d.contact[: d.ncon]):
                ns = [geom_names[g] for g in [c.geom1, c.geom2]]
                bs = [geom_body_names[g] for g in [c.geom1, c.geom2]]
                if set(bs) == {"pot", "coffee_machine"} and not is_machine_support_geom(
                    ns[bs.index("coffee_machine")]
                ):
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(m, d, i, wrench)
                    if wrench[0] > 0.02:
                        raise RuntimeError(
                            "fixed housing contact after intentional release " + str(ns)
                        )
            if d.body("pot").xmat[8] < np.cos(np.deg2rad(8)):
                raise RuntimeError("pot tilt after intentional release")
    mujoco.mj_forward(m, d)


a._advance = advance


def require(keys):
    a.task_evidence = a.task_evaluator.update()
    missing = [k for k in keys if a.task_evidence["evidence"][k].value is not True]
    if (
        "remaining_cups_retained" in keys
        and not a.task_evidence["rack_support_diagnostics"]["all_supported_now"]
    ):
        missing.append("fresh_support_required_at_stage_boundary")
    if "rack_inventory_supported" in keys and not a.task_evidence["rack_inventory_support_diagnostics"]["all_supported_now"]:
        missing.append("fresh_inventory_support_required_at_stage_boundary")
    if "cup_supported_by_table" in keys and cup_footprint_margin() < 0.02:
        missing.append("cup_bottom_footprint_not_inside_table_with_20mm_margin")
    if missing:
        raise RuntimeError("missing object relations " + str(missing))


unassisted_tick = e.tick


def raw_tick():
    global left_commanded_this_tick
    if (monitor_on and last_planned_pose is not None and not left_commanded_this_tick
            and a.task_evidence["evidence"]["pot_clear_of_outlet"].value is True
            and upward_platform_support() <= .02):
        if m.opt.timestep > args.held_contact_timestep:
            e.parameters.setdefault("precision_transitions",[]).append({
                "time_s":float(d.time),"stage":e.stage,
                "old_timestep_s":float(m.opt.timestep),
                "new_timestep_s":args.held_contact_timestep,
                "reason":"held pot contact precision while other arm operates"})
            m.opt.timestep=args.held_contact_timestep
        actual = pose()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,pot_body_id,velocity,0)
        G, position, angular = stationary_grasp_target(
            actual,last_planned_pose,held,feedback,angular_feedback,velocity[3:],
            angular_limit_deg=air_correction_limit(),loaded_pd=args.loaded_pd)
        feedback[:]=position;angular_feedback[:]=angular
        command("left",G)
        e.parameters["held_pot_stationkeeping_samples"]=e.parameters.get("held_pot_stationkeeping_samples",0)+1
        e.parameters["held_pot_stationkeeping_max_error_m"]=max(e.parameters.get("held_pot_stationkeeping_max_error_m",0),float(np.linalg.norm(actual[:3,3]-last_planned_pose[:3,3])))
    try:
        unassisted_tick()
    finally:
        left_commanded_this_tick=False


def supervised_tick():
    try:
        raw_tick()
        if "cup_supported_by_table" in e.required_relations and cup_footprint_margin() < 0.02:
            raise RuntimeError("placed cup footprint margin fell below 20 mm")
    except RuntimeError as exc:
        import ast

        prefix = "previously verified relation lost: "
        if not str(exc).startswith(prefix):
            raise
        lost = set(ast.literal_eval(str(exc)[len(prefix) :]))
        support = a.task_evidence["rack_support_diagnostics"]["cups"]
        inventory = a.task_evidence["rack_inventory_support_diagnostics"]["cups"]
        members = rack_settle_members(lost, support, inventory)
        if members is None:
            raise
        events = e.parameters.setdefault("rack_settle_recoveries", [])
        saved = e.stage
        if len(events) >= 10 or sum(x["parent_stage"] == saved for x in events) >= 3:
            raise RuntimeError("rack settling recovery budget exhausted") from exc
        kinematics = {}
        for name in members:
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body(name).id, v, 0)
            kinematics[name] = {
                "linear_m_s": float(np.linalg.norm(v[3:])),
                "angular_rad_s": float(np.linalg.norm(v[:3])),
                "drift_m": float(
                    np.linalg.norm(d.body(name).xpos - a.task_evaluator.initial[name])
                ),
                "tilt_deg": float(np.rad2deg(np.arccos(np.clip(d.body(name).xmat[8], -1, 1)))),
            }
            if kinematics[name]["drift_m"] >= 0.015 or kinematics[name]["tilt_deg"] >= 8:
                raise
        events.append(
            {
                "time_s": float(d.time),
                "parent_stage": saved,
                "reason": str(exc),
                "support_at_pause": support,
                "inventory_at_pause": inventory,
                "monitored_cups": list(members),
                "kinematics": kinematics,
            }
        )
        e.stage = saved + "_rack_settle"
        e.required_relations -= lost
        for _ in range(25):
            raw_tick()
            for name in members:
                if np.linalg.norm(
                    d.body(name).xpos - a.task_evaluator.initial[name]
                ) >= 0.015 or d.body(name).xmat[8] <= np.cos(np.deg2rad(8)):
                    raise RuntimeError(
                        "remaining cup moved outside retained envelope during settle"
                    )
            current_members = rack_settle_members(
                lost, a.task_evidence["rack_support_diagnostics"]["cups"],
                a.task_evidence["rack_inventory_support_diagnostics"]["cups"])
            if current_members != members:
                raise RuntimeError("remaining cup lost support during settle")
        require(lost)
        e.required_relations |= lost
        e.stage = saved
        events[-1]["verified_time_s"] = float(d.time)


e.tick = supervised_tick

feedback = np.zeros(3)
angular_feedback = np.zeros(3)
held = None
last_planned_pose = None

extra_suspension_m = 0.0
if args.phase == "loaded_return":
    context = resume_meta.get("grasp_context")
    if not context or hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest() != resume_meta.get("grasp_context_sha256"):
        raise ValueError("loaded-return grasp context checksum mismatch")
    grasp_epoch = int(context["grasp_epoch"])
    held = np.asarray(context["held_relative"], dtype=float)
    reference_seat = np.asarray(context["seat_pose"], dtype=float)
    local_h = np.asarray(context["handle_local_m"], dtype=float)
    feedback = np.asarray(context["feedback_m"], dtype=float)
    angular_feedback = np.asarray(context["angular_feedback_rad"], dtype=float)
    load_ff_start = context["load_ff_start"]
    last_planned_pose = np.asarray(context["last_planned_pose"], dtype=float)
    extra_suspension_m = float(context["extra_suspension_m"])
    use_posture = bool(context["use_posture"])
    monitor = HandleGraspMonitor(held, local_h, HandleGraspLimits(**context["grasp_limits"]))
    monitor.last_time = context["grasp_timers"]["last_time"]
    monitor.unilateral_since = context["grasp_timers"]["unilateral_since"]
    monitor.bilateral_since = context["grasp_timers"]["bilateral_since"]
    load_monitor = PadLoadMonitor(load_limits)
    load_monitor.last_time = context["load_timers"]["last_time"]
    load_monitor.over_since = context["load_timers"]["over_since"]
    load_monitor.asymmetric_since = context["load_timers"]["asymmetric_since"]
    outlet_allowance.disengaged = bool(context["outlet_disengaged"])
    airborne_since = context["airborne_since_s"]
    low_payload_restore_since = context["low_payload_restore_since_s"]
    monitor_on = True
    right_load_on = False


def clearance_lift(radius, base_mm):
    # Follow the support profile as the leading pot edge climbs out of the recess.
    return (
        support_height(e.config["machine"], e.config["pot"]["radius_m"] + radius)
        - nominal_seated_height(e.config["machine"], e.config["pot"])
        + extra_suspension_m
    )


def upward_platform_support():
    force = 0.0
    pot_id = m.body("pot").id
    for i, c in enumerate(d.contact[: d.ncon]):
        bodies = [m.geom_bodyid[g] for g in (c.geom1, c.geom2)]
        if pot_id not in bodies:
            continue
        side = bodies.index(pot_id)
        other = (c.geom1, c.geom2)[1 - side]
        if not is_machine_support_geom(m.geom(other).name):
            continue
        # Only upward contact near the pot base establishes bottom support.
        upward = float(c.frame[2]) * (1 if side == 1 else -1)
        if upward < 0.9 or abs(float(c.pos[2] - d.body("pot").xpos[2])) > 0.004:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        force += max(0.0, float(f[0])) * upward
    return force


def sliding_stage():
    return e.stage.startswith(("pot_extract_", "pot_return_")) or e.stage in {
        "slide_height_transition",
        "seat_contact",
    }


def slide_check(target, velocity):
    actual = pose()
    yaw = lambda T: np.arctan2(T[1, 0], T[0, 0])
    yaw_error = abs(
        float(
            np.rad2deg(
                np.arctan2(np.sin(yaw(target) - yaw(actual)), np.cos(yaw(target) - yaw(actual)))
            )
        )
    )
    return supported_slide_settled(
        position_error_m=float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
        yaw_error_deg=yaw_error,
        tilt_deg=float(np.rad2deg(np.arccos(np.clip(actual[2, 2], -1, 1)))),
        linear_speed_m_s=float(np.linalg.norm(velocity[3:])),
        angular_speed_rad_s=float(np.linalg.norm(velocity[:3])),
        upward_support_n=upward_platform_support(),
        max_tilt_deg=args.supported_slide_tilt_deg,
        max_yaw_error_deg=args.slide_yaw_tolerance_deg,
    )


def air_correction_limit():
    # Do not level against the supporting platform: prove separation first,
    # then ramp controller authority. Fixed grasp and load limits are unchanged.
    global airborne_since, low_payload_restore_since
    base = min(2.0, args.air_correction_deg)
    if not outlet_allowance.disengaged or upward_platform_support() > 0.02:
        airborne_since = None
        return bumpless_rotation_limit(angular_feedback, requested_deg=base)
    if airborne_since is None:
        airborne_since = float(d.time)
    elapsed = max(0.0, float(d.time) - airborne_since - 0.5)
    requested = min(args.air_correction_deg, base + 0.5 * elapsed)
    low_payload = transfer.pot_volume_m3 < transfer.cavities[active_cup].capacity_m3
    if low_payload and e.stage.startswith("pour_restore_"):
        if low_payload_restore_since is None:
            low_payload_restore_since = float(d.time)
        requested = min(args.low_payload_restore_correction_deg,
                        args.air_correction_deg + .5 * (float(d.time) - low_payload_restore_since))
    else:
        low_payload_restore_since = None
    # Independent grasp/contact/load/velocity limits are unchanged; do not reset held.
    return bumpless_rotation_limit(angular_feedback, requested_deg=requested)


def object_move(target, speed=0.008, settle=3):
    global planned_pot_z, last_planned_pose, airborne_since
    if allow_pour_tilt and args.pour_timestep is not None and m.opt.timestep != args.pour_timestep:
        e.parameters.setdefault("precision_transitions",[]).append({
            "time_s":float(d.time),"stage":e.stage,"old_timestep_s":float(m.opt.timestep),
            "new_timestep_s":args.pour_timestep,"reason":"resolved loaded-pour contact impacts"})
        m.opt.timestep = args.pour_timestep
    if (args.return_timestep is not None
        and (e.stage.startswith("pot_return_") or e.stage == "seat_contact")
        and np.linalg.norm(target[:2, 3] - reference_seat[:2, 3]) <= 0.031
        and m.opt.timestep != args.return_timestep):
        e.parameters.setdefault("precision_transitions", []).append({
            "time_s": float(d.time), "stage": e.stage,
            "old_timestep_s": float(m.opt.timestep),
            "new_timestep_s": args.return_timestep,
            "reason": "fine contact integration for final seating and release"})
        m.opt.timestep = args.return_timestep
    if args.transport_contact_timestep is not None and base_stage_speed()>args.contact_speed_scale and m.opt.timestep!=args.transport_contact_timestep:
        e.parameters.setdefault("precision_transitions",[]).append({"time_s":float(d.time),"stage":e.stage,"old_timestep_s":float(m.opt.timestep),"new_timestep_s":args.transport_contact_timestep,"reason":"gripper-handle contact precision during accelerated transport"})
        m.opt.timestep=args.transport_contact_timestep
    if args.extraction_contact_timestep is not None and e.stage.startswith("pot_extract_"):
        desired_step = args.extraction_contact_timestep if np.linalg.norm(target[:2,3]-reference_seat[:2,3])<=.060001 else (args.transport_timestep or .001)
        if m.opt.timestep != desired_step:
            e.parameters.setdefault("precision_transitions",[]).append({"time_s":float(d.time),"stage":e.stage,"old_timestep_s":float(m.opt.timestep),"new_timestep_s":desired_step,"reason":"outlet extraction contact precision"})
            m.opt.timestep=desired_step
    first = pose() if last_planned_pose is None else last_planned_pose.copy()
    e.parameters.setdefault("reference_segments", []).append(
        {
            "stage": e.stage,
            "measured_start": pose().tolist(),
            "planned_start": first.tolist(),
            "target": target.tolist(),
        }
    )
    angle = Rotation.from_matrix(target[:3, :3] @ first[:3, :3].T).as_rotvec()
    n = max(
        1,
        int(
            np.ceil(
                max(
                    np.linalg.norm(target[:3, 3] - first[:3, 3]) / speed,
                    np.linalg.norm(angle) / (args.pour_angular_rate if allow_pour_tilt else 0.04),
                )
                / 0.02
            )
        ),
    )
    factor = stage_speed()
    timing = MotionTiming(n*.02,factor)
    n = max(1, int(np.ceil(timing.duration_seconds/.02)))
    speed_segment = {"stage": e.stage, "factor": factor, "planned_ticks": n,
                     "ramp_seconds":timing.ramp_seconds, "reference_peak_factor":factor}
    if e.stage.startswith(("pot_extract_", "pot_return_")):
        speed_segment["pot_transport_speed_decision"] = pot_speed_decision()[1]
    e.parameters.setdefault("speed_segments", []).append(speed_segment)
    stable_ticks = 0
    for step in range(1, n + 1 + round(settle / 0.02)):
        if step % 50 == 0:
            print(json.dumps({"progress_time_s": float(d.time), "stage": e.stage,
                              "elapsed_control_ticks": step, "planned_ticks": n}), flush=True)
        t = timing.fraction(min(1.,step/n)*timing.duration_seconds)
        P = first.copy()
        P[:3, 3] += (target[:3, 3] - first[:3, 3]) * t
        P[:3, :3] = Rotation.from_rotvec(angle * t).as_matrix() @ first[:3, :3]
        if e.stage.startswith(("pot_extract_", "pot_return_")):
            radius_now = float(np.linalg.norm(P[:2, 3] - reference_seat[:2, 3]))
            P[2, 3] = reference_seat[2, 3] + clearance_lift(radius_now, 0)
        last_planned_pose = P.copy()
        planned_pot_z = float(P[2, 3])
        feedback[:] += (0.01 if args.loaded_pd else 0.04) * (P[:3, 3] - pose()[:3, 3])
        feedback[:] *= min(1.0, 0.02 / max(np.linalg.norm(feedback), 1e-12))
        platform_force = upward_platform_support()
        if platform_force > 0.02:
            airborne_since = None
        if platform_force > 0.02 and (
            sliding_stage() or e.stage in {"load_transfer", "return_load_transfer"}
        ):
            yaw_error = np.arctan2(P[1, 0], P[0, 0]) - np.arctan2(pose()[1, 0], pose()[0, 0])
            yaw_error = float(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))
            if abs(yaw_error) < np.deg2rad(args.slide_yaw_tolerance_deg / 2):
                yaw_error = 0.0
            if args.supported_level_correction_deg and e.stage.startswith("pot_extract_"):
                level_error = Rotation.from_matrix(P[:3, :3] @ pose()[:3, :3].T).as_rotvec()
                old_yaw = angular_feedback[2]
                angular_feedback[:] = bounded_rotation_integral(
                    angular_feedback, level_error, maximum_deg=args.supported_level_correction_deg
                )
                angular_feedback[2] = old_yaw
            angular_feedback[:] = supported_yaw_integral(
                angular_feedback,
                yaw_error,
                maximum_deg=args.return_yaw_max_deg
                if e.stage.startswith("pot_return_") or e.stage == "seat_contact"
                else args.extraction_yaw_max_deg,
            )
        else:
            angular_feedback[:] = bounded_rotation_integral(
                angular_feedback,
                Rotation.from_matrix(P[:3, :3] @ pose()[:3, :3].T).as_rotvec(),
                maximum_deg=air_correction_limit()
                if a.task_evidence["evidence"]["pot_clear_of_outlet"].value is True
                else bumpless_rotation_limit(angular_feedback, requested_deg=1.0),
            )
        G = P @ np.linalg.inv(held)
        G[:3, 3] += feedback
        G[:3, :3] = Rotation.from_rotvec(angular_feedback).as_matrix() @ G[:3, :3]
        if args.loaded_pd:
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("pot").id, velocity, 0)
            G[:3, 3] += 0.8 * (P[:3, 3] - pose()[:3, 3]) - 0.1 * velocity[3:]
        command("left", G)
        e.tick()
        actual = pose()
        yaw_now = float(
            np.rad2deg(
                np.arctan2(
                    np.sin(np.arctan2(P[1, 0], P[0, 0]) - np.arctan2(actual[1, 0], actual[0, 0])),
                    np.cos(np.arctan2(P[1, 0], P[0, 0]) - np.arctan2(actual[1, 0], actual[0, 0])),
                )
            )
        )
        control_rows.append(
            {
                "time_s": float(d.time),
                "stage": e.stage,
                "yaw_error_deg": yaw_now,
                "angular_feedback_deg": np.rad2deg(angular_feedback).tolist(),
                "position_error_m": float(np.linalg.norm(actual[:3, 3] - P[:3, 3])),
            }
        )
        if args.adaptive_settle and step > n:
            pe = np.linalg.norm(pose()[:3, 3] - target[:3, 3])
            re = Rotation.from_matrix(target[:3, :3] @ pose()[:3, :3].T).magnitude()
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("pot").id, velocity, 0)
            precise = (
                pe < 0.002
                and re < np.deg2rad(0.5)
                and np.linalg.norm(velocity[3:]) < 0.003
                and np.linalg.norm(velocity[:3]) < 0.03
            )
            supported = sliding_stage() and slide_check(target, velocity)
            stable_ticks = stable_ticks + 1 if precise or supported else 0
            if e.stage == "prelift":
                support = sum(
                    x["normal_force_n"]
                    for x in e.contacts()
                    if "hotplate" in [x["geom1"], x["geom2"]]
                    and "pot_bottom" in [x["geom1"], x["geom2"]]
                )
                if support > 0.02 or abs(pose()[2, 3] - target[2, 3]) > 0.0001:
                    stable_ticks = 0
            if (
                e.stage == "seat_contact"
                and a.task_evidence["evidence"]["pot_on_hotplate"].value is not True
            ):
                stable_ticks = 0
            if stable_ticks >= 25:
                break
    err = {
        "position_m": float(np.linalg.norm(pose()[:3, 3] - target[:3, 3])),
        "orientation_deg": float(
            np.rad2deg(Rotation.from_matrix(target[:3, :3] @ pose()[:3, :3].T).magnitude())
        ),
    }
    e.parameters.setdefault("waypoint_errors", []).append(
        {
            "stage": e.stage,
            **err,
            "settle_elapsed_s": max(0, step - n) * 0.02,
            "feedback_m": feedback.tolist(),
            "angular_feedback_deg": np.rad2deg(angular_feedback).tolist(),
            "upward_platform_support_n": upward_platform_support(),
            "supported_slide_accepted": bool(supported) if args.adaptive_settle else False,
        }
    )
    if args.adaptive_settle and stable_ticks < 25:
        raise RuntimeError("waypoint did not sustain the required stability window")
    if (err["position_m"] > 0.002 or err["orientation_deg"] > 0.5) and not (
        args.adaptive_settle and supported
    ):
        raise RuntimeError("object waypoint error " + json.dumps(err))


def finish_cup_extraction():
    global cup_separation_monitor
    vertical = selected_cup_withdraw_clearance(d,e.config,active_cup,a.task_evaluator.cups[1:])
    surface = cup_surface_gap()
    e.parameters.update(
        {
            "cup_clearance_m": float(vertical),
            "cup_surface_gap_before_withdraw_m": surface,
            "cup_release_check": "exact_convex_surface_3mm_plus_per_step_remaining_cup_contact_guard",
            "dispenser_base_support_allowed_during_withdrawal": True,
        }
    )
    if surface < 0.003:
        raise RuntimeError("cup not clear of remaining cup surfaces")
    require(["cup_in_right_gripper", "remaining_cups_retained"])
    e.required_relations |= {"cup_in_right_gripper", "remaining_cups_retained"}
    cup_separation_monitor = True
    e.stage = "cup_withdraw"
    T = np.array(grasp_geometry(m, d, "right")["T_world_grasp"])
    T[:3, 3] += [-0.12, 0, 0]
    e.move_grasp("right", T)
    e.hold(0.5)
    require(cup_keys)
    e.required_relations |= set(cup_keys)
    cup_separation_monitor = False
    e.stage = "cup_hold"
    e.hold(3)
    e.save_checkpoint(e.output / "cup_held.npz")


def supported_regrasp():
    global \
        monitor_on, \
        load_ff_start, \
        held, \
        local_h, \
        monitor, \
        load_monitor, \
        last_planned_pose, \
        use_posture, \
        grasp_epoch
    e.stage = "regrasp_seat"
    T = pose()
    T[:3, :3] = Rotation.from_euler("z", heading).as_matrix()
    T[2, 3] = reference_seat[2, 3] + clearance_lift(args.regrasp_radius_mm / 1000, 0)
    object_move(T, speed=0.0005, settle=args.settle)
    e.stage = "supported_regrasp_open"
    monitor_on = False
    load_ff_start = None
    command("left", np.array(grasp_geometry(m, d, "left")["T_world_grasp"]))
    e.parameters["release_opening_fraction"] = args.release_opening
    e.close_gripper("left", args.release_opening)
    e.hold(2)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body("pot").id, velocity, 0)
    if upward_platform_support() < 0.02 or np.linalg.norm(velocity[3:]) > 0.003:
        raise RuntimeError("regrasp requires stable independent pot support")
    e.stage = "supported_regrasp_withdraw"
    withdraw = np.array(grasp_geometry(m, d, "left")["T_world_grasp"])
    withdraw[:3, 3] -= 0.04 * withdraw[:3, 2]
    e.move_grasp("left", withdraw, speed=0.005)
    e.parameters["release_opening_fraction"] = args.release_opening
    e.close_gripper("left", args.release_opening)
    e.save_checkpoint(e.output / "regrasp_pre.npz")
    use_posture = True
    e.stage = "supported_regrasp_approach"
    target = pot_handle_grasp_pose(d)
    roll = -np.arctan2(
        e.config["pot"]["handle_bottom_inset_m"], e.config["pot"]["handle_outer_height_m"]
    )
    target[:3, :3] = target[:3, :3] @ Rotation.from_euler("z", roll).as_matrix()
    target[2, 3] += args.regrasp_offset_mm / 1000
    target[:3, :3] = (
        target[:3, :3] @ Rotation.from_euler("x", args.regrasp_roll_deg, degrees=True).as_matrix()
    )
    pre = target.copy()
    pre[:3, 3] -= 0.04 * pre[:3, 2]
    e.move_grasp("left", pre, speed=0.005)
    e.move_grasp("left", target, speed=0.005)
    grasp_epoch += 1
    e.stage = "supported_regrasp_close"
    e.close_gripper("left", 0.0)
    e.check_pot_pinch()
    held = relative()
    local_h = pose()[:3, :3].T @ (
        np.array(grasp_geometry(m, d, "left")["T_world_grasp"])[:3, 3] - pose()[:3, 3]
    )
    monitor = HandleGraspMonitor(
        held, local_h, HandleGraspLimits(relative_rotation_deg=args.diagnostic_grasp_angle_deg)
    )
    monitor_on = True
    load_monitor = PadLoadMonitor(load_limits)
    load_ff_start = float(d.time)
    feedback[:] = 0
    angular_feedback[:] = 0
    last_planned_pose = pose()
    e.parameters.update(
        {
            "deliberate_supported_regrasp": True,
            "regrasp_offset_mm": args.regrasp_offset_mm,
            "regrasp_roll_deg": args.regrasp_roll_deg,
            "regrasp_radius_mm": args.regrasp_radius_mm,
            "held_relative": held.tolist(),
            "grasp_limits": asdict(monitor.limits),
            "original_anchor_source": str(e.output / "result.json"),
        }
    )
    e.parameters.setdefault("grasp_events", []).append(
        {
            "epoch": grasp_epoch,
            "time_s": float(d.time),
            "radius_mm": args.regrasp_radius_mm,
            "roll_deg": args.regrasp_roll_deg,
            "held_relative": held.tolist(),
        }
    )
    e.parameters["active_grasp_checkpoint"] = f"regrasp_pinch_{grasp_epoch:02d}.npz"
    e.save_checkpoint(e.output / e.parameters["active_grasp_checkpoint"])
    if not (e.output / "pot_pinch.npz").exists():
        e.save_checkpoint(e.output / "pot_pinch.npz")


cup_relative = None


def cup_footprint_margin():
    center = np.array(e.config["table"]["center_xy_m"])
    half = np.array(e.config["table"]["size_m"][:2]) / 2
    return float(
        np.min(half - np.abs(pose(active_cup)[:2, 3] - center)) - e.config["cup"]["bottom_radius_m"]
    )


def cup_move(C):
    e.move_grasp("right", C @ np.linalg.inv(cup_relative), speed=0.015)
    error = float(np.linalg.norm(pose(active_cup)[:3, 3] - C[:3, 3]))
    if error > 0.002:
        # A bounded IK fallback is an intermediate command, not endpoint success.
        # Retain the requested object/grasp target and original 2 mm criterion.
        initial_error = error
        stable = 0
        target = C @ np.linalg.inv(cup_relative)
        for correction in range(1, 1 + round(args.cup_endpoint_max_seconds/.02)):
            command("right", target)
            e.tick()
            error = float(np.linalg.norm(pose(active_cup)[:3, 3] - C[:3, 3]))
            stable = stable + 1 if error <= 0.002 else 0
            if stable >= 20:
                break
        e.parameters.setdefault("cup_endpoint_settling", []).append({
            "stage":e.stage,"initial_error_m":initial_error,"final_error_m":error,
            "correction_ticks":correction,"stable_ticks":stable,"target":C.tolist(),
            "guard_limits_changed":False})
        if stable < 20:
            raise RuntimeError("cup endpoint did not settle within 2 mm")
    if error > 0.002:
        raise RuntimeError("cup position error")


def place_cup_with_footprint_margin():
    C = pose(active_cup)
    center = np.array(e.config["table"]["center_xy_m"])
    half = np.array(e.config["table"]["size_m"][:2]) / 2
    clearance = e.config["cup"]["bottom_radius_m"] + 0.035
    slot = len(service.served_cups)
    requested = fixed_delivery_slot(e.config)
    if np.any(np.abs(requested-center)+clearance>half):
        raise RuntimeError("no remaining qualified table slot")
    target = requested
    C[2,3] = max(C[2,3],e.config["table"]["top_height_m"]+.16)
    e.stage = "cup_placement_clearance_raise"
    cup_move(C)
    e.parameters["cup_placement_plan"] = {
        "xy_m": target.tolist(),
        "bottom_radius_m": e.config["cup"]["bottom_radius_m"],
        "planned_footprint_margin_m": 0.035,
        "minimum_accepted_footprint_margin_m": 0.02,
        "previous_footprint_margin_m": cup_footprint_margin(),
    }
    if np.linalg.norm(target - C[:2, 3]) > 0.002:
        e.stage = "cup_table_clearance_lift"
        C[2, 3] = max(C[2, 3], e.config["table"]["top_height_m"] + 0.05)
        cup_move(C)
        e.stage = "cup_table_inset"
        C[:2, 3] = target
        cup_move(C)
    e.stage = "cup_place"
    C[2, 3] = e.config["table"]["top_height_m"] - 0.00005
    cup_move(C)
    e.hold(1)
    e.stage = "cup_support_probe"
    start = np.array(grasp_geometry(m, d, "right")["T_world_grasp"])
    G = start.copy()
    for step in range(1, 11):
        if a.task_evidence["evidence"]["cup_supported_by_table"].value is True:
            break
        G = start.copy()
        G[2, 3] -= step * 0.0001
        e.move_grasp("right", G, speed=0.0005)
        e.hold(0.5)
    e.parameters["cup_support_probe_lowering_m"] = float(start[2, 3] - G[2, 3])
    e.parameters["actual_cup_footprint_margin_m"] = cup_footprint_margin()
    require(["cup_supported_by_table", "cup_upright"])
    e.save_checkpoint(e.output / "cup_supported.npz")


def verify_fixture():
    global anchored, b, b1, b2, c, drift, graph, initial, k, name, row, seen, stack_checks, todo, x, yaw
    e.stage = 'verify_initial_fixture'
    initial = pose()
    e.hold(8)
    require(['pot_on_hotplate', 'lid_outlet_aligned', 'pot_upright'])
    stack_checks = []
    for row in e.trace[-10:]:
        graph = {}
        for c in row['contacts']:
            if c['normal_force_n'] <= 0.001:
                continue
            b1, b2 = [m.body(m.geom_bodyid[m.geom(c[k]).id]).name for k in ['geom1', 'geom2']]
            graph.setdefault(b1, set()).add(b2)
            graph.setdefault(b2, set()).add(b1)
        for name in a.task_evaluator.cups:
            seen = set()
            todo = [name]
            anchored = False
            while todo:
                b = todo.pop()
                if b in seen:
                    continue
                seen.add(b)
                if b.startswith('retention_finger') or b == 'cup_dispenser':
                    anchored = True
                    break
                todo.extend((x for x in graph.get(b, ()) if x in a.task_evaluator.cups or x.startswith('retention_finger') or x == 'cup_dispenser'))
            drift = float(np.linalg.norm(np.array(row['object_positions'][name]) - a.task_evaluator.initial[name]))
            stack_checks.append({'time_s': row['time_s'], 'cup': name, 'anchored': anchored, 'drift_m': drift})
            if not anchored or drift > 0.015:
                raise RuntimeError('initial stack unsupported or displaced ' + str(stack_checks[-1]))
    e.parameters['initial_stack_support_checks'] = stack_checks
    yaw = lambda P: float(np.rad2deg(np.arctan2(P[1, 0], P[0, 0])))
    e.parameters['initial_fixture'] = {'position_drift_m': float(np.linalg.norm(pose()[:3, 3] - initial[:3, 3])), 'yaw_deg': yaw(pose())}
    if np.linalg.norm(pose()[:3, 3] - initial[:3, 3]) > 0.005 or abs(yaw(pose()) - args.nominal_yaw_deg) > args.yaw_tolerance_deg:
        raise RuntimeError('initial pot not stable at requested nominal heading')
    e.save_checkpoint(e.output / 'fixture_stable.npz')

def take_cup():
    global T, active_cup, b, cup_keys, g, name, pair, route, side
    active_cup = next((name for name in transfer.cups if name not in service.served_cups), None)
    if active_cup is None:
        raise RuntimeError('cup rack empty; restocking required')
    e.required_relations -= {'cup_supported_by_table', 'cup_upright', 'right_gripper_released', 'cup_in_right_gripper', 'single_cup_separated', 'remaining_cups_retained', 'rack_inventory_supported'}
    a.task_evaluator.select_cup(active_cup)
    transfer.select_cup(active_cup)
    update_cup_pairs()
    cup_keys = ['cup_in_right_gripper', 'single_cup_separated', 'remaining_cups_retained']
    e.allowed = {pair for pair in e.allowed if not pair[0].startswith('right_')}
    e.stage = 'cup_rejoin_pregrasp'
    route = plan_service_joint_path(a, 'right', ready_right_q, allowed_contacts={(g, b) for g, b in e.allowed if g.startswith('left_')}, seed=53, iterations=3000)
    e.parameters.setdefault('cup_rejoin_paths', []).append(route)
    if not route['found']:
        raise RuntimeError('cannot rejoin cup pregrasp with pot held')
    e.joint_path('right', route['waypoints_urdf_rad'])
    e.stage = 'cup_approach'
    T = np.array(grasp_geometry(m, d, 'right')['T_world_grasp'])
    T[:3, 3] = d.body(active_cup).xpos + [0, 0, 0.035]
    e.move_grasp('right', T)
    e.stage = 'cup_close'
    e.allowed |= {(f'right_{side}_pad_contact', active_cup) for side in ['left', 'right']}
    e.close_gripper('right', 0.2)
    e.stage = 'cup_extract_down'
    T = np.array(grasp_geometry(m, d, 'right')['T_world_grasp'])
    T[:3, 3] += [0, 0, -0.085]
    e.move_grasp('right', T, speed=0.01)
    finish_cup_extraction()

def take_pot():
    global P, T, airborne_since, angular_feedback, centre, centres, cup_keys, event, extra_suspension_m, feedback, grasp_epoch, heading, held, last_planned_pose, load_ff_start, load_monitor, local_h, monitor, monitor_on, outlet_allowance, planned_pot_z, pre, radii, radius, reference_seat, return_centres, right_load_on, side, target
    e.required_relations -= {'pot_on_hotplate', 'pot_stable_after_release', 'lid_outlet_aligned', 'left_gripper_released'}
    cup_keys = ['rack_inventory_supported']
    e.required_relations.add('rack_inventory_supported')
    monitor_on = False
    right_load_on = False
    load_ff_start = None
    last_planned_pose = None
    planned_pot_z = None
    feedback = np.zeros(3)
    angular_feedback = np.zeros(3)
    airborne_since = None
    extra_suspension_m = 0.0
    load_monitor = PadLoadMonitor(load_limits)
    outlet_allowance = OutletUnilateralAllowance(args.allow_outlet_unilateral_ms / 1000)
    grasp_epoch = sum((event['action'] == 'take_pot' for event in service.events))
    retake_step = min(args.physics_timestep or .001, args.held_contact_timestep) if grasp_epoch else (args.physics_timestep or .001)
    if float(m.opt.timestep) != retake_step:
        e.parameters.setdefault('precision_transitions', []).append(dict(
            time_s=float(d.time), stage='pot_retake_precision', old_timestep_s=float(m.opt.timestep),
            new_timestep_s=retake_step, reason='preserve rack contact precision on subsequent pot approach'))
    m.opt.timestep = retake_step
    e.stage = 'pot_reopen_empty_gripper'
    if abs(a.get_robot_state().gripper_positions['left'] - 0.23) > 0.005:
        e.close_gripper('left', 0.23)
    e.parameters['yaw_before_grasp_deg'] = float(np.rad2deg(np.arctan2(pose()[1, 0], pose()[0, 0])))
    if abs(e.parameters['yaw_before_grasp_deg'] - args.nominal_yaw_deg) > args.yaw_tolerance_deg:
        raise RuntimeError('fixture yaw drift before grasp')
    e.stage = 'pot_align'
    target = upper_handle_target(d, e.config['pot'], height_m=args.grasp_height_offset_mm / 1000, side_tilt_deg=args.initial_grasp_roll_deg, radial_m=args.grasp_radial_mm / 1000, depth_m=args.grasp_depth_mm / 1000, inplane_deg=args.grasp_inplane_deg)
    e.parameters['upper_handle_grasp'] = {'height_offset_mm': args.grasp_height_offset_mm, 'side_tilt_deg': args.initial_grasp_roll_deg, 'radial_offset_mm': args.grasp_radial_mm, 'depth_offset_mm': args.grasp_depth_mm, 'inplane_deg': args.grasp_inplane_deg}
    pre = target.copy()
    pre[:3, 3] -= 0.04 * pre[:3, 2]
    e.move_grasp('left', pre, speed=0.01)
    e.stage = 'pot_approach'
    e.move_grasp('left', target, speed=0.008)
    e.stage = 'pot_close'
    e.allowed |= {(f'left_{side}_pad_contact', 'pot') for side in ['left', 'right']}
    e.close_gripper('left', 0.0)
    e.check_pot_pinch()
    if args.extraction_timestep is not None:
        e.parameters.setdefault('precision_transitions',[]).append({'time_s':float(d.time),'stage':e.stage,'old_timestep_s':float(m.opt.timestep),'new_timestep_s':args.extraction_timestep,'reason':'initial extraction contact precision'})
        m.opt.timestep=args.extraction_timestep
    reference_seat = pose()
    e.parameters['yaw_after_grasp_deg'] = float(np.rad2deg(np.arctan2(reference_seat[1, 0], reference_seat[0, 0])))
    if abs(e.parameters['yaw_after_grasp_deg'] - args.nominal_yaw_deg) > args.yaw_tolerance_deg:
        raise RuntimeError('fixture yaw drift during grasp')
    held = relative()
    local_h = pose()[:3, :3].T @ (np.array(grasp_geometry(m, d, 'left')['T_world_grasp'])[:3, 3] - pose()[:3, 3])
    monitor = HandleGraspMonitor(held, local_h, HandleGraspLimits(relative_rotation_deg=args.diagnostic_grasp_angle_deg))
    monitor_on = True
    e.parameters['grasp_limits'] = asdict(monitor.limits)
    e.parameters['seat_pose'] = reference_seat.tolist()
    e.parameters['held_relative'] = held.tolist()
    e.save_checkpoint(e.output / 'pot_pinch.npz')
    if args.payload_feedforward and load_ff_start is None and (args.phase != 'seated'):
        load_ff_start = float(d.time)
    if args.prelift:
        e.stage = 'prelift'
        load_ff_start = float(d.time)
        T = reference_seat.copy()
        T[2, 3] += args.prelift_mm / 1000
        object_move(T, speed=0.0005, settle=args.settle)
        e.save_checkpoint(e.output / 'prelift.npz')
    heading = float(np.arctan2(reference_seat[1, 0], reference_seat[0, 0]))
    radii = [0.18, 0.14, 0.12, 0.1, 0.06, 0.04, 0.03, 0.02, 0.01, 0.0] if args.platform_roundtrip else [0.24, 0.22, 0.2, 0.18, 0.14, 0.12, 0.1, 0.06, 0.04, 0.03, 0.02, 0.01, 0.0]
    centres = clearance_entry_centres(reference_seat[:3, 3], heading if args.travel_heading_deg is None else np.deg2rad(args.travel_heading_deg), radii, detour_span_m=0.04, lateral_offset_m=0.002 * np.sqrt(2))
    return_centres = clearance_entry_centres(reference_seat[:3, 3], heading if args.travel_heading_deg is None else np.deg2rad(args.travel_heading_deg), radii, detour_span_m=0.04, lateral_offset_m=args.return_entry_offset_mm / 1000)
    for radius, centre in list(zip(radii, centres))[-2::-1]:
        if args.phase in {'regrasp', 'regrasped'} and radius <= args.regrasp_radius_mm / 1000 or (args.phase == 'extended' and radius <= 0.18) or (args.phase == 'extraction' and radius <= 0.01) or (args.phase == 'egress' and radius <= 0.06) or (args.phase == 'far' and radius <= 0.1):
            continue
        e.stage = f'pot_extract_{round(radius * 1000):03d}'
        T = reference_seat.copy()
        T[:3, :3] = Rotation.from_euler('z', heading).as_matrix()
        T[:3, 3] = centre
        T[2, 3] += clearance_lift(radius, args.extraction_lift_mm)
        object_move(T, speed=args.extract_speed if radius <= 0.24 else 0.008, settle=args.settle)
        e.save_checkpoint(e.output / (e.stage + '.npz'))
        if radius>=.06 and args.extraction_timestep is not None and a.task_evidence['evidence']['pot_clear_of_outlet'].value is True and m.opt.timestep != (args.transport_timestep or .001):
            e.parameters.setdefault('precision_transitions',[]).append({'time_s':float(d.time),'stage':e.stage,'old_timestep_s':float(m.opt.timestep),'new_timestep_s':args.transport_timestep or .001,'reason':'outlet clearance verified after extraction'})
            m.opt.timestep=args.transport_timestep or .001
        if args.cold_regrasp and (not e.parameters.get('deliberate_supported_regrasp', False)) and (round(radius * 1000) == args.regrasp_radius_mm):
            e.hold(0.4)
            supported_regrasp()
        if radius == 0.18 and args.suspend_height_mm and (extra_suspension_m == 0):
            extra_suspension_m = args.suspend_height_mm / 1000
            T = last_planned_pose.copy()
            T[2, 3] += extra_suspension_m
            e.stage = 'load_transfer'
            object_move(T, speed=0.0005, settle=args.settle)
            e.save_checkpoint(e.output / 'load_transfer.npz')
            if upward_platform_support() > 0.02:
                raise RuntimeError('pot not independently suspended')
    if args.platform_roundtrip:
        e.parameters['supported_outward_180mm_completed'] = True
        e.save_checkpoint(e.output / 'platform_turnaround.npz')
        for radius, centre in zip(radii, return_centres):
            if radius >= 0.18:
                continue
            e.stage = f'pot_return_{round(radius * 1000):03d}'
            T = reference_seat.copy()
            T[:3, :3] = Rotation.from_euler('z', heading).as_matrix()
            T[:3, 3] = centre
            T[2, 3] += clearance_lift(radius, 0)
            object_move(T, speed=args.return_speed, settle=args.settle)
            e.save_checkpoint(e.output / (e.stage + '.npz'))
        if radius>=.06 and args.extraction_timestep is not None and a.task_evidence['evidence']['pot_clear_of_outlet'].value is True and m.opt.timestep != (args.transport_timestep or .001):
            e.parameters.setdefault('precision_transitions',[]).append({'time_s':float(d.time),'stage':e.stage,'old_timestep_s':float(m.opt.timestep),'new_timestep_s':args.transport_timestep or .001,'reason':'outlet clearance verified after extraction'})
            m.opt.timestep=args.transport_timestep or .001
        e.stage = 'seat_contact'
        P = reference_seat.copy()
        P[2, 3] -= 0.0001
        object_move(P, speed=0.0005, settle=args.settle)
        require(['pot_on_hotplate', 'lid_outlet_aligned', 'pot_upright', *cup_keys])
        monitor_on = False
        load_ff_start = None
        command('left', np.array(grasp_geometry(m, d, 'left')['T_world_grasp']))
        e.stage = 'pot_release'
        e.close_gripper('left', 0.3)
        e.hold(2)
        require(['pot_stable_after_release', 'left_gripper_released', 'lid_outlet_aligned', 'pot_upright', *cup_keys])
        e.parameters['supported_return_completed'] = True
        e.parameters['full_coffee_sequence_completed'] = False
        e.save_checkpoint(e.output / 'supported_return_complete.npz')
        raise StopIteration
    require(['pot_clear_of_machine', 'pot_clear_of_outlet', 'pot_upright', *cup_keys])
    e.save_checkpoint(e.output / 'both_held.npz')
    e.parameters['withdrawal_completed'] = True
    e.required_relations |= {'pot_clear_of_machine', 'pot_clear_of_outlet', 'previous_served_cups_preserved'}

def pour_and_place_cup():
    global C, CG, G, P, allow_pour_tilt, corridor_end, corridor_high, corridor_start, cup_error, cup_feedback, cup_pose, cup_relative, error_position, parked, planned_pot_z, pour_pose, pre_pour, retreat, right_load_on, route_gap, route_offset, route_x, route_yaw, row, sd, source_cp, source_meta, target_volume, tilt, u, use_posture, velocity, yaw_error
    e.parameters.update({'full_geometry_sequence_completed': False, 'continuous_full_sequence': False, 'geometry_pour_completed': False, 'liquid_simulated': False, 'pour_offset_mm': args.pour_offset_mm, 'pour_gap_mm': args.pour_gap_mm})
    e.required_relations |= {'pot_clear_of_machine', 'pot_clear_of_outlet'}
    parked = pose()
    cup_relative = np.linalg.inv(np.array(grasp_geometry(m, d, 'right')['T_world_grasp'])) @ pose(active_cup)
    if args.phase in {'approach', 'transfer', 'tilted'}:
        parked = np.array(origin['parked_pose'])
        C = np.array(origin['cup_goal'])
        if 'cup_relative' in origin:
            cup_relative = np.array(origin['cup_relative'])
        else:
            source_cp = Path(args.checkpoint).parent / 'both_held.npz'
            source_meta = json.loads(source_cp.with_suffix('.json').read_text())
            if hashlib.sha256(source_cp.read_bytes()).hexdigest() != source_meta['npz_sha256'] or source_meta['model_sha256'] != e.model_hash:
                raise ValueError('cup grasp source checkpoint mismatch')
            sd = mujoco.MjData(m)
            mujoco.mj_setState(m, sd, np.load(source_cp)['state'], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(m, sd)
            cup_pose = np.eye(4)
            cup_pose[:3, 3] = sd.body(active_cup).xpos
            cup_pose[:3, :3] = sd.body(active_cup).xmat.reshape(3, 3)
            cup_relative = np.linalg.inv(np.array(grasp_geometry(m, sd, 'right')['T_world_grasp'])) @ cup_pose
        e.parameters['resumed_planned_descent_after_reference_guard_fix'] = args.phase == 'approach'
        e.parameters['resumed_after_cup_position'] = args.phase == 'transfer'
    else:
        C = pose(active_cup)
        transit_height = args.cup_pour_height_m
        if service.served_cups:
            transit_height = max(transit_height, e.config['table']['top_height_m'] + e.config['cup']['height_m'] + .05)
        e.parameters.setdefault('cup_transport_clearances', []).append({
            'cup':active_cup,'served_count':len(service.served_cups),
            'transit_bottom_m':transit_height,'pour_bottom_m':args.cup_pour_height_m,
            'offline_geometry_sha256':'a818cb5d7431546fd0605a48688fa82e99bd69dfeb45844646f398c7b2e8b24c'})
        e.stage = 'cup_raise'
        C[2, 3] = transit_height
        C[:3, :3] = Rotation.from_euler('z', np.arctan2(C[1, 0], C[0, 0])).as_matrix()
        cup_move(C)
        e.stage = 'cup_position'
        C[:3, 3] = [0.8, 0, transit_height] if args.pour_route == 'central' else [0.94, -0.3, transit_height]
        if args.pour_route == 'central':
            C[:3, :3] = Rotation.from_euler('z', 30, degrees=True).as_matrix() @ C[:3, :3]
        cup_move(C)
        if transit_height > args.cup_pour_height_m:
            e.stage = 'cup_lower_to_pour'
            C[2, 3] = args.cup_pour_height_m
            cup_move(C)
    e.parameters['parked_pose'] = parked.tolist()
    e.parameters['cup_goal'] = C.tolist()
    e.parameters['cup_relative'] = cup_relative.tolist()
    use_posture = True
    route_x = 0.87 if args.pour_route == 'central' else 0.92
    route_yaw = 60 if args.pour_route == 'central' else 75
    route_offset = args.pre_pour_offset_mm / 1000 if args.pour_route == 'central' else 0.09
    route_gap = args.pour_gap_mm / 1000
    e.stage = 'pot_retreat'
    if args.phase in {'approach', 'tilted'}:
        P = np.array(origin['retreat_pose'])
    elif args.phase == 'transfer':
        P = np.array(next((row['target'] for row in origin['reference_segments'] if row['stage'] == 'pot_retreat')))
        e.parameters['recovery_preserves_original_retreat_target'] = True
    else:
        P = pose()
    P[0, 3] = route_x
    if args.phase not in {'approach', 'tilted'}:
        object_move(P, settle=args.settle)
    retreat = P.copy()
    e.parameters['retreat_pose'] = retreat.tolist()
    e.stage = 'pot_transfer_height'
    P[2, 3] = reference_seat[2, 3] + support_height(e.config['machine'], 0.128) - nominal_seated_height(e.config['machine'], e.config['pot'])
    if args.phase not in {'approach', 'tilted'}:
        object_move(P, speed=0.001, settle=args.settle)
    corridor_start = P.copy()
    u = np.array([np.cos(np.deg2rad(route_yaw)), np.sin(np.deg2rad(route_yaw)), 0])
    P[:3, :3] = Rotation.from_euler('z', route_yaw, degrees=True).as_matrix()
    P[:3, 3] = C[:3, 3] + u * route_offset + [0, 0, 0.08 + route_gap] - P[:3, :3] @ m.site('pot_spout').pos
    pre_pour = P.copy()
    corridor_end = P.copy()
    corridor_end[0, 3] = route_x
    corridor_end[2, 3] = corridor_start[2, 3]
    e.stage = 'pot_lateral_transfer'
    if args.phase not in {'approach', 'tilted'}:
        object_move(corridor_end, speed=0.005, settle=args.settle)
    if args.phase not in {'approach', 'tilted'}:
        e.save_checkpoint(e.output / 'pot_lateral_transfer.npz')
    e.stage = 'pot_transfer_rise'
    corridor_high = corridor_end.copy()
    corridor_high[2, 3] = pre_pour[2, 3]
    if args.phase != 'tilted':
        object_move(corridor_high, speed=0.001, settle=args.settle)
        e.save_checkpoint(e.output / 'pot_before_cup.npz')
        e.stage = 'pot_above_cup'
        object_move(pre_pour, speed=0.003, settle=args.settle)
        e.save_checkpoint(e.output / 'pre_pour.npz')
    allow_pour_tilt = True

    def pour_pose(tilt):
        P = np.eye(4)
        P[:3, :3] = (Rotation.from_euler('z', route_yaw, degrees=True) * Rotation.from_euler('y', -tilt, degrees=True)).as_matrix()
        P[:3, 3] = C[:3, 3] + u * (route_offset - (route_offset - args.pour_offset_mm / 1000) * min(1, tilt / 30)) + [0, 0, 0.08 + args.pour_gap_mm / 1000] - P[:3, :3] @ m.site('pot_spout').pos
        return P
    for tilt in range(5, 46, 5):
        if args.phase == 'tilted' and tilt <= args.resume_tilt:
            continue
        e.stage = 'pour_tilt_' + str(tilt)
        object_move(pour_pose(tilt), settle=args.settle)
        e.save_checkpoint(e.output / (e.stage + '.npz'))
    e.stage = 'pour_geometry_hold'
    e.hold(2)
    require(['spout_over_cup', 'pour_clearance', *cup_keys])
    e.parameters['geometry_pour_completed'] = True
    e.save_checkpoint(e.output / 'pour_45.npz')
    if args.stop_after_pour_geometry:
        raise StopIteration
    e.stage = 'prescribed_payload_transfer'
    transfer_dt_before = float(m.opt.timestep)
    transfer_dt = args.transfer_timestep
    e.parameters["prescribed_transfer_timestep_s"] = transfer_dt
    e.parameters["pour_tilt_speed_scale"] = args.pour_tilt_speed_scale
    e.parameters["pour_angular_limit_deg_s"] = args.pour_angular_limit_deg_s
    e.parameters.setdefault('precision_transitions', []).append({'time_s': float(d.time), 'stage': e.stage, 'old_timestep_s': transfer_dt_before, 'new_timestep_s': transfer_dt, 'reason': 'prescribed static payload transfer; preserve pose/contacts while reducing redundant integration'})
    m.opt.timestep = transfer_dt
    right_load_on = True
    cup_feedback = np.zeros(3)
    target_volume = transfer.cavities[active_cup].capacity_m3 * 0.8
    while transfer.cup_volume_m3 < target_volume:
        require(['spout_over_cup', 'pour_clearance', *cup_keys])
        transfer.transfer(min(args.prescribed_rate_ml_s * 1e-6 * 0.02, target_volume - transfer.cup_volume_m3), geometry_eligible=True)
        P = pour_pose(45)
        planned_pot_z = float(P[2, 3])
        error_position = P[:3, 3] - pose()[:3, 3]
        feedback[:] += 0.01 * error_position
        feedback[:] *= min(1.0, 0.02 / max(np.linalg.norm(feedback), 1e-12))
        if sliding_stage() and upward_platform_support() > 0.02:
            yaw_error = np.arctan2(P[1, 0], P[0, 0]) - np.arctan2(pose()[1, 0], pose()[0, 0])
            yaw_error = float(np.arctan2(np.sin(yaw_error), np.cos(yaw_error)))
            angular_feedback[:] = supported_yaw_integral(angular_feedback, yaw_error, maximum_deg=args.return_yaw_max_deg if e.stage.startswith('pot_return_') or e.stage == 'seat_contact' else 1.0)
        else:
            angular_feedback[:] = bounded_rotation_integral(angular_feedback, Rotation.from_matrix(P[:3, :3] @ pose()[:3, :3].T).as_rotvec(), maximum_deg=air_correction_limit() if a.task_evidence['evidence']['pot_clear_of_outlet'].value is True else bumpless_rotation_limit(angular_feedback, requested_deg=1.0))
        G = P @ np.linalg.inv(held)
        G[:3, 3] += feedback
        G[:3, :3] = Rotation.from_rotvec(angular_feedback).as_matrix() @ G[:3, :3]
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, m.body('pot').id, velocity, 0)
        G[:3, 3] += 0.8 * error_position - 0.1 * velocity[3:]
        command('left', G)
        cup_error = C[:3, 3] - pose(active_cup)[:3, 3]
        cup_feedback += 0.01 * cup_error
        cup_feedback *= min(1.0, 0.01 / max(np.linalg.norm(cup_feedback), 1e-12))
        CG = C @ np.linalg.inv(cup_relative)
        CG[:3, 3] += cup_feedback + 0.8 * cup_error
        command('right', CG)
        e.tick()
        require(['spout_over_cup', 'pour_clearance', *cup_keys])
    m.opt.timestep = transfer_dt_before
    e.parameters.setdefault('precision_transitions', []).append({'time_s': float(d.time), 'stage': 'prescribed_payload_transfer_complete', 'old_timestep_s': transfer_dt, 'new_timestep_s': transfer_dt_before, 'reason': 'restore precise timestep before pour tilt restoration'})
    e.parameters['prescribed_payload_target_reached'] = True
    e.parameters['prescribed_rate_ml_s'] = args.prescribed_rate_ml_s
    e.save_checkpoint(e.output / 'prescribed_transfer_complete.npz')
    for tilt in range(40, -1, -5):
        e.stage = 'pour_restore_' + str(tilt)
        object_move(pour_pose(tilt), settle=args.settle)
        e.save_checkpoint(e.output / (e.stage + '.npz'))
    allow_pour_tilt = False
    if args.transport_timestep is not None:
        e.parameters.setdefault('precision_transitions', []).append({'time_s': float(d.time), 'stage': e.stage, 'old_timestep_s': float(m.opt.timestep), 'new_timestep_s': args.transport_timestep, 'reason': 'upright transport'})
        m.opt.timestep = args.transport_timestep
    require(['pot_upright'])
    e.stage = 'pot_restore_approach'
    object_move(pre_pour, settle=args.settle)
    e.stage = 'pot_back_to_front'
    object_move(corridor_high, settle=args.settle)
    e.stage = 'pot_back_to_height'
    object_move(corridor_end, speed=0.001, settle=args.settle)
    e.stage = 'pot_lateral_return'
    # Keep 4x free transport while bounding tracking error on the long lateral
    # return. Subdivide the same straight path and re-close feedback at each point.
    lateral_start = pose()
    lateral_rotation_delta = Rotation.from_matrix(
        corridor_start[:3, :3] @ lateral_start[:3, :3].T
    ).as_rotvec()
    for waypoint_index, fraction in enumerate(np.linspace(0.2, 1.0, 5), start=1):
        lateral_waypoint = np.eye(4)
        lateral_waypoint[:3, 3] = (
            (1 - fraction) * lateral_start[:3, 3] + fraction * corridor_start[:3, 3]
        )
        lateral_waypoint[:3, :3] = (
            Rotation.from_rotvec(fraction * lateral_rotation_delta).as_matrix()
            @ lateral_start[:3, :3]
        )
        e.parameters.setdefault('pot_lateral_return_waypoints', []).append(
            {'index': waypoint_index, 'fraction': float(fraction),
             'target': lateral_waypoint.tolist(), 'speed_factor': 4.0}
        )
        object_move(lateral_waypoint, speed=0.005, settle=args.settle)
    def segmented_return_waypoints(target, count, label):
        start = pose()
        rotation_delta = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T).as_rotvec()
        for index, fraction in enumerate(np.linspace(1.0 / count, 1.0, count), start=1):
            waypoint = np.eye(4)
            waypoint[:3, 3] = (1 - fraction) * start[:3, 3] + fraction * target[:3, 3]
            waypoint[:3, :3] = Rotation.from_rotvec(fraction * rotation_delta).as_matrix() @ start[:3, :3]
            e.parameters.setdefault(label + '_waypoints', []).append({
                'index': index, 'fraction': float(fraction), 'target': waypoint.tolist(),
                'speed_factor': 4.0,
            })
            object_move(waypoint, settle=args.settle)

    e.stage = 'pot_back_to_retreat'
    segmented_return_waypoints(retreat, 3, 'pot_back_to_retreat')
    e.stage = 'pot_back_to_staging'
    segmented_return_waypoints(parked, 3, 'pot_back_to_staging')
    use_posture = args.constrained_all
    e.save_checkpoint(e.output / 'pot_back_to_staging.npz')
    place_cup_with_footprint_margin()
    e.required_relations -= {'cup_in_right_gripper'}
    e.required_relations |= {'cup_upright'}
    command('right', np.array(grasp_geometry(m, d, 'right')['T_world_grasp']))
    e.stage = 'cup_release'
    e.close_gripper('right', 1.0)
    e.hold(1)
    require(['cup_supported_by_table', 'right_gripper_released', 'cup_upright', 'remaining_cups_retained'])
    right_load_on = False
    command('right', np.array(grasp_geometry(m, d, 'right')['T_world_grasp']))
    e.required_relations |= {'cup_supported_by_table', 'right_gripper_released', 'cup_upright'}
    e.stage = 'right_retract'
    G = np.array(grasp_geometry(m, d, 'right')['T_world_grasp'])
    e.parameters['right_release_retract_direction_world'] = (-G[:3, 2]).tolist()
    G[:3, 3] -= 0.05 * G[:3, 2]
    e.move_grasp('right', G)
    e.save_checkpoint(e.output / 'cup_placed_right_retracted.npz')
    a.task_evaluator.mark_served()
    require(['previous_served_cups_preserved'])

def return_pot():
    global G, P, centre, extra_suspension_m, heading, load_ff_start, monitor_on, radii, radius
    e.required_relations -= {'pot_clear_of_machine', 'pot_clear_of_outlet'}
    heading = np.arctan2(reference_seat[1, 0], reference_seat[0, 0])
    radii = [0.24, 0.22, 0.2, 0.18, 0.14, 0.12, 0.1, 0.06, 0.04, 0.03, 0.02, 0.01, 0]
    for radius, centre in list(zip(radii, clearance_entry_centres(reference_seat[:3, 3], heading if args.travel_heading_deg is None else np.deg2rad(args.travel_heading_deg), radii, detour_span_m=0.04, lateral_offset_m=args.return_entry_offset_mm / 1000)))[1:]:
        if args.phase == 'seated' or (args.phase == 'loaded_return' and radius >= args.return_start_mm / 1000):
            continue
        e.stage = 'pot_return_' + str(radius)
        P = reference_seat.copy()
        P[:3, :3] = Rotation.from_euler('z', heading).as_matrix()
        P[:3, 3] = centre
        P[2, 3] += clearance_lift(radius, args.return_lift_mm)
        object_move(P, speed=args.return_speed if radius <= 0.24 else 0.008, settle=args.settle)
        e.save_checkpoint(e.output / f'loaded_return_{round(radius * 1000):03d}.npz')
        if radius == 0.06 and args.pour_route == 'central' and args.allow_return_regrasp:
            args.regrasp_radius_mm = 60
            args.regrasp_roll_deg = args.return_grasp_roll_deg
            supported_regrasp()
        if radius == 0.2 and extra_suspension_m:
            e.stage = 'return_load_transfer'
            P = last_planned_pose.copy()
            P[2, 3] -= extra_suspension_m
            extra_suspension_m = 0.0
            object_move(P, speed=0.0005, settle=args.settle)
    if args.phase != 'seated':
        e.stage = 'seat_contact'
        P = pose()
        P[2, 3] = reference_seat[2, 3] - 0.0001
        object_move(P, speed=0.0005, settle=args.settle)
    e.stage = 'seat_contact'
    require(['pot_on_hotplate', 'lid_outlet_aligned', 'pot_upright'])
    monitor_on = False
    load_ff_start = None
    command('left', np.array(grasp_geometry(m, d, 'left')['T_world_grasp']))
    e.save_checkpoint(e.output / 'pot_seated_before_release.npz')
    e.stage = 'pot_release'
    e.parameters['release_opening_fraction'] = args.release_opening
    e.close_gripper('left', args.release_opening)
    e.hold(2)
    predicates={k:v.value for k,v in a.task_evidence['evidence'].items()}
    if release_retry_allowed(predicates,e.contacts(),model_sha256=e.model_hash,opening=args.release_opening):
        # One retry per return. Preserve original release thresholds and support.
        event=dict(start_s=float(d.time),original_opening=args.release_opening,
                   retry_opening=RETRY_OPENING,contacts_before=e.contacts())
        e.parameters.setdefault('supported_release_retries',[]).append(event)
        e.save_checkpoint(e.output / 'supported_release_retry_before.npz')
        e.hold(.4)
        predicates={k:v.value for k,v in a.task_evidence['evidence'].items()}
        if not release_retry_allowed(predicates,e.contacts(),model_sha256=e.model_hash,opening=args.release_opening):
            raise RuntimeError('supported release retry prerequisites changed')
        command('left',np.array(grasp_geometry(m,d,'left')['T_world_grasp']))
        e.close_gripper('left',RETRY_OPENING)
        e.hold(2)
        require(['pot_stable_after_release','pot_on_hotplate','lid_outlet_aligned','pot_upright','left_gripper_released','previous_served_cups_preserved','right_gripper_empty','rack_inventory_supported'])
        event.update(verified_time_s=float(d.time),contacts_after=e.contacts(),verified=True)
        e.save_checkpoint(e.output / 'supported_release_retry_verified.npz')
    require(['pot_stable_after_release', 'pot_on_hotplate', 'lid_outlet_aligned'])
    e.stage = 'left_gripper_retract'
    G = np.array(grasp_geometry(m, d, 'left')['T_world_grasp'])
    e.parameters['left_release_retract_direction_world'] = (-G[:3, 2]).tolist()
    G[:3, 3] -= 0.04 * G[:3, 2]
    e.move_grasp('left', G, speed=0.005)
    e.hold(2)
    require(['pot_on_hotplate', 'pot_stable_after_release', 'lid_outlet_aligned', 'pot_upright', 'left_gripper_released', 'right_gripper_empty', 'previous_served_cups_preserved', 'remaining_cups_retained'])


def observed_service_evidence():
    a.task_evidence=a.task_evaluator.update()
    return ServiceEvidence({k:v.value for k,v in a.task_evidence["evidence"].items()},
        remaining_pot_ml=transfer.pot_volume_m3*1e6,
        cup_id=active_cup,cup_volume_ml=transfer.cup_volume_m3*1e6)


def collect_served_cup(cup):
    e.stage = "staff_collection_wait"
    e.hold(.22)
    require(["previous_served_cups_preserved", "right_gripper_empty"])
    e.stage = "external_staff_collection"
    e.record(force=True)
    before = e.save_checkpoint(e.output / f"staff_collection_before_{cup}.npz")
    receipt = collector.collect(cup, staff_confirmed=True)
    e.required_relations -= {"cup_supported_by_table", "cup_upright", "right_gripper_released", "cup_in_right_gripper", "single_cup_separated"}
    a.task_evidence = a.task_evaluator.update()
    a._last_evaluation_s = float(d.time)
    require(["previous_served_cups_preserved", "right_gripper_empty"])
    e.record(force=True)
    after = e.save_checkpoint(e.output / f"staff_collection_after_{cup}.npz")
    e.parameters.setdefault("staff_collection_checkpoints", []).append({
        "cup_id": cup, "time_s": receipt["time_s"],
        "before": before.name, "after": after.name,
        "before_sha256": hashlib.sha256(before.read_bytes()).hexdigest(),
        "after_sha256": hashlib.sha256(after.read_bytes()).hexdigest(),
    })
    print(json.dumps(receipt), flush=True)

checkpoint_serial=0
service_checkpoint_delegate=e.save_checkpoint
def service_save(path):
    global checkpoint_serial
    checkpoint_serial+=1
    actual = Path(path).with_name(f"{checkpoint_serial:04d}_"+Path(path).name)
    service_checkpoint_delegate(actual)
    return actual
e.save_checkpoint=service_save

error=None
try:
    e.parameters = {'arguments': vars(args).copy(), 'physics_timestep_override_s': args.physics_timestep, 'operator_upper_grasp_constraints': True, 'outlet_unilateral_window_s': outlet_allowance.maximum_s, 'sustained_load_override': {'original_n': e.config['simulation_diagnostics']['pad_load']['sustained_n'], 'trial_n': args.sustained_load_n, 'clear_n': load_limits.clear_n, 'peak_n': load_limits.peak_n, 'simulation_only': True}, 'bumpless_platform_transition': True, 'airborne_orientation_control': {'platform_free_hold_s': 0.5, 'ramp_deg_s': 0.5, 'initial_bound_deg': 2, 'final_bound_deg': args.air_correction_deg, 'grasp_limit_unchanged_deg': args.diagnostic_grasp_angle_deg}, 'force_diagnostic_basis': 'per_finger_normal_vector_resultant_plus_single_contact_peak', 'pot_drop_guard_reference': 'current_command_height_else_seat', 'return_grasp_roll_deg': args.return_grasp_roll_deg, 'supported_level_correction_deg': args.supported_level_correction_deg, 'extraction_yaw_max_deg': args.extraction_yaw_max_deg, 'initial_grasp_roll_deg': args.initial_grasp_roll_deg, 'grasp_height_offset_mm': args.grasp_height_offset_mm, 'cup_pour_height_m': args.cup_pour_height_m, 'pour_route': args.pour_route, 'constrained_all': args.constrained_all, 'diagnostic_grasp_angle_deg': args.diagnostic_grasp_angle_deg, 'original_grasp_limit_qualification': args.diagnostic_grasp_angle_deg == 5, 'runtime_grip': runtime_grip, 'counterfactual_grip_change_at_resume': args.phase != 'start' and args.grip_scale != 1, 'suspend_height_mm': args.suspend_height_mm, 'air_correction_deg': args.air_correction_deg, 'cold_extraction_air_correction_deg': min(2.0, args.air_correction_deg), 'return_entry_offset_mm': args.return_entry_offset_mm, 'return_clearance': args.return_clearance, 'return_ik_position_reference': 'grasp_center' if args.return_clearance else 'mount', 'return_tool_tilt_allowance_deg': 3 if args.return_clearance else None, 'return_clearance_target_m': 0.002 if args.return_clearance else 0.004, 'return_ik_minimum_clearance_m': 0.001 if args.return_clearance else None, 'return_yaw_max_deg': args.return_yaw_max_deg, 'return_start_mm': args.return_start_mm, 'supported_slide_max_tilt_deg': args.supported_slide_tilt_deg, 'multiccd': True, 'aligned_to_handle_rail_slope': True, 'stock_pad_geometry_preserved': True, 'initial_load_cup_equivalents': 2.0, 'platform_geometry': e.config['platform_geometry'], 'platform_roundtrip_test': args.platform_roundtrip, 'initial_settle_s': 8.0, 'pad_load_diagnostic': asdict(load_limits), 'rack_support': e.config['simulation_diagnostics']['rack_support'], 'loaded_pd': args.loaded_pd, 'prelift_mm': args.prelift_mm, 'prelift': args.prelift, 'payload_feedforward': args.payload_feedforward, 'payload': e.config.get('payload'), 'contact_guard': 'all_contacts_v2', 'simulation_only': True, 'pot_initial_yaw_deg': 135, 'reset_robot_at_pregrasp': args.phase == 'start', 'debug_resume': args.phase != 'start', 'continuous_object_reference': True, 'clearance_profile': {'base_mm': args.extraction_lift_mm, 'r40_to60_added_mm': 0.0, 'supported_sliding': True, 'supported_yaw_correction': {'gain_per_20ms': 0.04, 'deadband_deg': 0.25, 'maximum_deg': 1.0, 'return_maximum_deg': args.return_yaw_max_deg}, 'extended_exit_radius_m': 0.24, 'r100_to120_added_mm': 0.0, 'r140_to180_added_mm': 0.0}, 'angular_integral_policy': {'gain_per_20ms': 0.04, 'deadband_deg': 0.5, 'maximum_contact_deg': 1.0, 'maximum_outlet_clear_deg': args.air_correction_deg}, 'cold_start_objects_on_fixture': args.phase == 'start', 'resumed_cup_extraction': args.phase == 'cup_extract', 'physical_motion_executed': False, 'pot_roundtrip_completed': False, 'grasp_anchor_reset_after_close_only': True, 'support_profile_following': True, 'extraction_lift_mm': args.extraction_lift_mm, 'resume_source_lift_mm': args.extraction_lift_mm, 'waypoint_settle_s': args.settle, 'extraction_speed_m_s': args.extract_speed, 'adaptive_settle': args.adaptive_settle, 'return_speed_m_s': args.return_speed, 'return_lift_mm': 0.0, 'robot_contact_check_period_s': 0.02, 'torso_and_grasp_guard_period_s': float(m.opt.timestep)}
    e.parameters["low_payload_restore_control"]={"only_stage_prefix":"pour_restore_","only_below_one_full_cup":True,"maximum_feedback_deg":args.low_payload_restore_correction_deg,"ramp_deg_s":.5,"grasp_anchor_reset":False,"guard_limits_changed":False,"local40_checkpoint_validation_sha256":"evidence hash set below"}
    e.parameters["low_payload_restore_control"]["local40_checkpoint_validation_sha256"]=hashlib.sha256((Path(__file__).parent/"low_payload_restore40_checkpoint_validation.json").read_bytes()).hexdigest()
    e.parameters["cup_endpoint_budget"]={"maximum_s":args.cup_endpoint_max_seconds,"position_tolerance_m":.002,"stable_ticks":20,"guard_limits_changed":False,"local_recovery_validation_sha256":hashlib.sha256((Path(__file__).parent/"second_cup_inset_recovery_validation.json").read_bytes()).hexdigest()}
    e.parameters["simulation_step_implementation"]="fresh step1 at control boundary; step2/full forward each physics step"
    e.parameters["simulation_step_validation_sha256"]=hashlib.sha256((Path(__file__).parent/"cached_step_validation.json").read_bytes()).hexdigest()
    e.parameters["service_completed"]=False
    e.parameters["postpour_return_speed_scale"] = 2.0
    e.parameters["postpour_return_local_validation_sha256"] = '7723bde230e0c9d5a774132e1f1fc1aa0687a34c021cf0ff83f81133e2aca94f'
    e.parameters["held_contact_precision_evidence_sha256"]='c757755ef740ddb69bfe0c2101fc23ad7a10c0f6e5ff3ee6140abfd452b556d2'
    e.parameters["lifecycle_included"]=False
    e.parameters.update(delivery_mode="staff_collect_fixed_slot", fixed_delivery_slot_m=fixed_delivery_slot(e.config).tolist(), external_actor_object_pose_writes=True, staff_pickup_assumed=True)
    service=CoffeeService(transfer.cavities[active_cup].capacity_m3*1e6,transfer.pot_volume_m3*1e6)
    if args.phase == "loaded_return":
        service.state = "holding_pot"
        service.served_cups = {cup: volume * 1e6 for cup, volume in transfer.state()["vessel_volumes_m3"].items() if cup in a.task_evaluator.served}
        e.parameters["continued_from_verified_checkpoint"] = str(args.checkpoint)
        e.parameters["resumed_service_state"] = "holding_pot_after_one_completed_serving"
        e.parameters["lifecycle_included"] = True
        e.required_relations.update({"previous_served_cups_preserved"})
        a.task_evidence = a.task_evaluator.update()
    else:
        service.request_cups(args.requested_cups)
        lifecycle.prepare()
        verify_fixture()
        initial_evidence=observed_service_evidence()
        initial_evidence.predicates.update(lifecycle.pose_evidence())
        service.complete(CoffeeAction.PREPARE,initial_evidence)
        e.required_relations.update({"previous_served_cups_preserved","rack_inventory_supported"})
    refill_count=0
    while True:
        if not service.pending_cups and service.state != "holding_pot":
            service.request_exit()
        action=service.next_action()
        print(json.dumps({"business_action":action.value,"time_s":float(d.time),"pending_cups":service.pending_cups,"remaining_ml":service.remaining_pot_ml}),flush=True)
        if action==CoffeeAction.TAKE_POT:
            take_pot()
            service.complete(action,observed_service_evidence())
            if args.stop_after_service_take_pot:
                e.parameters["stopped_after_service_take_pot"]=True
                break
        elif action==CoffeeAction.SERVE_CUP:
            take_cup();pour_and_place_cup()
            service.complete(action,observed_service_evidence())
            collect_served_cup(active_cup)
        elif action==CoffeeAction.RETURN_POT:
            return_pot()
            service.complete(action,observed_service_evidence())
        elif action==CoffeeAction.WAIT_REFILL:
            e.stage="wait_for_external_refill";e.hold(5)
            if not service.pending_cups or refill_count>=args.demo_refills:
                e.parameters["waiting_for_external_refill"]=True
                break
            evidence=observed_service_evidence()
            evidence.require("pot_stable_after_release","both_grippers_empty","previous_served_cups_preserved")
            # Explicit benchmark environmental event, not an observed real refill.
            amount=transfer.initial_volume_m3-transfer.pot_volume_m3
            transfer.refill(amount,staff_confirmed=True,pot_supported_released=True)
            e.hold(1)
            evidence=observed_service_evidence();evidence.predicates["staff_refill_confirmed"]=True
            service.acknowledge_refill(transfer.pot_volume_m3*1e6,evidence)
            refill_count+=1
        elif action==CoffeeAction.EXIT:
            lifecycle.exit(transport_timestep=args.transport_timestep or .001)
            evidence=observed_service_evidence()
            evidence.predicates.update(lifecycle.pose_evidence(standing=True))
            service.complete(action,evidence)
            break
        elif action==CoffeeAction.IDLE:
            break
        else:
            raise RuntimeError("unimplemented service action: "+action.value)
    e.parameters["service_completed"]=not service.pending_cups and service.state=="standing" and e.parameters.get("exit_completed",False)
    e.parameters["business_final_state"]=service.state
except BaseException as exc:
    error=f"{type(exc).__name__}: {exc}"
    (e.output/"exception.txt").write_text(__import__("traceback").format_exc())
finally:
    e.parameters.update(lifecycle.diagnostics())
    if "service" in globals():e.parameters["business_events"]=service.events
    e.parameters["runtime_payload"]=transfer.state()
    e.parameters["staff_collection"]=collector.state()
    if release_last_load is not None and release_load_rows[-1]['time_s'] != release_last_load['time_s']:
        release_load_rows.append(release_last_load)
    for name,rows in [("release_load",release_load_rows),("command_trace",command_rows),("velocity_trace",velocity_rows),("control_trace",control_rows),("support_trace",support_rows),("force_trace",force_rows),("grasp_guard",guard_rows)]:
        (e.output/(name+".json")).write_text(json.dumps(rows,indent=2)+"\n")
    e.finish(error)
    print("finished",error,flush=True)
