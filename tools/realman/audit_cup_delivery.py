"""Audit fixed-point delivery and explicit staff pickup from recorded states."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_served_cups import inspect_sample


def inspect_transition(model, before, after, bm, am, config, *, fixed_slot=True):
    """Check the declared external change, independently of controller flags."""
    old = bm["staff_collection"]["events"]
    new = am["staff_collection"]["events"]
    if len(new) != len(old) + 1 or new[:-1] != old:
        raise ValueError("collection ledger must append exactly one event")
    event = new[-1]
    cup = event["cup_id"]
    body = model.body(cup).id
    joint = int(model.body_jntadr[body])
    if joint < 0 or model.jnt_type[joint] != 0:
        raise ValueError("collected body must have its own free joint")
    qadr, vadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    qi = np.ones(model.nq, bool); qi[qadr:qadr + 7] = False
    vi = np.ones(model.nv, bool); vi[vadr:vadr + 6] = False
    half = np.asarray(config["table"]["size_m"][:2]) / 2
    center = np.asarray(config["table"]["center_xy_m"])
    slot = center + [-half[0] + config["cup"]["bottom_radius_m"] + .035,
                     -(2 * config["cup"]["rim_radius_m"] + .02)]
    volume = bm["runtime_payload"]["vessel_volumes_m3"][cup] * 1e6
    checks = {
        "explicit_staff_event": event["event"] == "external_staff_cup_collection"
        and event["actor"] == "simulated_staff" and event["confirmation_source"] == "operator_assumption"
        and event["event_id"] == len(old) and cup not in {x["cup_id"] for x in old},
        "same_clock": max(abs(float(d.time) - event["time_s"]) for d in [before, after]) < 1e-8
        and abs(bm["time_s"] - am["time_s"]) < 1e-8,
        "declared_before_pose": np.allclose(before.qpos[qadr:qadr+3], event["before_position_m"], atol=1e-9, rtol=0)
        and np.allclose(before.qpos[qadr+3:qadr+7], event["before_quaternion_wxyz"], atol=1e-9, rtol=0),
        "declared_after_pose": np.allclose(after.qpos[qadr:qadr+3], event["after_position_m"], atol=1e-9, rtol=0)
        and np.allclose(after.qpos[qadr+3:qadr+7], event["after_quaternion_wxyz"], atol=1e-9, rtol=0),
        "other_objects_and_robot_unchanged": np.array_equal(before.qpos[qi], after.qpos[qi])
        and np.array_equal(before.qvel[vi], after.qvel[vi]),
        # mj_forward refreshes this solver cache, with roundoff at ~1e-13.
        # Physical positions and velocities above must remain bitwise equal.
        "solver_cache_unchanged_within_roundoff": np.allclose(
            before.qacc_warmstart[vi], after.qacc_warmstart[vi], atol=1e-10, rtol=0),
        "controls_unchanged": all(np.array_equal(getattr(before, k), getattr(after, k))
            for k in ["ctrl", "act", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata", "eq_active"]),
        "payload_and_grasp_preserved": bm["runtime_payload"] == am["runtime_payload"]
        and bm.get("grasp_context") == am.get("grasp_context")
        and bm["runtime_grip"] == am["runtime_grip"] and bm["runtime_options"] == am["runtime_options"],
        "eighty_percent_serving": abs(volume - .8 * config["payload"]["cup_capacity_ml"]) < 1e-6
        and abs(volume - event["volume_ml"]) < 1e-6,
        "outside_workspace": bool(np.any(np.abs(after.qpos[qadr:qadr+2] - center) > half + .25)),
        "fixed_delivery_point": not fixed_slot or np.linalg.norm(before.qpos[qadr:qadr+2] - slot) <= .002,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return {"passed": all(checks.values()), "checks": checks, "event": event,
            "fixed_slot_required": fixed_slot, "slot_xy_m": slot.tolist(),
            "maximum_other_solver_cache_change": float(np.max(np.abs(
                before.qacc_warmstart[vi] - after.qacc_warmstart[vi]), initial=0))}


def audit_transitions(trial, expected_cups, *, fixed_slot=True):
    import mujoco as mj
    trial = Path(trial).resolve()
    model_path = trial.parent / "review/model/scene.xml"
    config = json.loads((trial.parent / "review/scene_config.json").read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    model = mj.MjModel.from_xml_path(str(model_path))
    pairs = []
    for path in sorted(trial.glob("*_staff_collection_before_cup_*.json")):
        bm = json.loads(path.read_text())
        cup = path.stem.split("staff_collection_before_", 1)[1]
        matches = list(trial.glob("*_staff_collection_after_" + cup + ".json"))
        if len(matches) != 1:
            raise ValueError("missing or duplicate after-pickup checkpoint")
        after_path = matches[0]
        am = json.loads(after_path.read_text())
        data = []
        for p, meta in [(path, bm), (after_path, am)]:
            if sha(p.with_suffix(".npz")) != meta["npz_sha256"] or sha(model_path) != meta["model_sha256"]:
                raise ValueError("pickup checkpoint identity mismatch")
            if hashlib.sha256(json.dumps(meta["staff_collection"], sort_keys=True).encode()).hexdigest() != meta["staff_collection_sha256"]:
                raise ValueError("pickup ledger checksum mismatch")
            d = mj.MjData(model)
            mj.mj_setState(model, d, np.load(p.with_suffix(".npz"))["state"], mj.mjtState.mjSTATE_INTEGRATION)
            mj.mj_kinematics(model, d)
            data.append(d)
        pair = inspect_transition(model, *data, bm, am, config, fixed_slot=fixed_slot)
        if pair["event"]["cup_id"] != cup:
            raise ValueError("pickup identity does not match checkpoint name")
        pair.update(before_checkpoint=path.with_suffix(".npz").name, after_checkpoint=after_path.with_suffix(".npz").name,
                    before_sha256=bm["npz_sha256"], after_sha256=am["npz_sha256"])
        pairs.append(pair)
    checks = {"expected_count": len(pairs) == expected_cups > 0,
              "unique_cups": len({x["event"]["cup_id"] for x in pairs}) == len(pairs),
              "all_transitions_valid": bool(pairs) and all(x["passed"] for x in pairs)}
    return {"passed": all(checks.values()), "scope": "external pickup checkpoint transitions only",
            "checks": checks, "transitions": pairs, "model_sha256": sha(model_path),
            "physical_motion_allowed": False, "full_service_qualified": False}


def audit(trial, expected_cups):
    import mujoco as mj
    trial = Path(trial).resolve()
    transitions = audit_transitions(trial, expected_cups)
    trace = json.loads((trial / "trace.json").read_text())
    result = json.loads((trial / "result.json").read_text())
    config = json.loads((trial.parent / "review/scene_config.json").read_text())
    m = mj.MjModel.from_xml_path(str(trial.parent / "review/model/scene.xml"))
    d = mj.MjData(m)
    geom_bodies = {m.geom(i).name: m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)}
    stats = {}
    for pair in transitions["transitions"]:
        event = pair["event"]; cup = event["cup_id"]
        placements = []
        for p in trial.glob("*_cup_placed_right_retracted.json"):
            meta = json.loads(p.read_text())
            if meta["runtime_payload"]["active_cup"] == cup:
                if hashlib.sha256(p.with_suffix(".npz").read_bytes()).hexdigest() != meta["npz_sha256"]:
                    raise ValueError("placement identity mismatch")
                placements.append(meta)
        if len(placements) != 1:
            raise ValueError("one physical placement required before each pickup")
        start, end = placements[0]["time_s"], event["time_s"]
        before_rows, after_rows, failures = [], [], []
        for row in trace:
            if row["time_s"] < start - 1e-8:
                continue
            d.qpos[:] = row["qpos"]; mj.mj_kinematics(m, d)
            collected = {x["cup_id"] for x in row.get("staff_collection", {}).get("events", [])}
            if cup not in collected:
                sample = inspect_sample(d.body(cup).xpos, d.body(cup).xmat[8], event["before_position_m"], row["contacts"], geom_bodies, cup, config)
                before_rows.append(row["time_s"])
                if not sample["passed"] or row["time_s"] > end + 1e-8:
                    failures.append({"time_s": row["time_s"], "phase": "awaiting_pickup", **sample})
            else:
                half = np.asarray(config["table"]["size_m"][:2]) / 2
                outside = np.any(np.abs(d.body(cup).xpos[:2] - config["table"]["center_xy_m"]) > half + .25)
                unexpected = []
                for c in row["contacts"]:
                    if c["normal_force_n"] <= .02:
                        continue
                    b1, b2 = geom_bodies[c["geom1"]], geom_bodies[c["geom2"]]
                    if cup in {b1, b2}:
                        other = c["geom2"] if b1 == cup else c["geom1"]
                        if other != "floor" and geom_bodies[other] != cup:
                            unexpected.append(c)
                after_rows.append(row["time_s"])
                volume = row["runtime_payload"]["vessel_volumes_m3"][cup] * 1e6
                if not outside or unexpected or abs(volume - event["volume_ml"]) > 1e-6:
                    failures.append({"time_s": row["time_s"], "phase": "collected", "outside": bool(outside), "contacts": unexpected})
        coverage = len(before_rows) >= 2 and len(after_rows) >= 2
        if coverage:
            coverage = before_rows[0] - start <= .101 and abs(before_rows[-1] - end) < 1e-8
            coverage &= abs(after_rows[0] - end) < 1e-8 and abs(after_rows[-1] - result["simulation_seconds"]) < 1e-6
            coverage &= max(np.diff(before_rows)) <= .101 and max(np.diff(after_rows)) <= .101
        stats[cup] = {"passed": bool(coverage) and not failures, "coverage": bool(coverage), "before_samples": len(before_rows), "after_samples": len(after_rows), "violations": failures}
    served = [x["cup_id"] for x in result["parameters"]["business_events"] if x["action"] == "serve_cup"]
    checks = {"transitions": transitions["passed"], "identities": len(served) == len(set(served)) == expected_cups and set(served) == set(stats),
              "table_then_collection_preserved": bool(stats) and all(x["passed"] for x in stats.values()),
              "final_ledger": result["parameters"]["staff_collection"]["events"] == [x["event"] for x in transitions["transitions"]]}
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    return {"passed": all(checks.values()), "checks": checks, "scope": "recorded fixed-point delivery and explicit staff pickup",
            "cups": stats, "transitions": transitions, "trace_sha256": sha(trial / "trace.json"), "result_sha256": sha(trial / "result.json"),
            "physical_motion_allowed": False, "full_service_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--expected-cups", type=int, default=3)
    parser.add_argument("--transitions-only", action="store_true")
    parser.add_argument("--allow-diagnostic-prior-slots", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.allow_diagnostic_prior_slots and not args.transitions_only:
        parser.error("prior-slot allowance is for local transition diagnostics only")
    result = audit_transitions(args.trial, args.expected_cups, fixed_slot=not args.allow_diagnostic_prior_slots) if args.transitions_only else audit(args.trial, args.expected_cups)
    out = args.output or args.trial / ("collection_transitions_audit.json" if args.transitions_only else "cup_delivery_audit.json")
    if out.exists():
        raise FileExistsError("preserve prior audit; choose a new output")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
