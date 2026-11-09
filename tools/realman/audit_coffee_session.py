"""Audit a completed continuous coffee lifecycle/service simulation.

This consumes a single run's recorded trajectory and object evidence. It does not
convert oracle geometry or prescribed payload into fluid/hardware qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def audit_business_sequence(events, payload, capacity_ml, expected_cups):
    """Independently reconcile ordered service actions with the payload ledger."""
    state, pending = "standing", expected_cups
    remaining = payload["initial_volume_m3"] * 1e6
    served, refills, errors = set(), 0, []
    ledger = payload["external_events"]
    if (
        not math.isfinite(remaining)
        or remaining < 0
        or not math.isfinite(capacity_ml)
        or capacity_ml <= 0
        or isinstance(expected_cups, bool)
        or not isinstance(expected_cups, int)
        or expected_cups <= 0
    ):
        return {
            "passed": False,
            "errors": ["invalid service inventory or request"],
            "served_cups": [],
            "refills": 0,
        }
    for index, event in enumerate(events):
        action = event.get("action")
        previous = state
        try:
            if action == "prepare" and index == 0 and state == "standing":
                state = "ready"
            elif action == "take_pot" and state == "ready" and pending > 0:
                if remaining < capacity_ml - 1e-9:
                    raise ValueError("pot taken with less than one full cup")
                state = "holding_pot"
            elif action == "serve_cup" and state == "holding_pot" and pending > 0:
                if remaining < capacity_ml - 1e-9:
                    raise ValueError("serving began below the one-full-cup reserve")
                cup = event["cup_id"]
                if not cup or cup in served:
                    raise ValueError("cup identity reused")
                volume = payload["vessel_volumes_m3"][cup] * 1e6
                if not math.isfinite(volume) or abs(volume - 0.8 * capacity_ml) > 1e-6:
                    raise ValueError("served cup is not at 80% capacity")
                remaining -= volume
                pending -= 1
                served.add(cup)
            elif action == "return_pot" and state == "holding_pot":
                state = "ready" if remaining >= capacity_ml - 1e-9 else "waiting_refill"
            elif action == "external_refill" and state in {"ready", "waiting_refill"}:
                observed = event["remaining_pot_ml"]
                added = ledger[refills]["added_m3"] * 1e6
                if (
                    not math.isfinite(observed)
                    or not math.isfinite(added)
                    or added <= 0
                    or abs(observed - remaining - added) > 1e-6
                ):
                    raise ValueError("refill does not match the external payload ledger")
                remaining, refills = observed, refills + 1
                state = "ready" if remaining >= capacity_ml - 1e-9 else "waiting_refill"
            elif action == "exit" and state in {"ready", "waiting_refill"} and pending == 0:
                state = "standing"
            else:
                raise ValueError(f"action {action!r} is invalid from {state!r}")
            if action != "external_refill":
                if event["from"] != previous or event["to"] != state:
                    raise ValueError("reported state transition differs from action order")
                if event["pending_cups"] != pending:
                    raise ValueError("pending cup count differs from completed servings")
                observed = event["remaining_pot_ml"]
                if not math.isfinite(observed) or abs(observed - remaining) > 1e-6:
                    raise ValueError("reported pot inventory differs from transfers")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"event {index}: {exc}")
            break
    if (
        state != "standing"
        or pending != 0
        or len(served) != expected_cups
        or refills != len(ledger)
        or not events
        or events[-1].get("action") != "exit"
        or not math.isfinite(payload["pot_volume_m3"])
        or abs(payload["pot_volume_m3"] * 1e6 - remaining) > 1e-6
    ):
        errors.append("complete service, refill ledger and exit were not reconciled")
    volumes = payload["vessel_volumes_m3"]
    if (
        any(not math.isfinite(value) or value < 0 for value in volumes.values())
        or abs(volumes.get("pot", -1) - payload["pot_volume_m3"]) > 1e-12
        or any(
            abs(value) > 1e-12
            for name, value in volumes.items()
            if name != "pot" and name not in served
        )
        or abs(
            sum(volumes.values())
            - payload["initial_volume_m3"]
            - sum(event["added_m3"] for event in ledger)
        )
        > 1e-12
    ):
        errors.append("final vessel inventory differs from completed service and refills")
    return {
        "passed": not errors,
        "errors": errors,
        "served_cups": sorted(served),
        "refills": refills,
    }


def audit(trial: Path, *, service: bool, expected_cups: int = 1):
    import mujoco as mj

    trial = trial.resolve()
    result = json.loads((trial / "result.json").read_text())
    params = result["parameters"]
    args = params["arguments"]
    trace = json.loads((trial / "trace.json").read_text())
    guards = json.loads((trial / "grasp_guard.json").read_text())
    model_path = trial.parent / "review/model/scene.xml"
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    checks = {}

    def check(name, value):
        checks[name] = bool(value)

    check("run_reported_success", result["experiment_success"] and result["error"] is None)
    check("simulation_only", result["simulation_only"] and not result["physical_motion_allowed"])
    check("fresh_start_arguments", args["phase"] == "start" and args["checkpoint"] is None)
    init = Path(result["checkpoint"])
    meta = json.loads(init.with_suffix(".json").read_text())
    check("time_zero_initial_state", meta["time_s"] == 0 and sha(init) == meta["npz_sha256"])
    check("model_identity", sha(model_path) == result["model_sha256"] == meta["model_sha256"])
    check(
        "no_resume_or_regrasp",
        not params.get("restored_checkpoint_grasp_context")
        and not params.get("original_anchor_source")
        and not args["allow_return_regrasp"],
    )
    times = np.array([row["time_s"] for row in trace])
    check(
        "continuous_record",
        len(times) > 1
        and times[0] == 0
        and np.all(np.diff(times) >= 0)
        and np.max(np.diff(times)) <= 0.101
        and abs(times[-1] - result["simulation_seconds"]) < 1e-6,
    )
    stages = []
    for row in trace:
        if not stages or stages[-1] != row["stage"]:
            stages.append(row["stage"])
    check("preparation_completed", params.get("preparation_completed") is True)
    check(
        "exit_completed",
        params.get("exit_completed") is True and stages[-1] == "standing_final_hold",
    )
    if service:
        expected = [
            "prepare_raise_laterally",
            "prepare_lower_body",
            "pot_close",
            "pot_extract_240",
            "cup_close",
            "cup_extract_down",
            "pour_tilt_45",
            "prescribed_payload_transfer",
            "cup_release",
            "pot_release",
            "standing_final_hold",
        ]
    else:
        expected = [
            "prepare_raise_laterally",
            "prepare_lower_body",
            "cup_close",
            "cup_extract_down",
            "pot_close",
            "pot_extract_240",
            "pour_tilt_45",
            "prescribed_payload_transfer",
            "cup_release",
            "pot_release",
            "standing_final_hold",
        ]
    indices = [stages.index(name) if name in stages else -1 for name in expected]
    check("stage_order", min(indices) >= 0 and indices == sorted(indices))
    relations = [
        "pot_on_hotplate",
        "pot_stable_after_release",
        "lid_outlet_aligned",
        "pot_upright",
        "left_gripper_released",
        "cup_supported_by_table",
        "cup_upright",
        "right_gripper_released",
        "remaining_cups_retained",
    ]
    if service:
        relations += ["previous_served_cups_preserved", "both_grippers_empty"]
    check("final_object_relations", all(result["predicates"].get(key) is True for key in relations))
    check("guard_trace", bool(guards) and not any(row["failures"] for row in guards))
    check(
        "guard_limits_preserved",
        params["grasp_limits"]["relative_rotation_deg"] == 8
        and params["grasp_limits"]["handle_translation_m"] == 0.005
        and params["pad_load_diagnostic"]["sustained_n"] == 40
        and params["pad_load_diagnostic"]["peak_n"] == 50,
    )
    check(
        "loads_and_pour_rate",
        params["maximum_finger_normal_resultant_n_all_steps"] <= 50
        and params["maximum_pour_angular_speed_deg_s"] <= 5,
    )
    check(
        "stock_grip_drive",
        params["runtime_grip"]["kp"] == 36
        and params["runtime_grip"]["forcerange"] == [-2.5, 2.5]
        and args["gripper_damping"] == 0.2
        and args["left_contact_margin"] == 0,
    )
    m = mj.MjModel.from_xml_path(str(model_path))
    d = mj.MjData(m)
    tilts = []
    for row in trace:
        d.qpos[:] = row["qpos"]
        mj.mj_kinematics(m, d)
        tilts.append(float(np.rad2deg(np.arccos(np.clip(d.body("xb_link").xmat[8], -1, 1)))))
    check("body_tilt_record", max(tilts) <= 10)
    plan_path = trial.parents[1] / "lifecycle/plan.json"
    plan = json.loads(plan_path.read_text())
    reference = np.asarray(plan["prepare_waypoints"][0]["qpos"])
    names = ["xt_joint_1", "xt_joint_2", "dt_joint", "yb_joint", "xb_joint"]
    names += [f"{side}_joint_{i}" for side in ["l", "r"] for i in range(1, 8)]
    qids = [m.jnt_qposadr[m.joint(name).id] for name in names]
    final_error = float(np.max(np.abs(d.qpos[qids] - reference[qids])))
    check("final_standing_hanging", final_error < 0.002)
    payload = params["runtime_payload"]
    config = json.loads((trial.parent / "review/scene_config.json").read_text())
    target_ml = 0.8 * config["payload"]["cup_capacity_ml"]
    if service:
        events = params["business_events"]
        business = audit_business_sequence(
            events, payload, config["payload"]["cup_capacity_ml"], expected_cups
        )
        check("business_sequence_and_reserve", business["passed"])
        served = [event for event in events if event["action"] == "serve_cup"]
        identities = [event["cup_id"] for event in served]
        check(
            "requested_cups_completed",
            params.get("service_completed") is True
            and len(identities) == len(set(identities)) == expected_cups
            and events[-1]["action"] == "exit"
            and events[-1]["pending_cups"] == 0,
        )
        check(
            "all_served_volumes",
            all(
                abs(payload["vessel_volumes_m3"][name] * 1e6 - target_ml) < 1e-6
                for name in identities
            ),
        )
        check(
            "payload_conservation",
            all(
                abs(
                    sum(row["runtime_payload"]["vessel_volumes_m3"].values())
                    - row["runtime_payload"]["initial_volume_m3"]
                    - row["runtime_payload"]["external_refill_volume_m3"]
                )
                < 1e-12
                for row in trace
                if "runtime_payload" in row
            ),
        )
        check(
            "refill_ledger",
            len(payload["external_events"])
            == sum(event["action"] == "external_refill" for event in events),
        )
        check(
            "recorded_motion_counts",
            stages.count("prescribed_payload_transfer") == expected_cups
            and stages.count("pot_close") == sum(event["action"] == "take_pot" for event in events)
            and stages.count("pot_release")
            == sum(event["action"] == "return_pot" for event in events)
            and stages.count("wait_for_external_refill") == len(payload["external_events"]),
        )
        for i, event in enumerate(payload["external_events"]):
            before = next(row for row in reversed(trace) if row["time_s"] <= event["time_s"])
            check(
                f"refill_{i}_supported",
                all(
                    before["predicates"].get(key) is True
                    for key in [
                        "pot_stable_after_release",
                        "both_grippers_empty",
                        "previous_served_cups_preserved",
                    ]
                ),
            )
    else:
        check(
            "coffee_sequence_completed",
            params.get("lifecycle_baseline_continuous_completed") is True
            and params.get("full_geometry_sequence_completed") is True,
        )
        check("single_cup_volume", abs(payload["cup_volume_m3"] * 1e6 - target_ml) < 1e-6)
        check(
            "payload_conservation",
            abs(payload["cup_volume_m3"] + payload["pot_volume_m3"] - payload["initial_volume_m3"])
            < 1e-12,
        )
    check(
        "checkpoint_hashes",
        all(
            sha(path) == json.loads(path.with_suffix(".json").read_text())["npz_sha256"]
            for path in trial.glob("*.npz")
        ),
    )
    snapshots = trial / "implementation_snapshots"
    manifest = json.loads((snapshots / "captured_during_run.json").read_text())
    check(
        "captured_source_hashes",
        all(sha(snapshots / Path(name).name) == digest for name, digest in manifest.items()),
    )
    check("controller_identity", sha(trial / "experiment.py") in manifest.values())
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "trial": str(trial),
        "scope": "continuous_simulation_service" if service else "continuous_baseline_lifecycle",
        "simulation_seconds": result["simulation_seconds"],
        "maximum_trace_body_tilt_deg": max(tilts),
        "final_joint_error_rad": final_error,
        "maximum_finger_load_n": params["maximum_finger_normal_resultant_n_all_steps"],
        "maximum_pour_speed_deg_s": params["maximum_pour_angular_speed_deg_s"],
        "model_sha256": sha(model_path),
        "controller_sha256": sha(trial / "experiment.py"),
        "trace_sha256": sha(trial / "trace.json"),
        "result_sha256": sha(trial / "result.json"),
        "physical_motion_allowed": False,
        "flow_simulated": False,
        "learned_policy_evaluated": False,
        "business_sequence_audit": business if service else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--service", action="store_true")
    parser.add_argument("--expected-cups", type=int, default=1)
    args = parser.parse_args()
    result = audit(args.trial, service=args.service, expected_cups=args.expected_cups)
    (args.trial / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
