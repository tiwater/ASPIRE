"""Independently inspect persistent served cups in recorded simulation samples.

This supplements the continuous-session audit and runtime guards. Recorded
poses/contact forces cannot establish unrecorded between-sample behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect_sample(position, up, reference, contacts, geom_bodies, cup, config):
    """Check measured pose and recorded contacts without using success flags."""
    position, reference = np.asarray(position), np.asarray(reference)
    half = np.asarray(config["table"]["size_m"][:2]) / 2
    center = np.asarray(config["table"]["center_xy_m"])
    margin = float(np.min(half - np.abs(position[:2] - center)))
    margin -= config["cup"]["bottom_radius_m"]
    drift = float(np.linalg.norm(position - reference))
    tilt = float(np.rad2deg(np.arccos(np.clip(up, -1, 1))))
    support, unexpected = False, []
    for contact in contacts:
        force = contact["normal_force_n"]
        if not np.isfinite(force):
            raise ValueError("nonfinite contact force")
        if force <= 0.02:
            continue
        g1, g2 = contact["geom1"], contact["geom2"]
        b1, b2 = geom_bodies[g1], geom_bodies[g2]
        if cup not in {b1, b2}:
            continue
        other_geom = g2 if b1 == cup else g1
        other_body = b2 if b1 == cup else b1
        if other_geom == "tabletop":
            support = True
        elif other_body != cup:
            unexpected.append(contact)
    checks = {
        "finite_pose": bool(np.isfinite(position).all() and np.isfinite(up)),
        "table_support": support,
        "footprint_margin": margin >= 0.02,
        "upright": tilt < 8,
        "position_preserved": drift < 0.01,
        "no_other_object_contact": not unexpected,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "margin_m": margin,
        "drift_m": drift,
        "tilt_deg": tilt,
        "unexpected_contacts": unexpected,
    }


def audit(trial: Path, expected_cups: int):
    import mujoco as mj

    trial = trial.resolve()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    trace_path, result_path = trial / "trace.json", trial / "result.json"
    trace = json.loads(trace_path.read_text())
    result = json.loads(result_path.read_text())
    model_path = trial.parent / "review/model/scene.xml"
    if sha(model_path) != result["model_sha256"]:
        raise ValueError("model identity mismatch")
    config = json.loads((trial.parent / "review/scene_config.json").read_text())
    model, cups = mj.MjModel.from_xml_path(str(model_path)), {}
    data = mj.MjData(model)
    geom_bodies = {
        model.geom(i).name: model.body(model.geom_bodyid[i]).name for i in range(model.ngeom)
    }
    for path in sorted(trial.glob("*_cup_placed_right_retracted.json")):
        meta = json.loads(path.read_text())
        if (
            sha(path.with_suffix(".npz")) != meta["npz_sha256"]
            or meta["model_sha256"] != result["model_sha256"]
        ):
            raise ValueError("placement checkpoint identity mismatch")
        cup = meta["runtime_payload"]["active_cup"]
        if cup in cups:
            raise ValueError("duplicate served-cup placement")
        with np.load(path.with_suffix(".npz")) as saved:
            mj.mj_setState(model, data, saved["state"], mj.mjtState.mjSTATE_INTEGRATION)
        if abs(float(data.time) - meta["time_s"]) > 1e-8:
            raise ValueError("placement checkpoint time mismatch")
        mj.mj_kinematics(model, data)
        cups[cup] = {
            "placed_at_s": meta["time_s"],
            "checkpoint_sha256": meta["npz_sha256"],
            "samples": 0,
            "reference_m": data.body(cup).xpos.copy().tolist(),
            "maximum_drift_m": 0.0,
            "maximum_tilt_deg": 0.0,
            "minimum_margin_m": None,
            "violations": [],
        }
    times = np.asarray([row["time_s"] for row in trace])
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("missing or unordered trajectory")
    for row in trace:
        data.qpos[:] = row["qpos"]
        mj.mj_kinematics(model, data)
        for cup, stats in cups.items():
            if row["time_s"] < stats["placed_at_s"] - 1e-8:
                continue
            position = data.body(cup).xpos.copy()
            if stats["samples"] == 0:
                # Saving a checkpoint does not force a trajectory sample.
                # Keep the measured checkpoint pose as the reference, and
                # require the next record within the normal 100 ms interval.
                delay = row["time_s"] - stats["placed_at_s"]
                if delay > 0.101:
                    raise ValueError("post-placement sample missing beyond recording interval")
                stats["first_sample_delay_s"] = float(delay)
            sample = inspect_sample(
                position,
                data.body(cup).xmat[8],
                stats["reference_m"],
                row["contacts"],
                geom_bodies,
                cup,
                config,
            )
            stats["samples"] += 1
            stats["maximum_drift_m"] = max(stats["maximum_drift_m"], sample["drift_m"])
            stats["maximum_tilt_deg"] = max(stats["maximum_tilt_deg"], sample["tilt_deg"])
            previous = stats["minimum_margin_m"]
            stats["minimum_margin_m"] = (
                sample["margin_m"] if previous is None else min(previous, sample["margin_m"])
            )
            if not sample["passed"]:
                stats["violations"].append(
                    {"time_s": row["time_s"], "stage": row["stage"], **sample}
                )
    served = [
        event["cup_id"]
        for event in result["parameters"]["business_events"]
        if event["action"] == "serve_cup"
    ]
    checks = {
        "expected_distinct_placements": len(cups) == expected_cups > 0,
        "business_identities_match": len(served) == len(set(served)) and set(served) == set(cups),
        "all_cups_preserved_in_record": all(
            s["samples"] >= 2 and not s["violations"] for s in cups.values()
        ),
        "sample_gap": float(np.max(np.diff(times))) <= 0.101,
        "record_reaches_terminal_state": abs(times[-1] - result["simulation_seconds"]) < 1e-6,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "cups": cups,
        "scope": "recorded served-cup poses and contacts; supplement to continuous-session audit",
        "maximum_sample_gap_s": float(np.max(np.diff(times))),
        "trace_sha256": sha(trace_path),
        "result_sha256": sha(result_path),
        "model_sha256": sha(model_path),
        "physical_motion_allowed": False,
        "full_workflow_qualified_by_this_report": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--expected-cups", type=int, default=3)
    args = parser.parse_args()
    report = audit(args.trial, args.expected_cups)
    output = args.trial / "served_cups_audit.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
