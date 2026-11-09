"""Qualify fresh coffee service with operator-assumed pickup at a fixed slot.

Retains the original continuous-session checks. Only the final cup relation
changes: each cup must have been placed, released, and explicitly collected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_coffee_session import audit as audit_session
from audit_cup_delivery import audit as audit_delivery


def audit(trial: Path, expected_cups=3):
    trial = trial.resolve()
    result = json.loads((trial / "result.json").read_text())
    report = audit_session(trial, service=True, expected_cups=expected_cups)
    delivery = audit_delivery(trial, expected_cups)
    original_final_check = report["checks"]["final_object_relations"]
    relations = [
        "pot_on_hotplate", "pot_stable_after_release", "lid_outlet_aligned",
        "pot_upright", "left_gripper_released", "remaining_cups_retained",
        "previous_served_cups_preserved", "both_grippers_empty", "served_cups_collected",
    ]
    report["checks"]["final_object_relations"] = all(
        result["predicates"].get(key) is True for key in relations
    )
    report["checks"]["independent_fixed_delivery_and_collection"] = delivery["passed"]
    params = result["parameters"]
    report["checks"]["explicit_pickup_assumption"] = (
        params.get("delivery_mode") == "staff_collect_fixed_slot"
        and params.get("staff_pickup_assumed") is True
        and params.get("external_actor_object_pose_writes") is True
        and params["staff_collection"].get("simulation_only") is True
    )
    report.update(
        passed=all(report["checks"].values()),
        scope="continuous_simulation_service_with_assumed_staff_pickup",
        delivery_audit=delivery,
        final_relation_change={
            "operator_assumption": "staff immediately collects each placed and released cup",
            "legacy_cup_still_on_table_check": original_final_check,
            "required_final_relations": relations,
            "replacement_evidence": "per-cup placement and pickup checkpoints plus continuous trajectory",
        },
        human_motion_simulated=False,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--expected-cups", type=int, default=3)
    args = parser.parse_args()
    out = args.trial / "audit.json"
    if out.exists():
        raise FileExistsError("preserve prior audit; use a new trial")
    report = audit(args.trial, args.expected_cups)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
