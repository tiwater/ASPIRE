#!/usr/bin/env python3
"""Audit observed per-segment coffee pot transport speed decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(trial: Path) -> dict:
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text())
    if result.get("experiment_success") is not True or result.get("error") is not None:
        raise ValueError("simulation did not complete successfully")
    parameters = result.get("parameters", {})
    arguments = parameters.get("arguments", {})
    free_factor = float(arguments.get("free_speed_scale", math.nan))
    contact_factor = float(arguments.get("pot_contact_speed_scale", math.nan))
    if free_factor != 4 or contact_factor != 2:
        raise ValueError("candidate must request 4x free transport and 2x pot contact")
    segments = parameters.get("speed_segments", [])
    checked = []
    for segment in segments:
        stage = str(segment.get("stage", ""))
        if not stage.startswith(("pot_extract_", "pot_return_")):
            continue
        decision = segment.get("pot_transport_speed_decision")
        if not isinstance(decision, dict):
            raise ValueError(f"missing clearance evidence for {stage}")
        clear = decision.get("pot_clear_of_machine")
        support = float(decision.get("upward_platform_support_n", math.nan))
        threshold = float(decision.get("independent_clearance_threshold_n", math.nan))
        factor = float(segment.get("factor", math.nan))
        peak_factor = float(segment.get("reference_peak_factor", math.nan))
        planned_ticks = segment.get("planned_ticks", 0)
        ramp_seconds = float(segment.get("ramp_seconds", math.nan))
        recorded_factor = float(decision.get("factor", math.nan))
        if not isinstance(clear, bool):
            raise ValueError(f"invalid clearance predicate for {stage}")
        if isinstance(planned_ticks, bool) or not isinstance(planned_ticks, int):
            raise ValueError(f"invalid planned tick count for {stage}")
        if not all(math.isfinite(x) for x in (support, threshold, factor, peak_factor, ramp_seconds, recorded_factor)):
            raise ValueError(f"non-finite speed evidence for {stage}")
        if support < 0 or threshold != 0.02 or planned_ticks < 1 or ramp_seconds < 0:
            raise ValueError(f"unexpected support evidence or threshold for {stage}")
        expected = free_factor if clear and support <= threshold else contact_factor
        if not math.isclose(factor, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"speed factor contradicts clearance evidence for {stage}")
        if not math.isclose(recorded_factor, factor, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"decision and segment factors differ for {stage}")
        if not math.isclose(peak_factor, factor, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"reference peak factor differs from selected factor for {stage}")
        checked.append({
            "stage": stage,
            "factor": factor,
            "planned_ticks": planned_ticks,
            "ramp_seconds": ramp_seconds,
            "reference_peak_factor": peak_factor,
            "expected_factor": expected,
            "pot_clear_of_machine": clear,
            "upward_platform_support_n": support,
            "independent_clearance_threshold_n": threshold,
        })
    if not checked:
        raise ValueError("trial has no recorded pot extraction or return speed segments")
    clear_segments = [row for row in checked if row["pot_clear_of_machine"] and row["upward_platform_support_n"] <= 0.02]
    contact_segments = [row for row in checked if row not in clear_segments]
    if not clear_segments or not contact_segments:
        raise ValueError("trial must exercise both independently clear and contact/supported transport")
    return {
        "passed": True,
        "scope": "per-segment configured reference factor agrees with recorded machine-clearance and platform-support evidence; not a measured real-robot speed claim",
        "trial": str(trial.resolve()),
        "result_sha256": sha(result_path),
        "free_speed_factor": free_factor,
        "pot_contact_speed_factor": contact_factor,
        "segments_checked": len(checked),
        "independently_clear_segments": len(clear_segments),
        "contact_or_supported_segments": len(contact_segments),
        "segments": checked,
        "physical_motion_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.trial)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"preserve existing report: {args.output}")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "segments_checked", "independently_clear_segments", "contact_or_supported_segments")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
