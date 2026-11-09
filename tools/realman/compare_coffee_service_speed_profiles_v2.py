#!/usr/bin/env python3
"""Compare stage durations between two independently completed coffee services.

The report keeps configured reference factors separate from measured elapsed
time. A phase can remain slower than its nominal factor because endpoint holds,
support recovery, and fixed precision segments are included in stage durations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def speed_segment_summary(result):
    rows = result.get("parameters", {}).get("speed_segments", [])
    summary = {}
    sequence = []
    for row in rows:
        stage = row["stage"]
        sequence.append(stage)
        summary.setdefault(stage, []).append(
            {
                "factor": row.get("factor"),
                "planned_ticks": row.get("planned_ticks"),
                "ramp_seconds": row.get("ramp_seconds"),
            }
        )
    return sequence, summary


def summarize_reference_factor_attainment(by_stage):
    counts = {}
    for comparison in by_stage.values():
        for row in comparison.get("matched", []):
            factor = row.get("candidate_reference_factor")
            if factor is None:
                continue
            bucket = counts.setdefault(str(factor), {"segments": 0, "at_least_reference_factor": 0})
            bucket["segments"] += 1
            bucket["at_least_reference_factor"] += int(row.get("observed_at_least_reference_factor") is True)
    return counts


def summarize_speed_targets(by_stage, candidate_speed_segments):
    """Separate comparison integrity from measured 4x/2x target attainment."""
    targets = {}
    for target in (4, 2):
        expected_segments = sum(
            1
            for rows in candidate_speed_segments.values()
            for row in rows
            if row.get("factor") == target
        )
        measured = []
        for stage, comparison in by_stage.items():
            for row in comparison.get("matched", []):
                if row.get("candidate_reference_factor") == target:
                    measured.append({"stage": stage, **row})
        reached = [row for row in measured if row.get("observed_at_least_reference_factor") is True]
        below = [row for row in measured if row.get("observed_at_least_reference_factor") is False]
        unverified = [row for row in measured if row.get("observed_at_least_reference_factor") is None]
        unmatched = max(0, expected_segments - len(measured))
        targets[str(target)] = {
            "target_speedup": target,
            "expected_segments": expected_segments,
            "measured_segments": len(measured),
            "segments_at_or_above_target": len(reached),
            "segments_below_target": len(below),
            "segments_unverified": len(unverified),
            "unmatched_segments": unmatched,
            "all_segments_met": expected_segments > 0 and len(reached) == expected_segments,
            "below_target": below,
            "unverified": unverified,
        }
    return {
        "scope": "measured stage durations relative to the supplied baseline trial; factors 4 and 2 denote the candidate's free and contact speed policies",
        "targets": targets,
        "all_measured_targets_met": all(row["all_segments_met"] for row in targets.values()),
    }


def load_trial(path: Path):
    result = json.loads((path / "result.json").read_text())
    audit = json.loads((path / "audit.json").read_text())
    if result.get("experiment_success") is not True or result.get("error") is not None:
        raise ValueError(f"trial did not complete successfully: {path}")
    if audit.get("passed") is not True or not all(audit.get("checks", {}).values()):
        raise ValueError(f"trial's continuous service audit did not pass: {path}")
    if audit.get("result_sha256") != sha(path / "result.json"):
        raise ValueError(f"stale result audit: {path}")
    if audit.get("trace_sha256") != sha(path / "trace.json"):
        raise ValueError(f"stale trajectory audit: {path}")
    if audit.get("scope") != "continuous_simulation_service_with_assumed_staff_pickup":
        raise ValueError(f"wrong service audit scope: {path}")
    delivery = json.loads((path / "delivery_order_audit.json").read_text())
    if (
        delivery.get("passed") is not True
        or not all(delivery.get("checks", {}).values())
        or delivery.get("result_sha256") != sha(path / "result.json")
        or delivery.get("trace_sha256") != sha(path / "trace.json")
    ):
        raise ValueError(f"missing, failed, or stale delivery-order audit: {path}")
    release = json.loads((path / "release_force_audit.json").read_text())
    if (
        release.get("passed") is not True
        or not all(release.get("checks", {}).values())
        or release.get("result_sha256") != sha(path / "result.json")
        or release.get("release_load_sha256") != sha(path / "release_load.json")
    ):
        raise ValueError(f"missing, failed, or stale release-force audit: {path}")
    trace = json.loads((path / "trace.json").read_text())
    groups = []
    for row in trace:
        if not groups or groups[-1]["stage"] != row["stage"]:
            groups.append({"stage": row["stage"], "start_s": row["time_s"]})
    for index, group in enumerate(groups):
        group["end_s"] = (
            groups[index + 1]["start_s"]
            if index + 1 < len(groups)
            else result["simulation_seconds"]
        )
        group["duration_s"] = group["end_s"] - group["start_s"]
    manifest_path = path / "implementation_snapshots" / "captured_during_run.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return result, audit, groups, manifest


def compare(base: Path, candidate: Path, *, unaccelerated_baseline: bool = False):
    base_result, base_audit, base_groups, base_manifest = load_trial(base)
    cand_result, cand_audit, cand_groups, cand_manifest = load_trial(candidate)
    policy_path = candidate / "speed_policy_audit.json"
    if not policy_path.exists():
        raise ValueError("candidate is missing the per-segment speed-policy audit")
    policy_audit = json.loads(policy_path.read_text())
    if policy_audit.get("passed") is not True or policy_audit.get("result_sha256") != sha(candidate / "result.json"):
        raise ValueError("candidate speed-policy audit is failed or stale")
    base_speed_sequence, base_speed_segments = speed_segment_summary(base_result)
    cand_speed_sequence, cand_speed_segments = speed_segment_summary(cand_result)
    if base_result.get("model_sha256") != cand_result.get("model_sha256"):
        raise ValueError("model hashes differ; comparison is not controlled")
    ba = base_result["parameters"]["arguments"]
    ca = cand_result["parameters"]["arguments"]
    base_fixture = base_result.get("parameters", {}).get("initial_fixture", {})
    candidate_fixture = cand_result.get("parameters", {}).get("initial_fixture", {})
    same_initial_fixture = (
        "yaw_deg" in base_fixture
        and "yaw_deg" in candidate_fixture
        and abs(base_fixture["yaw_deg"] - candidate_fixture["yaw_deg"]) <= 0.01
        and base_fixture.get("position_drift_m", float("inf")) <= 0.001
        and candidate_fixture.get("position_drift_m", float("inf")) <= 0.001
    )
    ignored = {"name"}
    argument_differences = {
        key: {"baseline": ba.get(key), "candidate": ca.get(key)}
        for key in sorted(set(ba) | set(ca))
        if ba.get(key) != ca.get(key) and key not in ignored
    }
    speed_keys = {"free_speed_scale", "contact_speed_scale", "pot_contact_speed_scale"}
    if unaccelerated_baseline:
        expected_factors = {
            "free_speed_scale": (1, 4),
            "contact_speed_scale": (1, 2),
            "pot_contact_speed_scale": (1, 2),
        }
        for key, (base_factor, candidate_factor) in expected_factors.items():
            if ba.get(key) != base_factor or ca.get(key) != candidate_factor:
                raise ValueError(
                    f"controlled 1x-to-target comparison expects {key} {base_factor}x -> {candidate_factor}x"
                )
        if not math.isclose(float(ba.get("pour_angular_rate", math.nan)), 0.04, rel_tol=0, abs_tol=1e-12):
            raise ValueError("controlled baseline expects the verified 0.04 rad/s pour rate")
        if not math.isclose(float(ca.get("pour_angular_rate", math.nan)), 0.0385, rel_tol=0, abs_tol=1e-12):
            raise ValueError("accelerated candidate must use the guard-compliant 0.0385 rad/s pour rate")
        expected_speed_differences = speed_keys | {"pour_angular_rate"}
    else:
        if ca.get("free_speed_scale") != ba.get("free_speed_scale"):
            raise ValueError("free-space speed factor changed; isolate contact-speed comparison")
        if ca.get("pot_contact_speed_scale") != ba.get("pot_contact_speed_scale"):
            raise ValueError("pot contact speed factor changed; isolate contact-speed comparison")
        expected_speed_differences = {"contact_speed_scale"}
    unexpected = set(argument_differences) - expected_speed_differences
    if unexpected:
        raise ValueError(f"unexpected run argument differences: {sorted(unexpected)}")

    common_paths = set(base_manifest) & set(cand_manifest)
    shared_source_differences = {
        path: {"baseline": base_manifest[path], "candidate": cand_manifest[path]}
        for path in sorted(common_paths)
        if base_manifest[path] != cand_manifest[path]
    }
    baseline_only_sources = sorted(set(base_manifest) - set(cand_manifest))
    candidate_only_sources = sorted(set(cand_manifest) - set(base_manifest))
    baseline_driver = next((p for p in base_manifest if Path(p).name.startswith("run_service_")), None)
    candidate_driver = next((p for p in cand_manifest if Path(p).name.startswith("run_service_")), None)
    if unaccelerated_baseline:
        expected_manifest_delta = (
            len(baseline_only_sources) == 2
            and len(candidate_only_sources) == 2
            and baseline_driver in baseline_only_sources
            and candidate_driver in candidate_only_sources
            and sum(Path(p).name.endswith("_command.json") for p in baseline_only_sources) == 1
            and sum(Path(p).name.endswith("_command.json") for p in candidate_only_sources) == 1
        )
    else:
        expected_manifest_delta = (
            len(baseline_only_sources) == 2
            and len(candidate_only_sources) == 3
            and baseline_driver in baseline_only_sources
            and candidate_driver in candidate_only_sources
            and sum(Path(p).name.endswith("_command.json") for p in baseline_only_sources) == 1
            and sum(Path(p).name.endswith("_command.json") for p in candidate_only_sources) == 1
            and sum(Path(p).name == "coffee_speed.py" for p in candidate_only_sources) == 1
        )

    by_stage = {}
    names = sorted({g["stage"] for g in base_groups} | {g["stage"] for g in cand_groups})
    for name in names:
        bg = [g for g in base_groups if g["stage"] == name]
        cg = [g for g in cand_groups if g["stage"] == name]
        if not bg or not cg:
            by_stage[name] = {"comparable": False, "baseline_repetitions": len(bg), "candidate_repetitions": len(cg)}
            continue
        matched = min(len(bg), len(cg))
        rows = []
        for index in range(matched):
            base_s, cand_s = bg[index]["duration_s"], cg[index]["duration_s"]
            candidate_reference_segments = cand_speed_segments.get(name, [])
            candidate_reference_factor = (
                candidate_reference_segments[index]["factor"]
                if index < len(candidate_reference_segments)
                else None
            )
            observed = base_s / cand_s if cand_s > 0 else None
            factor_attainment = (
                observed / candidate_reference_factor
                if observed is not None and candidate_reference_factor
                else None
            )
            rows.append({
                "occurrence": index,
                "baseline_duration_s": base_s,
                "candidate_duration_s": cand_s,
                "candidate_reference_factor": candidate_reference_factor,
                "observed_speedup": observed,
                "observed_speedup_over_reference_factor": factor_attainment,
                "observed_at_least_reference_factor": (
                    observed >= candidate_reference_factor
                    if observed is not None and candidate_reference_factor
                    else None
                ),
            })
        by_stage[name] = {
            "comparable": len(bg) == len(cg),
            "baseline_repetitions": len(bg),
            "candidate_repetitions": len(cg),
            "matched": rows,
        }

    same_speed_stage_coverage = base_speed_sequence == cand_speed_sequence
    if unaccelerated_baseline:
        base_trace_stages = {row["stage"] for row in base_groups}
        candidate_trace_stages = {row["stage"] for row in cand_groups}
        same_speed_stage_coverage = all(
            stage in base_trace_stages and stage in candidate_trace_stages
            for stage in cand_speed_sequence
        )

    checks = {
        "same_model": base_result.get("model_sha256") == cand_result.get("model_sha256"),
        "both_time_zero_initial_states_qualified": base_audit.get("checks", {}).get("time_zero_initial_state") is True and cand_audit.get("checks", {}).get("time_zero_initial_state") is True,
        "same_initial_fixture": same_initial_fixture,
        "both_continuous_service_audits_pass": base_audit.get("passed") and cand_audit.get("passed"),
        "same_free_speed_factor": unaccelerated_baseline or ba.get("free_speed_scale") == ca.get("free_speed_scale"),
        "same_pot_contact_factor": unaccelerated_baseline or ba.get("pot_contact_speed_scale") == ca.get("pot_contact_speed_scale"),
        "free_speed_factor_is_4x": ca.get("free_speed_scale") == 4,
        "pot_contact_factor_is_2x": ca.get("pot_contact_speed_scale") == 2,
        "contact_speed_factor_changed_1x_to_2x": ba.get("contact_speed_scale") == 1 and ca.get("contact_speed_scale") == 2,
        "controlled_unaccelerated_baseline_factors": (not unaccelerated_baseline) or all(
            ba.get(key) == value for key, value in {
                "free_speed_scale": 1,
                "contact_speed_scale": 1,
                "pot_contact_speed_scale": 1,
            }.items()
        ),
        "only_expected_speed_arguments_changed": set(argument_differences) == expected_speed_differences,
        "both_implementation_snapshots_present": bool(base_manifest) and bool(cand_manifest),
        "candidate_per_segment_speed_policy_audit_passes": policy_audit.get("passed") is True,
        "candidate_speed_policy_exercises_free_and_contact_segments": policy_audit.get("independently_clear_segments", 0) > 0 and policy_audit.get("contact_or_supported_segments", 0) > 0,
        "shared_runtime_sources_unchanged": not shared_source_differences,
        "only_expected_driver_and_command_files_differ": expected_manifest_delta,
        "same_requested_cups": ba.get("requested_cups") == ca.get("requested_cups"),
        "same_refill_count": ba.get("demo_refills") == ca.get("demo_refills"),
        "same_stage_occurrences": all(row["comparable"] for row in by_stage.values()),
        "same_reference_motion_stage_sequence": same_speed_stage_coverage,
    }
    speed_target_counts = summarize_reference_factor_attainment(by_stage)
    measured_speed_targets = summarize_speed_targets(by_stage, cand_speed_segments)
    scope = (
        "controlled same-scene, same-order, same-load full-service comparison of a 1x run against the configured 4x free / 2x contact profile; candidate pour rate is trimmed from 0.04 to 0.0385 rad/s to retain the 5 deg/s guard after the 2x contact timing scale; clear-transport integration timestep is matched; reports simulated stage durations and does not claim real-robot speed"
        if unaccelerated_baseline
        else "same-scene full-service measured stage-duration comparison; driver scripts are separately hashed and may encode different speed policies; descriptive comparison, not causal attribution to one argument or a claim about real-robot speeds"
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "scope": scope,
        "baseline_trial": str(base.resolve()),
        "candidate_trial": str(candidate.resolve()),
        "baseline_factors": {
            "free_space": ba.get("free_speed_scale"),
            "contact": ba.get("contact_speed_scale"),
            "pot_contact": ba.get("pot_contact_speed_scale"),
            "postpour_policy": base_result["parameters"].get("postpour_return_speed_policy", "fixed 2x"),
        },
        "candidate_factors": {
            "free_space": ca.get("free_speed_scale"),
            "contact": ca.get("contact_speed_scale"),
            "pot_contact": ca.get("pot_contact_speed_scale"),
            "postpour_policy": cand_result["parameters"].get("postpour_return_speed_policy", "fixed 2x"),
        },
        "initial_fixture_comparison": {
            "baseline": base_fixture,
            "candidate": candidate_fixture,
            "within_tolerance": same_initial_fixture,
            "tolerances": {"yaw_deg": 0.01, "position_drift_m_each": 0.001},
        },
        "argument_differences": argument_differences,
        "implementation_snapshot": {
            "baseline_manifest_present": bool(base_manifest),
            "candidate_manifest_present": bool(cand_manifest),
            "shared_runtime_sources_unchanged": not shared_source_differences,
            "shared_source_differences": shared_source_differences,
            "baseline_only_sources": baseline_only_sources,
            "candidate_only_sources": candidate_only_sources,
            "only_expected_driver_and_command_files_differ": expected_manifest_delta,
            "baseline_driver": {"path": baseline_driver, "sha256": base_manifest.get(baseline_driver)},
            "candidate_driver": {"path": candidate_driver, "sha256": cand_manifest.get(candidate_driver)},
            "driver_changed": base_manifest.get(baseline_driver) != cand_manifest.get(candidate_driver),
        },
        "stage_duration_comparisons": by_stage,
        "observed_reference_factor_attainment": speed_target_counts,
        "measured_speed_target_assessment": measured_speed_targets,
        "reference_speed_segments": {
            "baseline": base_speed_segments,
            "candidate": cand_speed_segments,
            "note": "Configured peak reference factors and ramps are listed separately from measured stage durations.",
        },
        "candidate_speed_policy_audit": policy_audit,
        "baseline_total_simulation_s": base_result.get("simulation_seconds"),
        "candidate_total_simulation_s": cand_result.get("simulation_seconds"),
        "baseline_result_sha256": sha(base / "result.json"),
        "candidate_result_sha256": sha(candidate / "result.json"),
        "baseline_trace_sha256": sha(base / "trace.json"),
        "candidate_trace_sha256": sha(candidate / "trace.json"),
        "physical_motion_allowed": False,
        "comparison_mode": "controlled_unaccelerated_baseline" if unaccelerated_baseline else "contact_speed_isolation",
        "fluid_simulated": False,
        "staff_pickup_assumed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--unaccelerated-baseline",
        action="store_true",
        help="require baseline 1x and candidate 4x free / 2x contact and pot-contact factors",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing timing report")
    report = compare(args.baseline, args.candidate, unaccelerated_baseline=args.unaccelerated_baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "output": str(args.output)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
