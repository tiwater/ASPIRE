"""Check immediate simulated pickup and clearance before the next cup starts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def inspect_order(trace, events, placements, *, maximum_pickup_delay_s=.5):
    identities = [event["cup_id"] for event in events]
    errors = []
    if not identities or len(set(identities)) != len(identities):
        errors.append("missing or duplicate collected cup identities")
    if set(placements) != set(identities):
        errors.append("placement and collection identities differ")
    delays = {}
    for event in events:
        cup = event["cup_id"]
        if cup not in placements:
            continue
        delay = event["time_s"] - placements[cup]
        delays[cup] = delay
        if not math.isfinite(delay) or not 0 <= delay <= maximum_pickup_delay_s:
            errors.append(f"{cup}: pickup not immediate after completed placement/retraction")
    previous_count, first_active, last_time = 0, [], -math.inf
    for row in trace:
        time = row["time_s"]
        if not math.isfinite(time) or time < last_time:
            errors.append("invalid trace clock")
            break
        last_time = time
        payload = row.get("runtime_payload")
        if payload is None:
            continue
        active = payload["active_cup"]
        ledger = row.get("staff_collection", {}).get("events", [])
        count = len(ledger)
        if ledger != events[:count] or count < previous_count or count > previous_count + 1:
            errors.append(f"{time}: pickup ledger skipped, regressed or changed")
            break
        if count > previous_count and abs(ledger[-1]["time_s"] - time) > 1e-6:
            errors.append(f"{time}: pickup is not recorded at its actual event boundary")
            break
        previous_count = count
        if active not in identities:
            errors.append(f"{time}: unexpected active cup {active}")
            break
        index = identities.index(active)
        if count < index:
            errors.append(f"{time}: {active} started before prior cups were collected")
            break
        if not first_active or first_active[-1] != active:
            first_active.append(active)
    if first_active != identities or previous_count != len(events):
        errors.append("active cup order or final collection ledger is incomplete")
    return {"passed": not errors, "errors": errors, "pickup_delays_s": delays,
            "maximum_pickup_delay_s": maximum_pickup_delay_s,
            "active_cup_order": first_active,
            "scope": "simulated delivery timing/order only; not full service qualification"}


def audit(trial):
    trial = Path(trial)
    result = json.loads((trial / "result.json").read_text())
    events = result["parameters"]["staff_collection"]["events"]
    placements = {}
    for path in sorted(trial.glob("*_cup_placed_right_retracted.json")):
        meta = json.loads(path.read_text())
        cup = meta["runtime_payload"]["active_cup"]
        if cup in placements:
            raise ValueError("duplicate physical placement checkpoint")
        placements[cup] = meta["time_s"]
    report = inspect_order(json.loads((trial / "trace.json").read_text()), events, placements)
    report.update(physical_motion_allowed=False, full_service_qualified=False,
                  trace_sha256=hashlib.sha256((trial / "trace.json").read_bytes()).hexdigest(),
                  result_sha256=hashlib.sha256((trial / "result.json").read_bytes()).hexdigest())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    out = args.trial / "delivery_order_audit.json"
    if out.exists():
        raise FileExistsError("preserve prior audit")
    report = audit(args.trial)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
