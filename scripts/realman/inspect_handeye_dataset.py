"""Offline acquisition quality audit. Does not move a robot or qualify hand-eye."""

import argparse
import json
from pathlib import Path

from solve_handeye import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--robust-corners", action="store_true")
    args = parser.parse_args()
    records, _, _, _, _ = load_records(
        args.manifest.parent,
        json.loads(args.manifest.read_text()),
        json.loads(Path("configs/realman/calibration_board.json").read_text()),
        args.arm,
        robust_corners=args.robust_corners,
    )
    summaries = [
        dict(
            sample=r["sample"],
            role=r["role"],
            corners_used=len(r["corners"]),
            pnp_rms_px=r["pnp_rms_px"],
            detector=r["detector"],
            excluded_ids=r["corner_quality"]["excluded_ids"] if r["corner_quality"] else [],
        )
        for r in records
    ]
    report = dict(acquisition_checks_passed=True, handeye_qualified=False, records=summaries)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(count=len(records), latest=summaries[-1] if summaries else None)))


if __name__ == "__main__":
    main()
