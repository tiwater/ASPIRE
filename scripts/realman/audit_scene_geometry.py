#!/usr/bin/env python3
"""Run offline, annotated surface measurements on archived RGB-D captures."""

import argparse
import hashlib
import json
from pathlib import Path

from cap_harness.realman.scene_metrology import audit_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    definitions = json.loads(args.annotations.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "purpose": "offline geometric estimates, not physical qualification or motion targets",
        "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "cases": [],
    }
    for case in definitions["cases"]:
        capture = args.root / case["capture"]
        measured = audit_capture(capture, case["annotations"])
        result["cases"].append(measured)
        print(case["capture"], "complete", flush=True)
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "annotations.json").write_bytes(args.annotations.read_bytes())


if __name__ == "__main__":
    main()
