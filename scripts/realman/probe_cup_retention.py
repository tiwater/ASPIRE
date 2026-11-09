#!/usr/bin/env python3
"""Counterfactual rack diagnostic: remove cup 0, measure the remaining cups.

This deliberately changes an object's pose to isolate the rack mechanism.
It is never a robot policy trial and cannot establish extraction success.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def probe(model_path, friction, seconds, settle_seconds):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    cups = [
        model.body(i).name
        for i in range(model.nbody)
        if model.body(i).name.startswith("cup_") and model.body(i).name != "cup_dispenser"
    ]
    cups = sorted(cups)
    for i in range(model.ngeom):
        if model.body(model.geom_bodyid[i]).name in cups:
            model.geom_friction[i, 0] = friction
    for _ in range(round(settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    references = {n: data.body(n).xpos.copy() for n in cups[1:]}
    # Explicit counterfactual intervention, NOT an extraction action.
    data.qpos[model.joint("cup_0_free").qposadr[0]] += 1.0
    mujoco.mj_forward(model, data)
    maximum_drift = dict.fromkeys(references, 0.0)
    maximum_tilt = dict.fromkeys(references, 0.0)
    samples = []
    for step in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if step % max(1, round(0.1 / model.opt.timestep)):
            continue
        mujoco.mj_forward(model, data)
        sample = {"time_s": float(data.time), "cups": {}}
        for name, position in references.items():
            drift = float(np.linalg.norm(data.body(name).xpos - position))
            tilt = float(np.degrees(np.arccos(np.clip(data.body(name).xmat[8], -1, 1))))
            maximum_drift[name] = max(maximum_drift[name], drift)
            maximum_tilt[name] = max(maximum_tilt[name], tilt)
            sample["cups"][name] = {"drift_m": drift, "tilt_deg": tilt}
        samples.append(sample)
    return {
        "cup_friction": friction,
        "rack_pad_friction": float(model.geom("retention_pad0").friction[0]),
        "settle_seconds": settle_seconds,
        "observation_seconds": seconds,
        "max_drift_m": maximum_drift,
        "max_tilt_deg": maximum_tilt,
        "mujoco_warnings": data.warning.number.tolist(),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--friction", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--settle-seconds", type=float, default=1)
    args = parser.parse_args()
    if not all(np.isfinite(f) and 0 < f <= 2 for f in args.friction):
        parser.error("friction must be finite and in (0, 2]")
    if not all(np.isfinite(t) and 0 < t <= 120 for t in [args.seconds, args.settle_seconds]):
        parser.error("durations must be finite and in (0, 120]")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "purpose": "counterfactual rack retention diagnostic, not policy evaluation",
        "coffee_task_success": None,
        "physical_motion_allowed": False,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "cases": [],
    }
    for friction in args.friction:
        case = probe(args.model, friction, args.seconds, args.settle_seconds)
        result["cases"].append(case)
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in case.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
