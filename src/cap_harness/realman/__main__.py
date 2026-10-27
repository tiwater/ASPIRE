"""CLI for RealBot commissioning and offline coffee-model iteration."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build MuJoCo scene from supplied URDF and parameter JSON")
    build.add_argument("--robot-config", type=Path, default=Path("configs/realman/robot_sim.json"))
    build.add_argument(
        "--scene-config", type=Path, default=Path("configs/realman/coffee_scene.json")
    )
    build.add_argument("--output", type=Path, default=Path("artifacts/realman/coffee"))
    doctor = sub.add_parser("doctor", help="report outstanding commissioning; never enables motion")
    doctor.add_argument("--robot-config", type=Path, default=Path("configs/realman/robot.json"))
    capture = sub.add_parser(
        "capture-summary", help="decode a recorded ROS capture and make RGB previews"
    )
    capture.add_argument("folder", type=Path)
    estimates = sub.add_parser(
        "estimate", help="derive capacity, fill target and contact-force priors"
    )
    estimates.add_argument(
        "--scene-config", type=Path, default=Path("configs/realman/coffee_scene.json")
    )
    calibration = sub.add_parser(
        "calibration-check", help="read-only board and CameraInfo precheck"
    )
    calibration.add_argument("folder", type=Path)
    calibration.add_argument(
        "--board-config", type=Path, default=Path("configs/realman/calibration_board.json")
    )
    for name in ("render", "viewer", "run", "ik", "validate"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--model", type=Path, default=Path("artifacts/realman/coffee/scene.xml"))
        if name == "render":
            cmd.add_argument("--output", type=Path, default=Path("artifacts/realman/coffee"))
        elif name == "run":
            cmd.add_argument("program", type=Path)
            cmd.add_argument("--output-root", type=Path, default=Path("artifacts/realman/runs"))
        elif name == "ik":
            cmd.add_argument("--arm", choices=["left", "right"], required=True)
            cmd.add_argument("--xyz", nargs=3, type=float, required=True)
            cmd.add_argument("--wxyz", nargs=4, type=float)
        elif name == "validate":
            cmd.add_argument("--seconds", type=float, default=2)
    args = p.parse_args()
    if args.command == "estimate":
        from .estimates import estimate_parameters

        result = estimate_parameters(json.loads(args.scene_config.read_text()))
    elif args.command == "calibration-check":
        from .calibration import inspect_capture

        result = inspect_capture(args.folder, json.loads(args.board_config.read_text()))
    elif args.command == "build":
        from .model import build_model

        config = json.loads(args.robot_config.read_text())
        result = build_model(Path(config["urdf"]), args.scene_config, args.output)
    elif args.command == "doctor":
        from .telemetry import commissioning_report

        result = commissioning_report(json.loads(args.robot_config.read_text()))
    elif args.command == "capture-summary":
        from .capture import summarize

        result = summarize(args.folder, previews=True)
    else:
        from .sim import CoffeeSimAdapter

        adapter = CoffeeSimAdapter(args.model)
        try:
            if args.command == "render":
                from PIL import Image

                args.output.mkdir(parents=True, exist_ok=True)
                for name in ("overview", "front", "side"):
                    Image.fromarray(adapter.camera(name).rgb).save(args.output / f"{name}.png")
                result = {
                    "output": str(args.output),
                    "views": ["overview.png", "front.png", "side.png"],
                }
            elif args.command == "viewer":
                import mujoco.viewer

                # Passive viewer displays the model for inspection. Physics runs
                # only through validate/run; opening the viewer does not drift objects.
                with mujoco.viewer.launch_passive(adapter.model, adapter.data) as viewer:
                    viewer.cam.lookat[:] = [0.55, 0, 0.85]
                    viewer.cam.distance = 2.6
                    viewer.cam.azimuth = 140
                    viewer.cam.elevation = -25
                    import time

                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                result = {"viewer_closed": True}
            elif args.command == "run":
                import hashlib
                import shutil
                import uuid

                from cap_harness.api import CapApi
                from cap_harness.artifacts import Serializer
                from cap_harness.run import run_program_with_api

                from .ik import SimulationIKProvider

                run_dir = args.output_root / ("run-" + uuid.uuid4().hex[:12])
                run_dir.mkdir(parents=True, exist_ok=False)
                serializer = Serializer(run_dir)
                shutil.copyfile(args.program, run_dir / "program.py")
                for filename in ("scene.xml", "build_report.json", "scene_parameters.json"):
                    source = args.model if filename == "scene.xml" else args.model.parent / filename
                    if source.exists():
                        shutil.copyfile(source, run_dir / filename)
                serializer.dump(run_dir / "initial_observation.json", adapter.get_observation())
                outcome = run_program_with_api(
                    api=CapApi(
                        adapter,
                        default_camera="front",
                        ik_providers={"realman_sim": SimulationIKProvider(adapter)},
                    ),
                    program_path=args.program,
                )
                from collections.abc import Mapping

                def plain(value):
                    if isinstance(value, Mapping):
                        return {k: plain(v) for k, v in value.items()}
                    if isinstance(value, (tuple, list)):
                        return [plain(v) for v in value]
                    return value

                result = {
                    "program_ok": outcome.ok,
                    "result": plain(outcome.result),
                    "error": str(outcome.error) if outcome.error else None,
                    "simulation_only": True,
                    "coffee_task_success": None,
                    "simulation_seconds": float(adapter.data.time),
                    "run_dir": str(run_dir),
                    "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                }
                serializer.dump(run_dir / "execution.json", outcome)
                serializer.dump(run_dir / "final_observation.json", adapter.get_observation())
                serializer.dump(run_dir / "outcome.json", result)
            elif args.command == "ik":
                from .ik import solve_mount_ik

                result = solve_mount_ik(adapter, args.arm, args.xyz, args.wxyz)
            else:
                import numpy as np

                if not np.isfinite(args.seconds) or not 0 < args.seconds <= 30:
                    raise ValueError("seconds must be in (0, 30]")
                names = ["pot"] + [
                    adapter.model.body(i).name
                    for i in range(adapter.model.nbody)
                    if adapter.model.body(i).name.startswith("cup_")
                    and adapter.model.body(i).name != "cup_dispenser"
                ]
                before = {n: adapter.data.body(n).xpos.copy() for n in names}
                adapter._advance(args.seconds)
                drift = {
                    n: float(np.linalg.norm(adapter.data.body(n).xpos - v))
                    for n, v in before.items()
                }
                warnings = adapter.data.warning.number.tolist()
                result = {
                    "simulation_seconds": args.seconds,
                    "object_drift_m": drift,
                    "warnings": warnings,
                    "finite": bool(np.isfinite(adapter.data.qpos).all()),
                    "initial_retention_stable": max(drift.values()) < 0.01 and not any(warnings),
                    "coffee_task_success": None,
                    "real_world_validated": False,
                }
        finally:
            adapter.close()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
