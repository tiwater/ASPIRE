"""Replay a reviewed pregrasp path through simulated servos, monitoring contacts."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .grasp_geometry import grasp_geometry
from .sim import CoffeeSimAdapter
from .workspace import robot_contacts


def rollout_pregrasp(review_dir, output, *, speed_rad_s=0.35):
    """Use a fresh simulation. No object teleport, attachment, or robot connection."""
    from PIL import Image

    review_dir, output = Path(review_dir), Path(output)
    report = json.loads((review_dir / "result.json").read_text())
    model_path = review_dir / "model/scene.xml"
    if (
        not report.get("pregrasp_transit_found")
        or hashlib.sha256(model_path.read_bytes()).hexdigest() != report["model_sha256"]
    ):
        raise ValueError("need a complete pregrasp plan for this exact model")
    if not np.isfinite(speed_rad_s) or not 0 < speed_rad_s <= 0.5:
        raise ValueError("invalid simulation speed")
    output.mkdir(parents=True, exist_ok=False)
    adapter = CoffeeSimAdapter(model_path, width=640, height=480)
    cfg = json.loads((review_dir / "scene_config.json").read_text())
    adapter.enable_task_evaluation(cfg)
    objects = ["pot", *[f"cup_{i}" for i in range(cfg["dispenser"]["stack_count"])]]
    result = {
        "simulation_only": True,
        "physical_motion_allowed": False,
        "coffee_task_success": None,
        "pregrasp_rollout_ok": False,
        "review_sha256": hashlib.sha256((review_dir / "result.json").read_bytes()).hexdigest(),
        "rollout_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_sha256": report["model_sha256"],
        "speed_rad_s": speed_rad_s,
        "collision_monitor_period_s": 0.02,
        "continuous_collision_checked": False,
        "max_joint_tracking_error_rad": 0.0,
        "paths": [],
    }
    baseline = {n: adapter.data.body(n).xpos.copy() for n in objects}
    try:
        for route in report["sequential_paths"]:
            arm = route["arm"]
            points = np.asarray(route["waypoints_urdf_rad"], dtype=float)
            adapter._validate_targets({arm: points})
            if np.max(np.abs(adapter.data.qpos[adapter.qadr[arm]] - points[0])) > 0.03:
                raise ValueError("stale planned start")
            for first, last in zip(points[:-1], points[1:]):
                steps = max(1, int(np.ceil(np.max(np.abs(last - first)) / (speed_rad_s * 0.02))))
                for i in range(1, steps + 1):
                    target = first + (last - first) * (i / steps)
                    adapter.data.ctrl[adapter.actuators[arm]] = target
                    adapter._advance(0.02)
                    error = float(np.max(np.abs(adapter.data.qpos[adapter.qadr[arm]] - target)))
                    result["max_joint_tracking_error_rad"] = max(
                        result["max_joint_tracking_error_rad"], error
                    )
                    contacts = robot_contacts(adapter.model, adapter.data)
                    if contacts:
                        result["first_robot_collision"] = contacts
                        raise RuntimeError("robot collision during simulated execution")
                    if error > 0.15 or adapter.task_evidence["failures"]:
                        raise RuntimeError(
                            "tracking or object-state failure during simulated execution"
                        )
            # Observe hold convergence with the same monitoring period.
            for _ in range(25):
                adapter._advance(0.02)
                if robot_contacts(adapter.model, adapter.data):
                    raise RuntimeError("robot collision during endpoint hold")
            goal = np.asarray(report["targets"][arm]["T_world_grasp"])
            actual = np.asarray(grasp_geometry(adapter.model, adapter.data, arm)["T_world_grasp"])
            from scipy.spatial.transform import Rotation

            error = float(np.linalg.norm(goal[:3, 3] - actual[:3, 3]))
            angle = float(
                np.linalg.norm(Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).as_rotvec())
            )
            result["paths"].append(
                {"arm": arm, "position_error_m": error, "orientation_error_rad": angle}
            )
            if error > 0.005 or angle > 0.03:
                raise RuntimeError("pregrasp endpoint tracking outside tolerance")
        result["pregrasp_rollout_ok"] = True
    except (RuntimeError, ValueError) as exc:
        result["error"] = str(exc)
    finally:
        result["simulation_seconds"] = float(adapter.data.time)
        result["object_drift_m"] = {
            n: float(np.linalg.norm(adapter.data.body(n).xpos - p)) for n, p in baseline.items()
        }
        if max(result["object_drift_m"].values()) > 0.01:
            result["pregrasp_rollout_ok"] = False
            result.setdefault("error", "fixture object drift during pregrasp transit")
        result["object_predicates"] = {
            k: v.value for k, v in adapter.task_evidence["evidence"].items()
        }
        Image.fromarray(adapter.camera("overview").rgb).save(output / "final.png")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        adapter.close()
    return result
