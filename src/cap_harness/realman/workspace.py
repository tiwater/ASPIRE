"""Offline, oracle pregrasp reachability and sampled joint-segment diagnostics."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_geometry import mount_pose_for_grasp
from .ik import solve_mount_ik


def robot_contacts(model, data, *, penetration_m=0.0001, normal_force_threshold_n=None):
    """Include both arms AND frozen torso/base; never filter a task-object collision."""
    root = model.body("base_link").id
    bodies = {root}
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    contacts = []
    for index, c in enumerate(data.contact[: data.ncon]):
        if not (
            int(model.geom_bodyid[c.geom1]) in bodies or int(model.geom_bodyid[c.geom2]) in bodies
        ):
            continue
        normal_force = None
        if normal_force_threshold_n is not None:
            import mujoco

            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            normal_force = float(wrench[0])
        if c.dist >= -penetration_m and (
            normal_force is None or normal_force <= normal_force_threshold_n
        ):
            continue
        contacts.append(
            {
                "geom1": model.geom(c.geom1).name,
                "geom2": model.geom(c.geom2).name,
                "penetration_m": -float(c.dist),
                "position_world_m": c.pos.tolist(),
                "normal_force_n": normal_force,
            }
        )
    return contacts


def check_joint_segment(adapter, targets, *, max_step_rad=0.025):
    """Sample a synchronized straight segment without altering the live simulation.

    A clear result is discrete geometric evidence, not continuous collision proof,
    a dynamics rollout, or a physical-robot command.
    """
    if not np.isfinite(max_step_rad) or not 0 < max_step_rad <= 0.1 or not targets:
        raise ValueError("invalid sampling step or empty targets")
    m, mj = adapter.model, adapter.mj
    d = mj.MjData(m)
    d.qpos[:] = adapter.data.qpos
    starts = {}
    for arm, value in targets.items():
        if arm not in adapter.arms:
            raise ValueError("unknown arm")
        target = np.asarray(value, dtype=float)
        limits = m.jnt_range[adapter.joint_ids[arm]]
        if (
            target.shape != (7,)
            or not np.isfinite(target).all()
            or np.any(target < limits[:, 0])
            or np.any(target > limits[:, 1])
        ):
            raise ValueError("target outside joint limits")
        starts[arm] = d.qpos[adapter.qadr[arm]].copy()
    count = max(
        1,
        int(
            np.ceil(
                max(np.max(np.abs(np.asarray(targets[a]) - q)) for a, q in starts.items())
                / max_step_rad
            )
        ),
    )
    for index in range(count + 1):
        alpha = index / count
        for arm, q in starts.items():
            d.qpos[adapter.qadr[arm]] = q + alpha * (np.asarray(targets[arm]) - q)
        mj.mj_forward(m, d)
        contacts = robot_contacts(m, d)
        if contacts:
            return {
                "sampled_collision_free": False,
                "first_collision_fraction": alpha,
                "contacts": contacts,
                "samples_checked": index + 1,
                "max_joint_step_rad": max_step_rad,
                "continuous_collision_checked": False,
                "dynamics_checked": False,
                "physical_motion_allowed": False,
            }
    return {
        "sampled_collision_free": True,
        "samples_checked": count + 1,
        "max_joint_step_rad": max_step_rad,
        "continuous_collision_checked": False,
        "dynamics_checked": False,
        "physical_motion_allowed": False,
    }


def pregrasp_targets(adapter, *, offset_m=0.08):
    """Diagnostic front approach, open pads; scene truth is explicitly oracle input."""
    if not np.isfinite(offset_m) or not 0.03 <= offset_m <= 0.2:
        raise ValueError("invalid pregrasp offset")
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = {}
    for arm, point in (
        ("left", adapter.data.site("pot_handle_grasp").xpos),
        ("right", adapter.data.body("cup_0").xpos + np.array([0.0, 0.0, 0.035])),
    ):
        target = np.eye(4)
        # Plan the right wrist's final roll from the initial posture. The local
        # approach axis (+Z) is unchanged, but the camera is above the jaws.
        target[:3, :3] = rotation @ np.diag([-1.0, -1.0, 1.0]) if arm == "right" else rotation
        target[:3, 3] = point - offset_m * rotation[:, 2]
        mount = mount_pose_for_grasp(target, adapter.get_grasp_geometry(arm)["T_mount_grasp"])
        quaternion = Rotation.from_matrix(mount[:3, :3]).as_quat()[[3, 0, 1, 2]]
        result[arm] = {
            "T_world_grasp": target.tolist(),
            "mount_xyz": mount[:3, 3].tolist(),
            "mount_wxyz": quaternion.tolist(),
            "oracle_geometry": True,
            "approach_offset_m": offset_m,
            "grasp_verified": False,
            # The left camera mount is mirrored in the URDF; its unchanged
            # grasp orientation already places that camera above the jaws.
            "wrist_camera_side": "above",
        }
    return result


def check_workspaces(config_path, output):
    """Keep every candidate including failures; no real robot IO or success grading."""
    from PIL import Image

    from .model import build_model
    from .sim import CoffeeSimAdapter

    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    robot_path, scene_path = Path(config["robot_config"]), Path(config["scene_config"])
    robot, scene = json.loads(robot_path.read_text()), json.loads(scene_path.read_text())
    if robot.get("physical_motion_backend") is not None:
        raise ValueError("requires simulation-only configuration")
    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "workspace_config": config_path,
        "robot_config": robot_path,
        "scene_config": scene_path,
        "workspace_code": Path(__file__),
        "ik_code": Path(__file__).with_name("ik.py"),
        "model_code": Path(__file__).with_name("model.py"),
        "grasp_code": Path(__file__).with_name("grasp_geometry.py"),
        "planner_code": Path(__file__).with_name("joint_planning.py"),
    }
    hashes = {}
    for name, source in sources.items():
        content = source.read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        (output / (name + source.suffix)).write_bytes(content)
    summary = {
        "simulation_only": True,
        "oracle_geometry": True,
        "coffee_task_success": None,
        "source_sha256": hashes,
        "candidates": [],
    }
    for index, candidate in enumerate(config["candidates"]):
        folder = output / f"candidate_{index}"
        folder.mkdir()
        variant = deepcopy(scene)
        variant["robot"]["frozen_joints_rad"].update(candidate["frozen_joints_rad"])
        variant["robot"]["base_xy_m"] = candidate.get("base_xy_m", variant["robot"]["base_xy_m"])
        variant_path = folder / "scene_config.json"
        variant_path.write_text(json.dumps(variant, indent=2) + "\n")
        build = build_model(Path(robot["urdf"]), variant_path, folder / "model")
        adapter = CoffeeSimAdapter(build["scene"], width=640, height=480)
        try:
            targets = pregrasp_targets(adapter, offset_m=config["pregrasp_offset_m"])
            result = {
                "name": candidate["name"],
                "configuration": candidate,
                "targets": targets,
                "initial_robot_contacts": robot_contacts(adapter.model, adapter.data),
                "arms": {},
                "both_endpoints_clear": False,
            }
            for arm, target in targets.items():
                result["arms"][arm] = solve_mount_ik(
                    adapter,
                    arm,
                    target["mount_xyz"],
                    target["mount_wxyz"],
                    seed=config["ik_seed"],
                    restarts=config["ik_restarts"],
                )
            original = adapter.data.qpos.copy()
            for arm, report in result["arms"].items():
                adapter.data.qpos[adapter.qadr[arm]] = report["joint_positions_urdf_rad"]
            adapter.mj.mj_forward(adapter.model, adapter.data)
            result["combined_robot_contacts"] = robot_contacts(adapter.model, adapter.data)
            result["both_endpoints_clear"] = (
                all(r["pose_solved"] for r in result["arms"].values())
                and not result["combined_robot_contacts"]
            )
            Image.fromarray(adapter.camera("overview").rgb).save(folder / "candidate_endpoints.png")
            adapter.data.qpos[:] = original
            adapter.mj.mj_forward(adapter.model, adapter.data)
            result["joint_segment"] = (
                check_joint_segment(
                    adapter,
                    {arm: r["joint_positions_urdf_rad"] for arm, r in result["arms"].items()},
                )
                if result["both_endpoints_clear"]
                else None
            )
            result["sequential_paths"] = []
            if result["both_endpoints_clear"]:
                from .joint_planning import plan_joint_path

                # Task order: right cup first, then left pot. Objects are not
                # attached or moved: this checks pregrasp transit only.
                for arm in ["right", "left"]:
                    route = plan_joint_path(
                        adapter,
                        arm,
                        result["arms"][arm]["joint_positions_urdf_rad"],
                        seed=config["ik_seed"],
                        iterations=config.get("path_iterations", 500),
                    )
                    result["sequential_paths"].append(route)
                    if not route["found"]:
                        break
                    adapter.data.qpos[adapter.qadr[arm]] = route["waypoints_urdf_rad"][-1]
                    adapter.mj.mj_forward(adapter.model, adapter.data)
                adapter.data.qpos[:] = original
                adapter.mj.mj_forward(adapter.model, adapter.data)
            result["pregrasp_transit_found"] = len(result["sequential_paths"]) == 2 and all(
                p["found"] for p in result["sequential_paths"]
            )
            result["model_sha256"] = hashlib.sha256(Path(build["scene"]).read_bytes()).hexdigest()
            (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            summary["candidates"].append(result)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            adapter.close()
    return summary
