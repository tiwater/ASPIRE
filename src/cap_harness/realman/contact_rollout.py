"""Simulation-only contact experiments, with traceable checkpoints and contact logs.

Uses scene truth as an explicitly labelled development baseline. No hardware IO,
object welding, gravity changes, or object pose writes during execution.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_geometry import (
    grasp_geometry,
    gripper_linkage_errors,
    mount_pose_for_grasp,
    pot_handle_grasp_pose,
)
from .ik import solve_mount_ik
from .sim import CoffeeSimAdapter
from .workspace import robot_contacts


class ContactRollout:
    def __init__(self, review, output, *, checkpoint=None):
        self.review, self.output = Path(review), Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.plan = json.loads((self.review / "result.json").read_text())
        self.source_bytes = Path(__file__).read_bytes()
        self.scene_path = self.review / "model/scene.xml"
        self.model_hash = hashlib.sha256(self.scene_path.read_bytes()).hexdigest()
        if self.model_hash != self.plan["model_sha256"]:
            raise ValueError("review/model mismatch")
        if checkpoint is not None:
            path = Path(checkpoint)
            meta = json.loads(path.with_suffix(".json").read_text())
            if meta.get("runtime_options") is not None:
                raise ValueError(
                    "runtime options checkpoint requires explicit option restoration; "
                    "do not silently restore it with the XML defaults"
                )
            if meta.get("runtime_payload") is not None:
                raise ValueError(
                    "runtime payload checkpoint requires explicit inertia restoration; "
                    "an XML hash and integration state alone are insufficient"
                )
        self.adapter = CoffeeSimAdapter(self.scene_path, width=640, height=480)
        self.initial_timestep_s = float(self.adapter.model.opt.timestep)
        self.initial_enable_flags = int(self.adapter.model.opt.enableflags)
        self.initial_disable_flags = int(self.adapter.model.opt.disableflags)
        self.config = json.loads((self.review / "scene_config.json").read_text())
        self.adapter.enable_task_evaluation(self.config)
        self.stage = "initial"
        self.trace = []
        self.allowed = set()
        self.required_relations = set()
        self.require_pot_pinch = False
        self.max_linkage_error_rad = dict.fromkeys(self.adapter.arms, 0.0)
        self.last_log_s = -1.0
        self.checkpoint = None
        self.parameters = {}
        if checkpoint is not None:
            if (
                meta["model_sha256"] != self.model_hash
                or meta["npz_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                raise ValueError("checkpoint checksum mismatch")
            with np.load(path, allow_pickle=False) as archive:
                self.adapter.mj.mj_setState(
                    self.adapter.model,
                    self.adapter.data,
                    archive["state"],
                    self.adapter.mj.mjtState.mjSTATE_INTEGRATION,
                )
            self.adapter.mj.mj_forward(self.adapter.model, self.adapter.data)
            self.allowed = {tuple(pair) for pair in meta.get("allowed_contacts", [])}
            # Initial object references remain those of the model's initial state;
            # temporal evidence must be observed again after a checkpoint restore.
            self.checkpoint = str(path)
        self.record(force=True)

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, value):
        self._stage = value
        print(f"{self.adapter.data.time:.2f}s: {value}", flush=True)

    def contacts(self):
        a = self.adapter
        result = []
        for i, c in enumerate(a.data.contact):
            f = np.zeros(6)
            a.mj.mj_contactForce(a.model, a.data, i, f)
            if f[0] > 0.001:
                result.append(
                    {
                        "geom1": a.model.geom(c.geom1).name,
                        "geom2": a.model.geom(c.geom2).name,
                        "normal_force_n": float(f[0]),
                        "distance_m": float(c.dist),
                    }
                )
        return result

    def record(self, *, force=False):
        a = self.adapter
        if not force and a.data.time - self.last_log_s < 0.1 - 1e-9:
            return
        self.last_log_s = float(a.data.time)
        self.trace.append(
            {
                "time_s": float(a.data.time),
                "stage": self.stage,
                "qpos": a.data.qpos.tolist(),
                "object_positions": {
                    n: a.data.body(n).xpos.tolist() for n in ["pot", *a.task_evaluator.cups]
                },
                "grippers": {arm: grasp_geometry(a.model, a.data, arm) for arm in a.arms},
                "predicates": {k: v.value for k, v in a.task_evidence["evidence"].items()},
                "contacts": self.contacts(),
                "gripper_linkage_errors_rad": {
                    arm: gripper_linkage_errors(a.model, a.data, arm) for arm in a.arms
                },
                "gripper_actuator_torque_nm": {
                    arm: float(a.data.actuator_force[a.model.actuator(arm + "_gripper_servo").id])
                    for arm in a.arms
                },
            }
        )

    def tick(self):
        a = self.adapter
        a._advance(0.02)
        self.record()
        for arm in a.arms:
            error = max(
                map(abs, gripper_linkage_errors(a.model, a.data, arm).values()), default=0.0
            )
            self.max_linkage_error_rad[arm] = max(self.max_linkage_error_rad[arm], error)
            if error > 0.005:
                raise RuntimeError(f"{arm} gripper linkage inconsistent: {error:.6f} rad")
        bad = []
        for c in robot_contacts(a.model, a.data, normal_force_threshold_n=0.02):
            names = [c["geom1"], c["geom2"]]
            allowed = False
            for index in (0, 1):
                other_body = a.model.body(a.model.geom(names[1 - index]).bodyid[0]).name
                valid_region = other_body != "pot" or names[1 - index].startswith(
                    "handle_collision_"
                )
                if (
                    (names[index], other_body) in self.allowed
                    and c["penetration_m"] <= 0.002
                    and valid_region
                ):
                    allowed = True
                # Tapered cup walls can meet the distal support edge just above
                # the rectangular pad. Accept only the target object's contact
                # within the small pad neighbourhood, never the whole support.
                for arm in a.arms:
                    for side in ["Left", "Right"]:
                        support = f"{arm[0]}tool_{side}_Support_Link_collision_0"
                        pad = f"{arm}_{side.lower()}_pad_contact"
                        if (
                            names[index] == support
                            and (pad, other_body) in self.allowed
                            and valid_region
                        ):
                            site = a.data.site(f"{arm}_{side.lower()}_pad_surface")
                            local = site.xmat.reshape(3, 3).T @ (
                                np.asarray(c["position_world_m"]) - site.xpos
                            )
                            if (
                                np.all(np.abs(local) <= [0.006, 0.020, 0.016])
                                and c["penetration_m"] <= 0.002
                            ):
                                allowed = True
            if not allowed:
                bad.append(c)
        if bad:
            raise RuntimeError("unexpected robot contact: " + json.dumps(bad[:4]))
        if a.task_evidence["failures"]:
            raise RuntimeError("object/numerical failure: " + str(a.task_evidence["failures"]))
        if self.stage.startswith("pregrasp_"):
            if any(
                np.linalg.norm(a.data.body(n).xpos - p) > 0.01
                for n, p in a.task_evaluator.initial.items()
            ):
                raise RuntimeError("fixture drift before object interaction")
        lost = [
            k for k in self.required_relations if a.task_evidence["evidence"][k].value is not True
        ]
        if lost:
            raise RuntimeError("previously verified relation lost: " + str(lost))
        if self.require_pot_pinch:
            self.check_pot_pinch()

    def check_pot_pinch(self):
        """Early slip guard before displacement is sufficient to grade a grasp."""
        loads = {"left_left_pad_contact": 0.0, "left_right_pad_contact": 0.0}
        for contact in self.contacts():
            for pad, other in [
                (contact["geom1"], contact["geom2"]),
                (contact["geom2"], contact["geom1"]),
            ]:
                if pad in loads and other.startswith("handle_collision_"):
                    loads[pad] += contact["normal_force_n"]
        if min(loads.values()) < 0.02:
            raise RuntimeError("lost bilateral pot-handle pinch: " + json.dumps(loads))

    def hold(self, seconds):
        for _ in range(round(seconds / 0.02)):
            self.tick()

    def joint_path(self, arm, points, speed=0.35):
        a = self.adapter
        points = np.asarray(points)
        a._validate_targets({arm: points})
        if np.max(np.abs(points[0] - a.data.qpos[a.qadr[arm]])) > 0.03:
            raise RuntimeError("stale joint path")
        for first, last in zip(points[:-1], points[1:]):
            n = max(1, int(np.ceil(np.max(np.abs(last - first)) / (speed * 0.02))))
            for i in range(1, n + 1):
                goal = first + (last - first) * (i / n)
                a.data.ctrl[a.actuators[arm]] = goal
                self.tick()
                if np.max(np.abs(a.data.qpos[a.qadr[arm]] - goal)) > 0.15:
                    raise RuntimeError("joint tracking failure")
        self.hold(0.5)

    def prepare(self):
        for route in self.plan["sequential_paths"]:
            self.stage = "pregrasp_" + route["arm"]
            self.joint_path(route["arm"], route["waypoints_urdf_rad"])
        self.save_checkpoint(self.output / "pregrasp.npz")

    def save_checkpoint(self, path):
        a = self.adapter
        size = a.mj.mj_stateSize(a.model, a.mj.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(size)
        a.mj.mj_getState(a.model, a.data, state, a.mj.mjtState.mjSTATE_INTEGRATION)
        np.savez_compressed(path, state=state)
        changed = {}
        if a.model.opt.timestep != self.initial_timestep_s:
            changed["timestep_s"] = float(a.model.opt.timestep)
        if a.model.opt.enableflags != self.initial_enable_flags:
            changed["enable_flags"] = int(a.model.opt.enableflags)
        if a.model.opt.disableflags != self.initial_disable_flags:
            changed["disable_flags"] = int(a.model.opt.disableflags)
        options = {"runtime_options": changed} if changed else {}
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "model_sha256": self.model_hash,
                    "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "time_s": float(a.data.time),
                    "stage": self.stage,
                    "allowed_contacts": sorted(self.allowed),
                    "simulation_only": True,
                    **options,
                },
                indent=2,
            )
            + "\n"
        )

    def command_grasp(self, arm, target):
        a = self.adapter
        mount = mount_pose_for_grasp(target, grasp_geometry(a.model, a.data, arm)["T_mount_grasp"])
        q = Rotation.from_matrix(mount[:3, :3]).as_quat()[[3, 0, 1, 2]]
        r = solve_mount_ik(a, arm, mount[:3, 3], q, restarts=1)
        if not r["pose_solved"]:
            raise RuntimeError("incremental IK failed: " + json.dumps(r))
        joints = np.asarray(r["joint_positions_urdf_rad"])
        if np.max(np.abs(joints - a.data.qpos[a.qadr[arm]])) > 0.12:
            raise RuntimeError("IK branch discontinuity")
        a.data.ctrl[a.actuators[arm]] = joints

    def move_grasp(self, arm, target, *, speed=0.015):
        a = self.adapter
        initial = np.asarray(grasp_geometry(a.model, a.data, arm)["T_world_grasp"])
        distance = np.linalg.norm(target[:3, 3] - initial[:3, 3])
        angular = Rotation.from_matrix(target[:3, :3] @ initial[:3, :3].T).as_rotvec()
        count = max(1, int(np.ceil(max(distance / speed, np.linalg.norm(angular) / 0.1) / 0.02)))
        for i in range(1, count + 1):
            t = initial.copy()
            t[:3, 3] += (target[:3, 3] - initial[:3, 3]) * (i / count)
            t[:3, :3] = Rotation.from_rotvec(angular * (i / count)).as_matrix() @ initial[:3, :3]
            self.command_grasp(arm, t)
            self.tick()
        self.hold(0.4)

    def close_gripper(self, arm, fraction, *, compensate=True):
        a = self.adapter
        target = np.asarray(grasp_geometry(a.model, a.data, arm)["T_world_grasp"])
        start = a.get_robot_state().gripper_positions[arm]
        for i in range(1, 101):
            a._grippers({arm: start + (fraction - start) * (i / 100)})
            if compensate:
                self.command_grasp(arm, target)
            self.tick()
        self.hold(0.6)

    def finish(self, error=None):
        from PIL import Image

        self.record(force=True)
        self.parameters["max_gripper_linkage_error_rad"] = self.max_linkage_error_rad
        a = self.adapter
        result = {
            "simulation_only": True,
            "physical_motion_allowed": False,
            "oracle_policy": True,
            "coffee_task_success": None,
            "last_stage": self.stage,
            "error": error,
            "experiment_success": error is None,
            "parameters": self.parameters,
            "model_sha256": self.model_hash,
            "checkpoint": self.checkpoint,
            "simulation_seconds": float(a.data.time),
            "predicates": {k: v.value for k, v in a.task_evidence["evidence"].items()},
        }
        (self.output / "trace.json").write_text(json.dumps(self.trace, indent=2) + "\n")
        (self.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (self.output / "experiment.py").write_bytes(self.source_bytes)
        Image.fromarray(a.camera("overview").rgb).save(self.output / "final.png")
        self.save_checkpoint(self.output / "final_state.npz")
        a.close()
        return result


def cup_withdraw_clearance(data, config):
    """Conservative vertical gap between the target lip and every remaining base.

    Oracle geometry for simulation supervision: commanded wrist travel cannot
    establish release when the gripper slipped along a tapered cup.
    """
    cup = config["cup"]
    target = data.body("cup_0")
    up = float(target.xmat[8])
    target_top = target.xpos[2] + cup["height_m"] * up
    target_top += cup["rim_radius_m"] * np.sqrt(max(0.0, 1 - up * up))
    bottoms = []
    for i in range(1, config["dispenser"]["stack_count"]):
        other = data.body(f"cup_{i}")
        up = float(other.xmat[8])
        bottoms.append(other.xpos[2] - cup["bottom_radius_m"] * np.sqrt(max(0.0, 1 - up * up)))
    return min(bottoms) - target_top if bottoms else float("inf")


def run_contact_experiment(
    review,
    output,
    *,
    checkpoint=None,
    cup_fraction=0.20,
    cup_height=0.035,
    hold_seconds=5.0,
):
    if not np.isfinite(hold_seconds) or not 0.3 <= hold_seconds <= 300:
        raise ValueError("hold_seconds must be in [0.3, 300]")
    exp = ContactRollout(review, output, checkpoint=checkpoint)
    exp.parameters = {
        "cup_fraction": cup_fraction,
        "cup_height_m": cup_height,
        "right_camera_up_from_pregrasp": True,
        "required_hold_seconds": hold_seconds,
    }
    try:
        if exp.plan["targets"]["right"].get("wrist_camera_side") != "above":
            raise ValueError("rebuild workspace review with camera-up right pregrasp")
        if checkpoint is None:
            exp.prepare()
        a = exp.adapter
        frame = np.asarray(grasp_geometry(a.model, a.data, "right")["T_world_grasp"])
        camera_height = float(a.data.body("r_camera_link").xpos[2] - frame[2, 3])
        exp.parameters["pregrasp_camera_above_grasp_m"] = camera_height
        if camera_height < 0.02:
            raise RuntimeError("right camera is not above the grasp before approach")
        exp.stage = "cup_approach"
        target = np.asarray(grasp_geometry(a.model, a.data, "right")["T_world_grasp"])
        target[:3, 3] = a.data.body("cup_0").xpos + [0, 0, cup_height]
        exp.move_grasp("right", target)
        exp.stage = "cup_close"
        exp.allowed = {(f"right_{jaw}_pad_contact", "cup_0") for jaw in ["left", "right"]}
        exp.close_gripper("right", cup_fraction)
        exp.stage = "cup_extract_down"
        target = np.asarray(grasp_geometry(a.model, a.data, "right")["T_world_grasp"])
        target[:3, 3] += [0, 0, -0.085]
        exp.move_grasp("right", target, speed=0.01)
        gap = cup_withdraw_clearance(a.data, exp.config)
        exp.parameters["prewithdraw_cup_clearance_m"] = float(gap)
        if not a.task_evidence["evidence"]["cup_upright"].value or gap < 0.001:
            raise RuntimeError("cup not clear for horizontal withdrawal; possible gripper slip")
        exp.stage = "cup_withdraw"
        target[:3, 3] += [-0.12, 0, 0]
        exp.move_grasp("right", target)
        exp.hold(0.5)
        exp.stage = "verify_single_cup"
        predicates = a.task_evidence["evidence"]
        if not all(
            predicates[k].value
            for k in ["cup_in_right_gripper", "single_cup_separated", "remaining_cups_retained"]
        ):
            raise RuntimeError("single-cup predicates not met")
        exp.required_relations = {
            "cup_in_right_gripper",
            "single_cup_separated",
            "remaining_cups_retained",
        }
        exp.stage = "sustained_single_cup_hold"
        exp.hold(hold_seconds)
        exp.parameters["completed_hold_seconds"] = hold_seconds
        exp.save_checkpoint(exp.output / "cup_held.npz")
        return exp.finish()
    except (ValueError, RuntimeError) as exc:
        return exp.finish(str(exc))


def run_pot_experiment(review, output, *, checkpoint):
    from .pot_bench import diagonal_withdraw_delta

    exp = ContactRollout(review, output, checkpoint=checkpoint)
    try:
        delta = diagonal_withdraw_delta(exp.config)
        exp.parameters.update(
            withdraw_delta_table_m=delta.tolist(),
            withdraw_speed_m_s=0.01,
            return_path="reverse_of_horizontal_left_back_diagonal",
            pot_pregrasp_open_fraction=0.25,
            gripper_servos=exp.config.get("gripper_servos", {}),
        )
        a = exp.adapter
        exp.allowed = {(f"right_{jaw}_pad_contact", "cup_0") for jaw in ["left", "right"]}
        exp.stage = "verify_cup_resume"
        exp.hold(0.6)
        cup_required = ["cup_in_right_gripper", "single_cup_separated", "remaining_cups_retained"]
        if not all(a.task_evidence["evidence"][k].value for k in cup_required):
            raise RuntimeError("resumed single-cup hold not established")
        exp.required_relations = set(cup_required)
        exp.stage = "pot_preshape"
        # A wide inner jaw fouls the loop's root rail. Leave clearance around
        # the outer grip rail while keeping that jaw inside the handle aperture.
        exp.close_gripper("left", 0.25)
        exp.stage = "pot_align_pregrasp"
        target = pot_handle_grasp_pose(a.data)
        exp.parameters["oracle_handle_grasp_world"] = target.tolist()
        pregrasp = target.copy()
        pregrasp[:3, 3] -= target[:3, 2] * 0.08
        exp.move_grasp("left", pregrasp, speed=0.01)
        exp.stage = "pot_approach"
        exp.move_grasp("left", target, speed=0.01)
        exp.stage = "pot_close"
        exp.allowed |= {(f"left_{jaw}_pad_contact", "pot") for jaw in ["left", "right"]}
        exp.close_gripper("left", 0.0)
        exp.require_pot_pinch = True
        exp.check_pot_pinch()
        exp.save_checkpoint(exp.output / "pot_pinch_unqualified.npz")
        seated = np.asarray(grasp_geometry(a.model, a.data, "left")["T_world_grasp"])
        exp.stage = "pot_withdraw"
        target = seated.copy()
        target[:3, 3] += delta
        exp.move_grasp("left", target, speed=0.01)
        exp.hold(0.5)
        required = [
            "pot_handle_in_left_gripper",
            "pot_clear_of_machine",
            "pot_clear_of_outlet",
            "pot_upright",
            "cup_in_right_gripper",
        ]
        if not all(a.task_evidence["evidence"][k].value for k in required):
            raise RuntimeError("pot withdrawal predicates not met")
        exp.save_checkpoint(exp.output / "both_held.npz")
        exp.stage = "pot_return"
        exp.move_grasp("left", seated, speed=0.01)
        exp.stage = "pot_release"
        exp.require_pot_pinch = False
        exp.close_gripper("left", 0.3)
        exp.hold(0.6)
        required = [
            "pot_stable_after_release",
            "pot_on_hotplate",
            "lid_outlet_aligned",
            "pot_upright",
            "left_gripper_released",
            "cup_in_right_gripper",
        ]
        if not all(a.task_evidence["evidence"][k].value for k in required):
            raise RuntimeError("pot return/release predicates not met")
        return exp.finish()
    except (ValueError, RuntimeError) as exc:
        return exp.finish(str(exc))


def render_contact_trace(review, run, output, *, view="overview"):
    """Render recorded qpos for review; never rerun physics or score these frames."""
    import imageio.v2 as imageio
    import mujoco

    review, run, output = Path(review), Path(run), Path(output)
    result = json.loads((run / "result.json").read_text())
    model_path = review / "model/scene.xml"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("trace/model mismatch")
    if output.exists():
        raise FileExistsError(output)
    frames = json.loads((run / "trace.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.78, -0.08, 0.85]
    camera.distance = 1.25
    camera.azimuth = -135
    camera.elevation = -22
    if view == "cup":
        # Side view avoids the coffee-machine housing occluding the dispenser.
        camera.lookat[:] = data.body("cup_dispenser").xpos + [-0.035, 0, 0.13]
        camera.distance = 0.60
        camera.azimuth = 90
        camera.elevation = -15
    elif view != "overview":
        raise ValueError("unknown contact review view")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        mujoco.Renderer(model, height=480, width=640) as renderer,
        imageio.get_writer(output, fps=10, codec="libx264", quality=7) as writer,
    ):
        for frame in frames:
            qpos = np.asarray(frame["qpos"])
            if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
                raise ValueError("invalid recorded state")
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    return {
        "video": str(output),
        "frames": len(frames),
        "fps": 10,
        "view": view,
        "purpose": "visualization of recorded simulation states only",
        "experiment_success": result.get("experiment_success"),
        "coffee_task_success": None,
    }
