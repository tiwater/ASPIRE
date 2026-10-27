"""CAP EnvironmentAdapter for the RealBot coffee prototype. Simulation IO only."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cap_harness.contracts import (
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotState,
    StepResult,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode


class CoffeeSimAdapter:
    embodiment = "realman"
    control_period_s = 0.02
    control_frequency = 50
    arms = ("left", "right")
    dof = 7

    def __init__(self, scene: str | Path, *, width=640, height=480):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.width, self.height = width, height
        self.renderer = None
        self.joint_names = {a: tuple(f"{a[0]}_joint_{i}" for i in range(1, 8)) for a in self.arms}
        self.joint_ids = {
            a: np.array([self.model.joint(n).id for n in names])
            for a, names in self.joint_names.items()
        }
        self.qadr = {a: self.model.jnt_qposadr[ids] for a, ids in self.joint_ids.items()}
        self.vadr = {a: self.model.jnt_dofadr[ids] for a, ids in self.joint_ids.items()}
        self.actuators = {
            a: np.array([self.model.actuator(n + "_servo").id for n in names])
            for a, names in self.joint_names.items()
        }
        self._task = TaskContext(
            "realman_coffee_prototype",
            0,
            "serve_coffee",
            "取一只杯，取壶，倒至约八成，放杯，放回壶。",
            "coffee",
            {
                "qualification": "geometric/contact prototype",
                "fluid_simulation": False,
                "tool_frame": "URDF link_8 mounting frame; not calibrated grasp center",
            },
        )
        self.mj.mj_forward(self.model, self.data)

    def resolve_arm(self, arm):
        return {"primary": "right", "secondary": "left"}.get(arm, arm)

    def get_task_context(self):
        return self._task

    def reset(self, task_ref="serve_coffee", seed=0):
        if task_ref not in (0, "serve_coffee", "realman_coffee_prototype"):
            raise ValueError("unknown coffee task")
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_forward(self.model, self.data)
        return self.get_observation()

    def _pose(self, position, matrix):
        q = Rotation.from_matrix(np.asarray(matrix).reshape(3, 3)).as_quat()
        return Pose(position, q[[3, 0, 1, 2]], "world")

    def get_robot_state(self):
        return RobotState(
            {a: self.data.qpos[self.qadr[a]] for a in self.arms},
            {a: self.data.qvel[self.vadr[a]] for a in self.arms},
            {
                a: self._pose(
                    self.data.site(a + "_tool_mount").xpos, self.data.site(a + "_tool_mount").xmat
                )
                for a in self.arms
            },
            {
                a: float(
                    np.clip(
                        (self.data.joint(a[0] + "tool_Left_1_Joint").qpos[0] + 0.91) / 0.91, 0, 1
                    )
                )
                for a in self.arms
            },
            "world",
            self.joint_names,
            float(self.data.time),
            self.embodiment,
        )

    def camera(self, name="front"):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=self.height, width=self.width)
        r = self.renderer
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=name)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=name)
        depth = r.render().copy()
        r.disable_depth_rendering()
        # MuJoCo camera convention (+X right, +Y up, -Z forward) -> optical frame.
        cid = self.model.camera(name).id
        rotation = self.data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        f = 0.5 * self.height / math.tan(math.radians(self.model.cam_fovy[cid]) / 2)
        intrinsics = np.array(
            [[f, 0, (self.width - 1) / 2], [0, f, (self.height - 1) / 2], [0, 0, 1]]
        )
        return CameraObservation(
            rgb,
            depth,
            intrinsics,
            name + "_optical",
            self._pose(self.data.cam_xpos[cid], rotation),
            float(self.data.time),
        )

    def get_observation(self):
        return Observation(
            {"front": self.camera()}, self.get_robot_state(), self._task, float(self.data.time)
        )

    def contacts(self):
        """Diagnostics only; contact does not establish a stable grasp."""
        return [
            {
                "geom1": self.model.geom(c.geom1).name,
                "geom2": self.model.geom(c.geom2).name,
                "distance_m": float(c.dist),
            }
            for c in self.data.contact[: self.data.ncon]
        ]

    def _validate_targets(self, targets):
        for arm, value in targets.items():
            if arm not in self.arms:
                raise ValueError(f"unknown arm {arm}")
            q = np.asarray(value)
            limits = self.model.jnt_range[self.joint_ids[arm]]
            if (
                q.shape[-1:] != (7,)
                or not np.isfinite(q).all()
                or np.any(q < limits[:, 0])
                or np.any(q > limits[:, 1])
            ):
                raise ValueError(f"{arm}: invalid or out-of-range joint target")

    def _grippers(self, targets):
        # Validate the whole batch before mutation.
        for arm, fraction in targets.items():
            if arm not in self.arms or not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("invalid gripper target")
        for arm, fraction in targets.items():
            self.data.ctrl[self.model.actuator(arm + "_gripper_servo").id] = -0.91 * (1 - fraction)

    def _advance(self, duration):
        n = round(duration / self.model.opt.timestep)
        if n < 1 or not np.isclose(n * self.model.opt.timestep, duration):
            raise ValueError("duration must be a positive multiple of the simulation timestep")
        for _ in range(n):
            self.mj.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError("non-finite simulation state")
        self.mj.mj_forward(self.model, self.data)

    def step(self, action):
        try:
            if action.embodiment != self.embodiment:
                raise ValueError("wrong embodiment")
            targets = {a: c.target for a, c in action.arms.items()}
            self._validate_targets(targets)
            self._grippers(
                {
                    a: c.gripper_position
                    for a, c in action.arms.items()
                    if c.gripper_position is not None
                }
            )
            for a, q in targets.items():
                self.data.ctrl[self.actuators[a]] = q
            self._advance(self.control_period_s)
            return StepResult(
                True,
                self.get_observation(),
                diagnostics={"task_success": None, "simulation_only": True},
            )
        except (ValueError, RuntimeError) as exc:
            return StepResult(False, error=ApiError(ErrorCode.EXECUTION_FAILED, str(exc)))

    def execute_trajectory(self, trajectory):
        steps = 0
        try:
            if trajectory.embodiment != self.embodiment:
                raise ValueError("wrong trajectory embodiment")
            single = isinstance(trajectory, Trajectory)
            targets = (
                {trajectory.arm: trajectory.joint_positions}
                if single
                else trajectory.joint_positions
            )
            self._validate_targets(targets)
            dt = trajectory.dt_s
            count = len(next(iter(targets.values())))
            if (
                count * dt > 60
                or not np.isclose(round(dt / self.model.opt.timestep) * self.model.opt.timestep, dt)
                or dt < self.model.opt.timestep
            ):
                raise ValueError("invalid trajectory duration/timestep")
            current = self.get_robot_state()
            for a in self.arms:
                if (
                    a not in trajectory.expected_start.joint_positions
                    or np.max(
                        np.abs(
                            current.joint_positions[a]
                            - trajectory.expected_start.joint_positions[a]
                        )
                    )
                    > 0.03
                ):
                    return ExecutionResult(
                        False, 0, error=ApiError(ErrorCode.STALE_PLAN, "robot start changed")
                    )
            for i in range(count):
                for a, q in targets.items():
                    self.data.ctrl[self.actuators[a]] = q[i]
                if trajectory.gripper_positions is not None:
                    g = (
                        {trajectory.arm: trajectory.gripper_positions[i]}
                        if single
                        else {a: v[i] for a, v in trajectory.gripper_positions.items()}
                    )
                    self._grippers(g)
                self._advance(dt)
                steps += 1
            self._advance(0.5)
            errors = {
                a: float(np.max(np.abs(self.data.qpos[self.qadr[a]] - q[-1])))
                for a, q in targets.items()
            }
            ok = max(errors.values()) <= 0.05
            return ExecutionResult(
                ok,
                steps,
                final_observation=self.get_observation(),
                final_errors=errors,
                error=None
                if ok
                else ApiError(ErrorCode.EXECUTION_FAILED, "simulated joint following error"),
                diagnostics={
                    "simulation_only": True,
                    "task_success": None,
                    "fluid_simulation": False,
                },
            )
        except (ValueError, RuntimeError) as exc:
            return ExecutionResult(
                False, steps, error=ApiError(ErrorCode.EXECUTION_FAILED, str(exc))
            )

    def set_gripper(self, position, *, arm="primary"):
        return self.set_grippers({self.resolve_arm(arm): position})

    def set_grippers(self, positions):
        try:
            self._grippers(positions)
            self._advance(0.5)
            state = self.get_robot_state()
            errors = {a: abs(state.gripper_positions[a] - p) for a, p in positions.items()}
            ok = max(errors.values(), default=0) < 0.1
            return ExecutionResult(
                ok,
                25,
                final_observation=self.get_observation(),
                final_errors=errors,
                error=None
                if ok
                else ApiError(
                    ErrorCode.EXECUTION_FAILED,
                    "gripper obstructed or not settled; grasp unverified",
                ),
                diagnostics={"grasp_verified": False, "simulation_only": True},
            )
        except (ValueError, RuntimeError) as exc:
            return ExecutionResult(False, 0, error=ApiError(ErrorCode.EXECUTION_FAILED, str(exc)))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
