"""Continuous simulation preparation/exit around a coffee service session."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .joint_planning import plan_joint_path
from .lifecycle import BODY_JOINTS, LifecycleMotion, body_tilt_deg
from .workspace import robot_contacts


class ServiceLifecycle:
    def __init__(self, rollout, plan_path):
        self.rollout = rollout
        self.plan_path = Path(plan_path)
        self.plan = json.loads(self.plan_path.read_text())
        if (
            not self.plan["sampled_prepare_passed"]
            or self.plan["model_sha256"] != rollout.model_hash
        ):
            raise ValueError("lifecycle plan/model mismatch")
        self.points = self.plan["prepare_waypoints"]
        self.motion = LifecycleMotion(rollout)
        self.maximum_tilt_deg = 0.0
        self.collision_samples = 0

    def verify_physics_step(self):
        """Call after each integration step, including during coffee actions."""
        e = self.rollout
        m, d = e.adapter.model, e.adapter.data
        tilt = body_tilt_deg(m, d)
        self.maximum_tilt_deg = max(self.maximum_tilt_deg, tilt)
        if tilt > 10:
            raise RuntimeError("body tilt exceeds 10 degrees")
        if e.stage.startswith(("standing", "prepare_", "exit_")):
            self.collision_samples += 1
            bad = robot_contacts(m, d, normal_force_threshold_n=0.02)
            if bad:
                raise RuntimeError("lifecycle robot contact: " + json.dumps(bad[:4]))

    def pose_evidence(self, *, standing=False):
        a = self.rollout.adapter
        m, d = a.model, a.data
        target = np.asarray(self.points[0 if standing else -1]["qpos"])
        body = np.array([m.jnt_qposadr[m.joint(n).id] for n in BODY_JOINTS])
        body_ok = bool(np.max(np.abs(d.qpos[body] - target[body])) < 0.002)
        arms_ok = all(
            np.max(np.abs(d.qpos[a.qadr[side]] - target[a.qadr[side]])) < 0.002 for side in a.arms
        )
        return (
            {"standing_body_pose": body_ok, "arms_hanging": arms_ok}
            if standing
            else {"working_body_pose": body_ok, "arms_pregrasp": arms_ok}
        )

    def prepare(self):
        e = self.rollout
        if not all(self.pose_evidence(standing=True).values()):
            raise ValueError("preparation requires declared standing initial pose")
        e.parameters.update(
            lifecycle_included=True,
            startup_pose_measured=False,
            physical_motion_allowed=False,
            lifecycle_plan_sha256=hashlib.sha256(self.plan_path.read_bytes()).hexdigest(),
        )
        e.stage = "standing_initial_hold"
        e.hold(8)
        for point in self.points[1:]:
            self.motion.move(point["qpos"], point["stage"])
        if not all(self.pose_evidence().values()):
            raise RuntimeError("preparation final pose not verified")
        e.save_checkpoint(e.output / "prepared_work_pose.npz")
        e.parameters["preparation_completed"] = True

    def exit(self, *, transport_timestep=0.001, timestep_ramp_hold_s=0.1):
        e = self.rollout
        a = e.adapter
        m, d = a.model, a.data
        if not np.isfinite([transport_timestep, timestep_ramp_hold_s]).all():
            raise ValueError("exit timestep settings must be finite")
        if transport_timestep <= 0 or timestep_ramp_hold_s < 0:
            raise ValueError("exit timestep must be positive and ramp hold nonnegative")
        required = {
            "pot_stable_after_release",
            "pot_on_hotplate",
            "lid_outlet_aligned",
            "pot_upright",
            "both_grippers_empty",
            "previous_served_cups_preserved",
            "remaining_cups_retained",
        }
        current = a.task_evaluator.update()
        if any(current["evidence"][name].value is not True for name in required):
            raise ValueError("exit requires supported released pot and preserved cups")
        e.required_relations.update(required)
        e.allowed.clear()
        old_timestep = float(m.opt.timestep)
        if timestep_ramp_hold_s and transport_timestep > old_timestep:
            # Deformable fixture contacts can receive a numerical impulse when
            # the integration step jumps directly from contact precision to a
            # coarse transport step. Ramp by at most 2x per transition and
            # recheck the same released-pot and retained-cup predicates at each
            # level before starting the exit motion.
            e.stage = "exit_timestep_ramp"
            timestep = old_timestep
            while timestep < transport_timestep:
                next_timestep = min(transport_timestep, timestep * 2.0)
                e.parameters.setdefault("precision_transitions", []).append(
                    {
                        "time_s": float(d.time),
                        "stage": e.stage,
                        "old_timestep_s": float(timestep),
                        "new_timestep_s": float(next_timestep),
                        "reason": "bounded timestep ramp with preserved fixture predicates",
                    }
                )
                m.opt.timestep = next_timestep
                e.hold(timestep_ramp_hold_s)
                current = a.task_evaluator.update()
                missing = [
                    name for name in required
                    if current["evidence"][name].value is not True
                ]
                if missing:
                    raise RuntimeError(
                        "exit timestep ramp lost required relations: " + str(missing)
                    )
                timestep = next_timestep
        else:
            e.parameters.setdefault("precision_transitions", []).append(
                {
                    "time_s": float(d.time),
                    "stage": "exit_preparation",
                    "old_timestep_s": old_timestep,
                    "new_timestep_s": transport_timestep,
                    "reason": "empty-hand exit after supported release",
                }
            )
            m.opt.timestep = transport_timestep
        # Match the aperture for which the preparation route was planned.
        e.stage = "exit_empty_gripper_aperture"
        e.close_gripper("left", 0.23)
        work = np.asarray(self.points[-1]["qpos"])
        for arm in ["left", "right"]:
            route = plan_joint_path(
                a, arm, work[a.qadr[arm]], seed=52, iterations=3000,
                resolution_rad=0.005, obstacle_body="pot", clearance_m=0.005,
            )
            e.parameters.setdefault("exit_rejoin_plans", []).append(route)
            if not route["found"]:
                raise RuntimeError("cannot rejoin empty-hand working pose")
            for q in route["waypoints_urdf_rad"][1:]:
                target = self.motion.arm_target(arm, q)
                self.motion.move(target, "exit_rejoin_" + arm)
        for i, point in enumerate(reversed(self.points[:-1])):
            self.motion.move(point["qpos"], f"exit_lateral_route_{i:02d}")
        e.stage = "standing_final_hold"
        e.hold(2)
        if not all(self.pose_evidence(standing=True).values()):
            raise RuntimeError("standing/hanging final pose not verified")
        e.parameters["exit_completed"] = True

    def diagnostics(self):
        return {
            "maximum_body_tilt_deg": self.maximum_tilt_deg,
            "lifecycle_physics_collision_samples": self.collision_samples,
            "startup_pose_measured": False,
            "physical_motion_allowed": False,
        }
