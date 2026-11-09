"""Simulation candidates for standing/working transitions around the coffee task.

The historical coffee body was frozen at an observed posture that conflicts with
the vendor dt_joint limit. This module keeps that discrepancy explicit: the
optional observed envelope is for geometric simulation only, never hardware IO.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

BODY_JOINTS = ("xt_joint_1", "xt_joint_2", "dt_joint", "yb_joint", "xb_joint")


def articulate_frozen_body(source_xml, config, urdf, destination, *, observed_envelope=False):
    """Restore body hinges while exactly preserving the baseline FK at work q.

    Range conflicts fail by default. An explicit offline observed-envelope option
    records the original limits and provisional range in the returned metadata.
    Base remains anchored and original gravity compensation remains unchanged.
    """
    source_xml, destination = Path(source_xml), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    root = ET.parse(source_xml).getroot()
    vendor = ET.parse(urdf).getroot()
    actuator = root.find("actuator")
    frozen = config["robot"]["frozen_joints_rad"]
    rows = []
    for name in BODY_JOINTS:
        joint = vendor.find(f"joint[@name='{name}']")
        child = joint.find("child").get("link")
        body = root.find(f".//body[@name='{child}']")
        if body.find("joint") is not None:
            raise ValueError(f"body is already articulated: {child}")
        axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        axis /= np.linalg.norm(axis)
        work = float(frozen[name])
        current = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        zero = Rotation.from_quat(current[[1, 2, 3, 0]]) * Rotation.from_rotvec(-axis * work)
        body.set("quat", " ".join(map(str, zero.as_quat()[[3, 0, 1, 2]])))
        # The five body pivots are the child-frame origin in this URDF.
        limit = joint.find("limit")
        original = [float(limit.get("lower")), float(limit.get("upper"))]
        bounds = original.copy()
        conflict = not bounds[0] <= work <= bounds[1]
        if conflict:
            if not observed_envelope:
                raise ValueError(f"{name}: observed work angle {work} outside {original}")
            bounds = [min(bounds[0], work), max(bounds[1], work)]
        ET.SubElement(
            body,
            "joint",
            name=name,
            type="hinge",
            axis=" ".join(map(str, axis)),
            range=" ".join(map(str, bounds)),
            damping="10",
            armature="0.1",
        )
        effort = float(limit.get("effort"))
        ET.SubElement(
            actuator,
            "position",
            name=name + "_servo",
            joint=name,
            kp="2000",
            kv="160",
            ctrlrange=" ".join(map(str, bounds)),
            forcerange=f"{-effort} {effort}",
        )
        rows.append(
            {
                "joint": name,
                "work_rad": work,
                "standing_rad": 0.0,
                "legacy_range_rad": original,
                "simulation_range_rad": bounds,
                "legacy_range_conflict": conflict,
                "effort_prior_nm": effort,
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding="unicode")
    result = {
        "simulation_only": True,
        "physical_motion_allowed": False,
        "hardware_limits_qualified": False,
        "startup_pose_measured": False,
        "observed_envelope_explicitly_enabled": observed_envelope,
        "source_model_sha256": hashlib.sha256(source_xml.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "body_joints": rows,
        "actuator_parameters": "simulation PD priors, not identified body actuator dynamics",
        "base": "anchored; whole-body balance not evaluated",
    }
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def copy_named_state(source_model, source_data, target_model, target_data):
    """Time-zero initialization across models; not a runtime transition shortcut."""
    import mujoco

    if source_data.time != 0 or target_data.time != 0:
        raise ValueError("named initialization is restricted to time zero")
    sizes = {
        mujoco.mjtJoint.mjJNT_FREE: (7, 6),
        mujoco.mjtJoint.mjJNT_BALL: (4, 3),
        mujoco.mjtJoint.mjJNT_SLIDE: (1, 1),
        mujoco.mjtJoint.mjJNT_HINGE: (1, 1),
    }
    for i in range(source_model.njnt):
        name = source_model.joint(i).name
        j = target_model.joint(name).id
        assert source_model.jnt_type[i] == target_model.jnt_type[j]
        nq, nv = sizes[source_model.jnt_type[i]]
        qa, qb = source_model.jnt_qposadr[i], target_model.jnt_qposadr[j]
        va, vb = source_model.jnt_dofadr[i], target_model.jnt_dofadr[j]
        target_data.qpos[qb : qb + nq] = source_data.qpos[qa : qa + nq]
        target_data.qvel[vb : vb + nv] = source_data.qvel[va : va + nv]
    for i in range(source_model.nu):
        target_data.ctrl[target_model.actuator(source_model.actuator(i).name).id] = (
            source_data.ctrl[i]
        )


def body_tilt_deg(model, data):
    return float(
        np.rad2deg(np.arccos(np.clip(data.body("xb_link").xmat.reshape(3, 3)[2, 2], -1, 1)))
    )


class LifecycleMotion:
    """Execute planned empty-hand configuration segments through position servos."""

    def __init__(self, rollout, *, body_speed=0.15, arm_speed=0.4):
        self.rollout = rollout
        self.body_speed, self.arm_speed = body_speed, arm_speed
        if not np.isfinite([body_speed, arm_speed]).all() or min(body_speed, arm_speed) <= 0:
            raise ValueError("positive finite speeds required")
        m = rollout.adapter.model
        self.names = [*BODY_JOINTS, *(f"{s}_joint_{i}" for s in ["l", "r"] for i in range(1, 8))]
        self.jids = np.array([m.joint(n).id for n in self.names])
        self.qids = m.jnt_qposadr[self.jids]
        self.vids = m.jnt_dofadr[self.jids]
        self.aids = np.array([m.actuator(n + "_servo").id for n in self.names])

    def arm_target(self, arm, joint_positions):
        """Retain other servos' references when composing a single-arm move.

        A measured joint may sit just beyond a soft limit. It is a valid start
        measurement, but must not become a new out-of-range hold command.
        ``move`` still validates every composed target against the same limits.
        """
        a = self.rollout.adapter
        if arm not in {"left", "right"}:
            raise ValueError("unknown arm")
        joints = np.asarray(joint_positions, dtype=float)
        if joints.shape != (7,):
            raise ValueError("seven arm joint targets required")
        target = a.data.qpos.copy()
        target[self.qids] = a.data.ctrl[self.aids]
        target[a.qadr[arm]] = joints
        return target

    def move(self, full_qpos, stage):
        e = self.rollout
        m, d = e.adapter.model, e.adapter.data
        target = np.asarray(full_qpos, dtype=float)[self.qids]
        limits = m.jnt_range[self.jids]
        if (
            not np.isfinite(target).all()
            or np.any(target < limits[:, 0])
            or np.any(target > limits[:, 1])
        ):
            raise ValueError("lifecycle target outside declared simulation envelope")
        start = d.qpos[self.qids].copy()
        speeds = np.array([self.body_speed] * len(BODY_JOINTS) + [self.arm_speed] * 14)
        duration = max(0.2, 1.875 * float(np.max(np.abs(target - start) / speeds)))
        count = int(np.ceil(duration / 0.02))
        e.stage = stage
        for i in range(1, count + 1):
            u = i / count
            alpha = 10 * u**3 - 15 * u**4 + 6 * u**5
            d.ctrl[self.aids] = start + alpha * (target - start)
            e.tick()
            if np.max(np.abs(d.qpos[self.qids] - d.ctrl[self.aids])) > 0.15:
                raise RuntimeError("lifecycle joint tracking error")
        stable = 0
        for _ in range(500):
            e.tick()
            error = float(np.max(np.abs(d.qpos[self.qids] - target)))
            velocity = float(np.max(np.abs(d.qvel[self.vids])))
            stable = stable + 1 if error < 0.002 and velocity < 0.01 else 0
            if stable >= 25:
                break
        if stable < 25:
            raise RuntimeError(f"lifecycle did not settle: error={error}, velocity={velocity}")
        e.parameters.setdefault("lifecycle_segments", []).append(
            {
                "stage": stage,
                "end_time_s": float(d.time),
                "trajectory_s": count * 0.02,
                "joint_error_rad": error,
                "maximum_joint_speed_rad_s_at_end": velocity,
            }
        )
