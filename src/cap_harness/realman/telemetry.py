"""Decode the observed RealBot ROS JSON schema without inventing FK or tool poses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json

import numpy as np


def vector(value, length: int, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name}: expected {length} finite numeric values")
    return raw.astype(float)


def millidegrees_to_radians(value, length=7) -> np.ndarray:
    return np.deg2rad(vector(value, length, "millidegrees") / 1000.0)


@dataclass(frozen=True)
class Telemetry:
    """Angles are controller conventions, NOT certified URDF coordinates.

    State messages have no sensor timestamp; received_at_s is host receipt time.
    Missing velocities and end-effector poses deliberately remain unavailable.
    """

    joints_rad: Mapping[str, np.ndarray]
    gripper_native: Mapping[str, float]
    body_rad: np.ndarray
    head_rad: np.ndarray
    faults: tuple[str, ...]
    received_at_s: float

    def as_dict(self):
        return {
            "joint_positions_controller_rad": {k: v.tolist() for k, v in self.joints_rad.items()},
            "gripper_native": dict(self.gripper_native),
            "body_controller_rad": self.body_rad.tolist(),
            "head_controller_rad": self.head_rad.tolist(),
            "faults": list(self.faults),
            "received_at_s": self.received_at_s,
            "source_timestamp_available": False,
            "end_effector_poses": None,
            "urdf_mapping_qualified": False,
        }


def decode_state(payload: str | Mapping, *, received_at_s: float) -> Telemetry:
    raw = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(raw, Mapping) or not np.isfinite(received_at_s) or received_at_s < 0:
        raise ValueError("invalid state envelope")
    joints, grippers, faults = {}, {}, []
    for arm in ("left", "right"):
        state = raw[f"arm_{arm}"]
        joints[arm] = millidegrees_to_radians(state["position"])
        errors = vector(state["joint_err"], 7, "joint_err")
        enabled = vector(state["joint_en"], 7, "joint_en")
        if state["sys_err"] != 0 or np.any(errors != 0):
            faults.append(f"{arm}: controller error")
        if np.any(enabled != 1):
            faults.append(f"{arm}: joint disabled")
        hand = raw[f"hand_{arm}"]
        pos = float(vector(hand["position"], 1, "gripper position")[0])
        if not 0 <= pos <= 1000:
            raise ValueError(f"{arm}: gripper outside observed native range")
        grippers[arm] = pos
    for part, n in (("body", 5), ("head", 2)):
        state = raw[part]
        if state["sys_err"] != 0 or np.any(vector(state["joint_err"], n, part) != 0):
            faults.append(f"{part}: controller error")
    return Telemetry(
        joints,
        grippers,
        millidegrees_to_radians(raw["body"]["position"], 5),
        millidegrees_to_radians(raw["head"]["position"], 2),
        tuple(faults),
        received_at_s,
    )


def commissioning_report(config: Mapping) -> dict:
    pending = []
    for name, record in config["commissioning"].items():
        if (
            not isinstance(record, Mapping)
            or record.get("status") != "verified"
            or not record.get("evidence_path")
        ):
            pending.append(name)
    return {
        "pending": pending,
        "physical_motion_available": False,
        "reason": "RealMan MCP control is separately commissioned and observation-only by default. This offline report cannot certify runtime gates, collision clearance or stop behavior; calibration records alone do not enable motion.",
    }
