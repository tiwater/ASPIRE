"""Offline audit of a native state snapshot. Never connects to or commands a robot.

The default identity joint mapping is deliberately UNQUALIFIED. Raw controller
and URDF poses are printed separately: their base and tool frames differ, so a
numerical subtraction is not a physical pose error or a collision certificate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def forward_chain(root, base, tip, angles):
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain, link = [], tip
    while link != base:
        joint = parents[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    transform = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        local = np.eye(4)
        if origin is not None:
            local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            local[:3, :3] = Rotation.from_euler(
                "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            ).as_matrix()
        transform = transform @ local
        if joint.get("type") != "fixed":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            axis /= np.linalg.norm(axis)
            motion = np.eye(4)
            angle = angles.get(joint.get("name"), 0.0)
            if joint.get("type") in ("revolute", "continuous"):
                motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            else:
                raise ValueError("audit supports revolute/fixed chains only")
            transform = transform @ motion
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = ET.parse(args.urdf).getroot()
    state = json.loads(args.state.read_text())
    angles = {}
    for arm, prefix in [("left", "l"), ("right", "r")]:
        for i, value in enumerate(state["controller"][arm]["joint_deg"], 1):
            angles[f"{prefix}_joint_{i}"] = np.deg2rad(value)
    report = {
        "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "state_sha256": hashlib.sha256(args.state.read_bytes()).hexdigest(),
        "qualified_for_motion": False,
        "mapping_assumption": "identity; NOT calibrated",
        "other_joint_assumption": "zero; NOT live body/head/gripper/mount mapping",
        "controller_base_to_urdf_base": None,
        "controller_tip_to_urdf_tip": None,
        "self_collision_validated": False,
        "arms": {},
    }
    for arm, prefix in [("left", "l"), ("right", "r")]:
        base, tip = f"{arm}_arm_base_link", f"{prefix}_link_7"
        local = forward_chain(root, base, tip, angles)
        world = forward_chain(root, "base_link", tip, angles)
        report["arms"][arm] = {
            "joint_deg_controller": state["controller"][arm]["joint_deg"],
            "controller_pose_raw": state["controller"][arm]["pose_raw"],
            "T_urdf_arm_base_link7_identity": local.tolist(),
            "T_urdf_base_link_link7_identity": world.tolist(),
            "urdf_joint7_axis": root.find(f"joint[@name='{prefix}_joint_7']/axis").get("xyz"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "qualified_for_motion": False}))


if __name__ == "__main__":
    main()
