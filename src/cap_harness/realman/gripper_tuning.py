"""Explicit, opt-in MuJoCo gripper tuning; never a hardware force command.

Links are already rigid bodies. These settings change actuator and constraint
response, not material strength, yield stress, backlash or finger bending.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class GripperSimulationProfile:
    kp_nm_per_rad: float
    kv_nms_per_rad: float
    effort_limit_nm: float
    linkage_time_constant_s: float = 0.004
    pad_time_constant_s: float = 0.005
    joint_damping_nms_per_rad: float = 0.03


def apply_gripper_profile(root, arm, profile):
    """Edit an XML tree before compilation; preserve geometry and friction.

    Reject efforts above the imported joint limit. This is a simulator limit,
    not certification of the motor, linkage, pads or held object.
    All validation precedes mutation. The other gripper remains unchanged.
    """
    if arm not in {"left", "right"}:
        raise ValueError("unknown arm")
    values = asdict(profile)
    if any(not math.isfinite(v) or v <= 0 for v in values.values()):
        raise ValueError("gripper parameters must be finite and positive")
    option = root.find("option")
    timestep = float(option.get("timestep", "0.002")) if option is not None else 0.002
    if min(profile.linkage_time_constant_s, profile.pad_time_constant_s) < 2 * timestep:
        raise ValueError("constraint time constants must be at least twice the timestep")
    prefix = arm[0] + "tool_"
    servo = root.find(f".//actuator/position[@name='{arm}_gripper_servo']")
    joint = root.find(f".//body/joint[@name='{prefix}Left_1_Joint']")
    equalities = [
        node
        for node in root.findall(".//equality/joint")
        if node.get("joint1", "").startswith(prefix)
    ]
    joints = [
        node for node in root.findall(".//body/joint") if node.get("name", "").startswith(prefix)
    ]
    pads = [root.find(f".//geom[@name='{arm}_{side}_pad_contact']") for side in ["left", "right"]]
    if (
        servo is None
        or joint is None
        or len(equalities) != 5
        or len(joints) != 6
        or any(p is None for p in pads)
    ):
        raise ValueError("expected the complete imported CTAG gripper and pad proxies")
    limits = [float(v) for v in joint.get("actuatorfrcrange", "").split()]
    if len(limits) != 2 or not all(math.isfinite(v) for v in limits):
        raise ValueError("missing imported gripper actuator limits")
    if profile.effort_limit_nm > min(-limits[0], limits[1]):
        raise ValueError("profile exceeds imported joint actuator limits")
    if servo.get("joint") != joint.get("name"):
        raise ValueError("unexpected gripper actuator mapping")
    servo.set("kp", str(profile.kp_nm_per_rad))
    servo.set("kv", str(profile.kv_nms_per_rad))
    servo.set("forcerange", f"{-profile.effort_limit_nm} {profile.effort_limit_nm}")
    for node in equalities:
        node.set("solref", f"{profile.linkage_time_constant_s} 1")
    for node in pads:
        node.set("solref", f"{profile.pad_time_constant_s} 1")
    for node in joints:
        node.set("damping", str(profile.joint_damping_nms_per_rad))
    return {
        "arm": arm,
        "profile": values,
        "simulation_only": True,
        "physical_force_mapping_verified": False,
        "material_strength_modeled": False,
        "geometry_or_friction_changed": False,
    }


def resize_pad_plane(root, arm, *, pad_y_span_m, pad_z_span_m):
    """Resize an explicit box pad concept in its own URDF Pad_Link frame.

    +X is the pad normal. Y and Z are its in-plane axes, NOT the corresponding
    grasp-frame axes. In the current outer-handle grasp, Z is approximately
    vertical and Y follows handle depth. Check this from FK for other grasps.
    Thickness, surface sites and friction are unchanged. Mass scales with area
    at the existing assumed density; this is not a stiffness/strength design.
    """
    if arm not in {"left", "right"}:
        raise ValueError("unknown arm")
    if any(not math.isfinite(v) or v <= 0 for v in (pad_y_span_m, pad_z_span_m)):
        raise ValueError("pad spans must be finite and positive")
    changes = []
    for side in ("Left", "Right"):
        body = root.find(f".//body[@name='{arm[0]}tool_{side}_Pad_Link']")
        pad = None if body is None else body.find(f"geom[@name='{arm}_{side.lower()}_pad_contact']")
        if (
            body is None
            or pad is None
            or body.find("inertial") is not None
            or pad.get("type") != "box"
        ):
            raise ValueError("need box pads with geometry-derived inertia")
        size = [float(v) for v in pad.get("size", "").split()]
        mass = float(pad.get("mass", "nan"))
        if len(size) != 3 or any(not math.isfinite(v) or v <= 0 for v in [*size, mass]):
            raise ValueError("need explicit positive pad dimensions and mass")
        new_size = [size[0], pad_y_span_m / 2, pad_z_span_m / 2]
        new_mass = mass * new_size[1] * new_size[2] / (size[1] * size[2])
        if not math.isfinite(new_mass) or new_mass <= 0:
            raise ValueError("pad mass overflow or underflow")
        changes.append((pad, new_size, new_mass))
    for pad, size, mass in changes:
        pad.set("size", " ".join(map(str, size)))
        pad.set("mass", str(mass))
    return {
        "arm": arm,
        "pad_y_span_m": pad_y_span_m,
        "pad_z_span_m": pad_z_span_m,
        "simulation_only": True,
        "geometry_changed": True,
        "friction_changed": False,
        "mass_scaling": "constant assumed density, unchanged thickness",
        "material_and_attachment_strength_verified": False,
        "collision_clearance_verified": False,
    }
