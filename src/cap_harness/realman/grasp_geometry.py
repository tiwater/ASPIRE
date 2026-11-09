"""Dynamic simulation grasp frames; no controller coordinates or physical commands."""

import numpy as np


def pivot_yaw_pose(initial_pose, pivot_world_m, angle_rad):
    """Move a grasp on a horizontal arc about an object's vertical centre.

    This is a target transform only; it never writes a simulated object pose.
    Applying it to a rigid grasp preserves the object's pivot position.
    """
    initial = np.asarray(initial_pose, dtype=float)
    pivot = np.asarray(pivot_world_m, dtype=float)
    if (
        initial.shape != (4, 4)
        or pivot.shape != (3,)
        or not np.isfinite(initial).all()
        or not np.isfinite(pivot).all()
        or not np.isfinite(angle_rad)
    ):
        raise ValueError("need a finite pose, pivot and yaw angle")
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    result = initial.copy()
    result[:3, :3] = rotation @ initial[:3, :3]
    result[:3, 3] = pivot + rotation @ (initial[:3, 3] - pivot)
    return result


def pot_handle_grasp_pose(data):
    """Oracle target: opening along the outer rail, approach through the loop.

    Pot +X points toward the handle, +Z is upright. Follow the observed object
    rotation instead of retaining an obsolete nominal wrist orientation.
    """
    result = np.eye(4)
    rotation = data.body("pot").xmat.reshape(3, 3)
    result[:3, :3] = rotation[:, [0, 2, 1]] * [1, 1, -1]
    result[:3, 3] = data.site("pot_handle_grasp").xpos
    return result


def gripper_linkage_errors(model, data, arm):
    """URDF mimic residuals in radians for this gripper's hinge linkage."""
    import mujoco

    if arm not in ("left", "right"):
        raise ValueError("unknown arm")
    errors = {}
    for i in range(model.neq):
        if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        j1, j2 = model.eq_obj1id[i], model.eq_obj2id[i]
        name = model.joint(j1).name
        if not name.startswith(arm[0] + "tool_"):
            continue
        a1 = model.jnt_qposadr[j1]
        x = data.qpos[a1] - model.qpos0[a1]
        y = 0.0
        if j2 >= 0:
            a2 = model.jnt_qposadr[j2]
            y = data.qpos[a2] - model.qpos0[a2]
        errors[name] = float(x - np.polynomial.polynomial.polyval(y, model.eq_data[i, :5]))
    return errors


def grasp_geometry(model, data, arm):
    """Use pad surface midpoints; +X opening axis, +Z approach, +Y completes frame.

    Gap is signed along the gripper-base X axis, preserving closed/crossed geometry
    rather than turning overlap into a positive aperture. URDF symmetry defines
    this axis; no measured hardware opening curve or TCP is implied.
    """
    if arm not in ("left", "right"):
        raise ValueError("unknown arm")
    mount = data.site(arm + "_tool_mount")
    mount_rotation = mount.xmat.reshape(3, 3)
    # URDF tool-base mounting includes an axial rotation; flange X is not jaw X.
    rotation = data.body(arm[0] + "tool_base_link").xmat.reshape(3, 3)
    points = np.array(
        [data.site(f"{arm}_{jaw}_pad_surface").xpos.copy() for jaw in ["left", "right"]]
    )
    center = points.mean(axis=0)
    gap = float((points[0] - points[1]) @ rotation[:, 0])
    transform = np.eye(4)
    transform[:3, :3] = mount_rotation.T @ rotation
    transform[:3, 3] = mount_rotation.T @ (center - mount.xpos)
    world = np.eye(4)
    world[:3, :3] = rotation
    world[:3, 3] = center
    return {
        "arm": arm,
        "frame": "world",
        "T_world_grasp": world.tolist(),
        "T_mount_grasp": transform.tolist(),
        "pad_surface_positions_world_m": points.tolist(),
        "signed_aperture_m": gap,
        "aperture_positive": gap > 0,
        "opening_axis_world": rotation[:, 0].tolist(),
        "approach_axis_world": rotation[:, 2].tolist(),
        "simulation_only": True,
        "hardware_tcp_qualified": False,
        "grasp_verified": False,
        "contact_proxy": "configured rectangular Pad_Link geometry",
    }


def mount_pose_for_grasp(T_world_grasp, T_mount_grasp):
    """Convert desired grasp pose for an explicit opening into a mounting target.

    This calculation is geometry only: it performs neither IK nor path checking.
    Recompute after aperture changes; never send these URDF/world poses to hardware.
    """
    matrices = [np.asarray(x, dtype=float) for x in [T_world_grasp, T_mount_grasp]]
    for matrix in matrices:
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)
        ):
            raise ValueError("invalid rigid transform")
    return matrices[0] @ np.linalg.inv(matrices[1])
