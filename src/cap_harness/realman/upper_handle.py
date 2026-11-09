"""Upper-handle bearing geometry and bounded outlet recovery, simulation only."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .grasp_geometry import grasp_geometry, pot_handle_grasp_pose


def upper_handle_target(
    data, pot_config, *, height_m, side_tilt_deg=0, radial_m=0, depth_m=0, inplane_deg=0
):
    """Approach approximately normal to the loop plane; preserve stock fingers.

    Height is relative to the existing outer-rail midpoint. This is a geometric
    target, not proof that the inner finger actually bears on the upper bar.
    """
    if not np.isfinite([height_m, side_tilt_deg, radial_m, depth_m, inplane_deg]).all():
        raise ValueError("nonfinite grasp target")
    if not 0 <= height_m <= 0.04 or abs(side_tilt_deg) > 10 or abs(inplane_deg) > 15:
        raise ValueError("upper side grasp outside configured search envelope")
    target = pot_handle_grasp_pose(data)
    pot_rotation = data.body("pot").xmat.reshape(3, 3)
    target[:3, 3] += pot_rotation @ np.array([radial_m, depth_m, height_m])
    slope = -np.arctan2(pot_config["handle_bottom_inset_m"], pot_config["handle_outer_height_m"])
    target[:3, :3] = (
        target[:3, :3]
        @ Rotation.from_euler("z", slope + np.deg2rad(inplane_deg)).as_matrix()
        @ Rotation.from_euler("x", side_tilt_deg, degrees=True).as_matrix()
    )
    return target


def upper_bar_contacts(model, data, pot_config, *, arm="left"):
    """Measure normal-force bearing separately from side-contact friction.

    A higher wrist, any handle contact, or a friction force is insufficient.
    Report actual upper-region contacts with a normal supporting pot-local +Z.
    No contacts are exempted from the independent robot collision checks here.
    """
    import mujoco

    pot = model.body("pot").id
    rotation = data.body("pot").xmat.reshape(3, 3)
    top = pot_config["handle_bottom_m"] + pot_config["handle_outer_height_m"]
    outer = pot_config["handle_outer_to_center_m"]
    rows = []
    for i, contact in enumerate(data.contact[: data.ncon]):
        geoms = [contact.geom1, contact.geom2]
        bodies = [int(model.geom_bodyid[g]) for g in geoms]
        if pot not in bodies:
            continue
        side = bodies.index(pot)
        other = 1 - side
        if not model.body(bodies[other]).name.startswith(arm[0] + "tool_"):
            continue
        if not model.geom(geoms[side]).name.startswith("handle_collision_"):
            continue
        local = rotation.T @ (contact.pos - data.body("pot").xpos)
        if not (
            top - pot_config["handle_rail_thickness_m"] - 0.006 <= local[2] <= top + 0.003
            and outer - pot_config["handle_outer_width_m"] <= local[0] <= outer + 0.003
        ):
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = contact.frame[:3] * (1 if side == 1 else -1)
        normal_local = rotation.T @ normal
        # A shallow upper-bar facet still supports weight. Require a positive
        # measured upward force, not an arbitrary minimum facet inclination.
        if normal_local[2] <= 0 or wrench[0] * normal_local[2] <= 0.001:
            continue
        rows.append(
            {
                "contact_index": i,
                "pot_geom": model.geom(geoms[side]).name,
                "tool_geom": model.geom(geoms[other]).name,
                "position_pot_m": local.tolist(),
                "normal_pot": normal_local.tolist(),
                "normal_force_n": float(wrench[0]),
                "local_upward_normal_n": float(wrench[0] * normal_local[2]),
                "world_upward_normal_n": float(wrench[0] * normal[2]),
            }
        )
    G = np.asarray(grasp_geometry(model, data, arm)["T_world_grasp"])
    alignment = abs(float(G[:3, 2] @ rotation[:, 1]))
    return {
        "contacts": rows,
        "local_upward_normal_n": sum(c["local_upward_normal_n"] for c in rows),
        "side_approach_deviation_deg": float(np.rad2deg(np.arccos(np.clip(alignment, -1, 1)))),
    }


def upper_handle_loads(model, data, pot_config, *, arm="left"):
    """Count opposing finger loads, including qualified upper-bar bearing.

    This mode requires real contact on BOTH fingers. Pure vertical support or
    contact elsewhere on the tool does not replace an opposing closing force.
    Preserve raw pad readings so a zero pad load is never hidden in the logs.
    """
    import mujoco

    bearing = upper_bar_contacts(model, data, pot_config, arm=arm)
    qualified = {c["contact_index"] for c in bearing["contacts"]}
    axis = np.asarray(grasp_geometry(model, data, arm)["T_world_grasp"])[:3, 0]
    pot = model.body("pot").id
    pads, fingers, closing = np.zeros(2), np.zeros(2), np.zeros(2)
    normal_vectors = np.zeros((2, 3))
    peak_contact = 0.0
    for i, c in enumerate(data.contact[: data.ncon]):
        geoms = [c.geom1, c.geom2]
        bodies = [int(model.geom_bodyid[g]) for g in geoms]
        if pot not in bodies:
            continue
        side = bodies.index(pot)
        object_name, tool_name = model.geom(geoms[side]).name, model.geom(geoms[1 - side]).name
        if not object_name.startswith("handle_collision_"):
            continue
        for k, name in enumerate(["left", "right"]):
            pad = tool_name == f"{arm}_{name}_pad_contact"
            bar = i in qualified and tool_name.startswith(f"{arm[0]}tool_{name.capitalize()}_")
            finger_contact = tool_name.startswith(f"{arm[0]}tool_{name.capitalize()}_")
            if not pad and not finger_contact:
                continue
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            force = max(0.0, float(wrench[0]))
            normal = c.frame[:3] * (1 if side == 1 else -1)
            inward = -axis if k == 0 else axis
            pads[k] += force if pad else 0
            fingers[k] += force
            normal_vectors[k] += force * normal
            peak_contact = max(peak_contact, force)
            # Every finger/handle contact contributes to overload diagnostics;
            # only pads or identified bearing contacts establish a grasp.
            if pad or bar:
                closing[k] += max(0.0, float(normal @ inward)) * force
    return {
        **bearing,
        "raw_pad_forces_n": pads.tolist(),
        "finger_normal_loads_n": fingers.tolist(),
        "finger_normal_resultants_n": np.linalg.norm(normal_vectors, axis=1).tolist(),
        "finger_normal_vectors_world_n": normal_vectors.tolist(),
        "maximum_contact_normal_n": peak_contact,
        "opposing_closing_forces_n": closing.tolist(),
        "opposed_contact": bool(np.min(closing) > 0.02),
    }


def handle_axis_to_seat_error(pot_pose, seat_position, *, minimum_distance_m=0.025):
    """Unsigned horizontal axis error, undefined near final center alignment."""
    pose, seat = np.asarray(pot_pose, dtype=float), np.asarray(seat_position, dtype=float)
    if (
        pose.shape != (4, 4)
        or seat.shape != (3,)
        or not np.isfinite(pose).all()
        or not np.isfinite(seat).all()
    ):
        raise ValueError("invalid handle-axis geometry")
    if not np.isfinite(minimum_distance_m) or minimum_distance_m <= 0:
        raise ValueError("invalid near-seat distance")
    direction, axis = pose[:2, 3] - seat[:2], pose[:2, 0]
    distance = np.linalg.norm(direction)
    if distance < minimum_distance_m:
        return None
    if np.linalg.norm(axis) < 1e-9:
        raise ValueError("handle axis has no horizontal projection")
    return float(
        np.rad2deg(
            np.arccos(np.clip(abs(direction @ axis) / (distance * np.linalg.norm(axis)), 0, 1))
        )
    )


@dataclass
class OutletUnilateralAllowance:
    """Operator-authorized short one-pad loss, only while positively constrained.

    Existing monitors keep their original anchors and timers. The allowance
    suppresses only unilateral/asymmetric flags, never both-pad loss, motion,
    overload, or collision flags. Once disengaged, it cannot re-arm on return.
    """

    maximum_s: float = 0.2
    disengaged: bool = False

    def __post_init__(self):
        if not np.isfinite(self.maximum_s) or not 0 < self.maximum_s <= 0.25:
            raise ValueError("invalid outlet one-pad-loss window")

    def apply(
        self,
        failures,
        *,
        extracting,
        outlet_contact,
        platform_support_n,
        outlet_clear,
        unilateral_s,
        bilateral_s,
    ):
        if not np.isfinite([platform_support_n, unilateral_s, bilateral_s]).all():
            raise ValueError("nonfinite outlet recovery evidence")
        if outlet_clear:
            self.disengaged = True
        eligible = (
            extracting
            and not self.disengaged
            and outlet_contact
            and platform_support_n > 0.02
            and unilateral_s < self.maximum_s
            and bilateral_s == 0
        )
        allowed = {"persistent_unilateral_unload", "pad_asymmetric_overload"} if eligible else set()
        return [f for f in failures if f not in allowed], [f for f in failures if f in allowed]
