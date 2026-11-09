"""Explicit conical outlet hypothesis; simulation only, not hardware metrology."""

import math
import xml.etree.ElementTree as ET


def install_conical_outlet(asset, body, config, *, lid_initial_z_m):
    """Replace the outlet alone; z is relative to the outlet's parent body.

    Reference compression includes the pot's initial 1 mm hotplate settling gap.
    q=0 is the extended mechanical stop, not the installed/loaded position.
    The convex tip is a contact envelope; its internal water channel is omitted.
    """
    o = config["outlet"]
    h = o["conical_hypothesis"]
    rb, rt, height = (float(h[k]) for k in ("bottom_radius_m", "top_radius_m", "height_m"))
    hole = float(config["pot"]["lid_hole_radius_m"])
    seated, gap = float(h["seated_compression_m"]), float(h["initial_pot_settling_gap_m"])
    k, preload, mu = (
        float(h[key]) for key in ("stiffness_n_per_m", "preload_at_zero_n", "sliding_friction")
    )
    travel = float(o["travel_m"])
    n = h.get("segments", 64)
    values = [rb, rt, height, hole, seated, gap, k, preload, mu, travel, lid_initial_z_m]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("conical outlet needs finite parameters")
    if not (
        0 < rb < hole < rt
        and height > 0
        and 0 < seated < travel
        and gap >= 0
        and k > 0
        and preload >= 0
        and 0 < mu <= 1
    ):
        raise ValueError("invalid conical outlet dimensions or force hypothesis")
    insertion = height * (hole - rb) / (rt - rb)
    if seated + gap >= travel or seated + insertion >= travel:
        raise ValueError("conical cam rise consumes available outlet travel")
    if type(n) is not int or not 16 <= n <= 128:
        raise ValueError("conical mesh needs 16..128 segments")
    pos = list(map(float, body.get("pos").split()))
    pos[2] = lid_initial_z_m - insertion
    body.set("pos", " ".join(map(str, pos)))
    for child in list(body):
        body.remove(child)
    ET.SubElement(
        body,
        "joint",
        name="outlet_compression",
        type="slide",
        axis="0 0 1",
        range=f"0 {travel}",
        ref=str(seated + gap),
        stiffness=str(k),
        springref=str(-preload / k),
        damping=str(h.get("damping_n_s_m", 0.2)),
        solreflimit="0.002 1",
        solimplimit="0.999 0.999 0.001",
    )
    vertices = []
    for z, r in [(0, rb), (height, rt)]:
        for i in range(n):
            a = 2 * math.pi * i / n
            vertices.extend([r * math.cos(a), r * math.sin(a), z])
    if asset.find("mesh[@name='outlet_conical_tip']") is not None:
        raise ValueError("conical outlet mesh already exists")
    ET.SubElement(asset, "mesh", name="outlet_conical_tip", vertex=" ".join(map(str, vertices)))
    # Priority selects an explicit outlet/lid interface friction, rather than
    # silently inheriting max(default-outlet friction, lid friction).
    ET.SubElement(
        body,
        "geom",
        name="outlet_tip",
        type="mesh",
        mesh="outlet_conical_tip",
        mass="0.02",
        rgba="0.15 0.15 0.16 1",
        priority="1",
        friction=f"{mu} 0.005 0.0001",
        condim="4",
    )
    o["spring_n_per_m"], o["preload_n"] = k, preload
    return {
        "simulation_only": True,
        "hardware_measured": False,
        "shape": "solid_convex_frustum_contact_envelope",
        "cam_rise_m": insertion,
        "body_initial_bottom_z_in_parent_m": pos[2],
        "initial_reference_q_m": seated + gap,
        "expected_seated_q_m": seated,
        "spring_force_seated_n": preload + k * seated,
        "spring_force_at_cam_crest_n": preload + k * (seated + insertion),
        "outlet_weight_n": 0.02 * 9.81,
        "interface_friction": mu,
        "parameters": dict(h),
    }
