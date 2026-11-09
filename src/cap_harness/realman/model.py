"""Reproducible MuJoCo model from vendor URDF and photo-derived scene parameters.

This is a geometric/contact workbench, not a calibrated digital twin. No fluids,
paper deformation, certified motor dynamics, camera calibration or real IO.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .platform import install_recessed_platform, nominal_seated_height


def numbers(values):
    return " ".join(f"{float(x):.9g}" for x in values)


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def geom(parent, name, kind, size, pos=(0, 0, 0), rgba="0.9 0.88 0.8 1", **attrs):
    return element(
        parent,
        "geom",
        name=name,
        type=kind,
        size=numbers(size),
        pos=numbers(pos),
        rgba=rgba,
        **attrs,
    )


def shell(
    asset,
    parent,
    name,
    radii,
    height,
    thickness,
    *,
    z=0,
    segments=24,
    rgba="0.9 0.88 0.8 1",
    mass=0.01,
):
    """A hollow frustum made of convex wedges, never a solid convex cup hull."""
    rb, rt = radii
    if min(rb, rt) <= thickness or height <= 0:
        raise ValueError("invalid shell dimensions")
    for i in range(segments):
        angles = [2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments]
        vertices = []
        for zz, radius in [(z, rb), (z + height, rt)]:
            for rr in [radius - thickness, radius]:
                for angle in angles:
                    vertices.extend([rr * math.cos(angle), rr * math.sin(angle), zz])
        mesh = f"{name}_segment_{i}"
        element(asset, "mesh", name=mesh, vertex=numbers(vertices))
        element(
            parent,
            "geom",
            name=mesh,
            type="mesh",
            mesh=mesh,
            rgba=rgba,
            mass=mass / segments,
            friction="0.5 0.005 0.0001",
            condim=4,
        )


def cup(asset, world, config, name, xyz):
    c = config["cup"]
    roll = float(c.get("rolled_rim_radius_m", 0))
    if not math.isfinite(roll) or not 0 <= roll < min(c["wall_m"] * 2, c["height_m"] / 10):
        raise ValueError("invalid rolled cup rim")
    friction = float(c.get("sliding_friction", 0.5))
    if not math.isfinite(friction) or not 0 < friction <= 2:
        raise ValueError("cup sliding friction must be in (0, 2]")
    body = element(world, "body", name=name, pos=numbers(xyz))
    element(body, "freejoint", name=name + "_free")
    geom(
        body,
        name + "_bottom",
        "cylinder",
        [c["bottom_radius_m"], c["wall_m"] / 2],
        pos=[0, 0, c["wall_m"] / 2],
        mass=c["mass_kg"] * 0.2,
    )
    shell(
        asset,
        body,
        name + "_wall",
        [c["bottom_radius_m"], c["rim_radius_m"] - roll],
        c["height_m"] - roll,
        c["wall_m"],
        mass=c["mass_kg"] * (0.7 if roll else 0.8),
    )
    if roll:
        # Hollow rolled lip: short capsules form a closed toroidal contact ring.
        # The specified rim_radius remains the OUTERMOST radius, not its centerline.
        segments = 48
        radius, z = c["rim_radius_m"] - roll, c["height_m"] - roll
        for i in range(segments):
            angles = [2 * math.pi * j / segments for j in (i, i + 1)]
            endpoints = [[radius * math.cos(a), radius * math.sin(a), z] for a in angles]
            element(
                body,
                "geom",
                name=f"{name}_rolled_rim_{i}",
                type="capsule",
                size=str(roll),
                fromto=numbers([v for p in endpoints for v in p]),
                mass=c["mass_kg"] * 0.1 / segments,
                rgba="0.94 0.92 0.84 1",
                condim="4",
                solref="0.005 1",
                solimp="0.999 0.999 0.001",
                solmix="1000",
            )
    # Include the bottom: leaving its default coefficient at 1 would override
    # the lower wall coefficient in cup/cup contacts (MuJoCo takes the max).
    for surface in body.findall("geom"):
        surface.set("friction", numbers([friction, 0.005, 0.0001]))
        if roll:
            # The 1 mm wall must resist the same pull load as its rolled lip.
            # A soft wall next to a stiff lip can interpenetrate and hook a nested cup.
            surface.set("solref", "0.005 1")
            surface.set("solimp", "0.999 0.999 0.001")
            surface.set("solmix", "1000")
    element(
        body,
        "site",
        name=name + "_rim",
        pos=numbers([0, 0, c["height_m"]]),
        size="0.002",
        rgba="0 0.6 0 1",
    )
    return body


def rounded_profile(xmin, xmax, zmin, zmax, left_radius, right_radius, steps=12):
    """Counterclockwise XZ outline with corresponding samples on each corner."""
    points = []
    for cx, cz, radius, start in [
        (xmax - right_radius, zmax - right_radius, right_radius, 0),
        (xmin + left_radius, zmax - left_radius, left_radius, 90),
        (xmin + left_radius, zmin + left_radius, left_radius, 180),
        (xmax - right_radius, zmin + right_radius, right_radius, 270),
    ]:
        for i in range(steps + 1):
            angle = math.radians(start + i * 90 / steps)
            points.append((cx + radius * math.cos(angle), cz + radius * math.sin(angle)))
    return points


def pot_vessel(asset, pot, p, glass):
    """Hollow shoulder/neck/rim with an integral pouring lip on local -X.

    Each axial/angular wall patch has its own convex collision hull, so neither
    the cavity nor the concave neck is filled by a whole-vessel convex hull.
    The lip extends from the neck, while remaining inside the belly envelope.
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    profile = np.asarray(p["body_profile_z_radius_m"], dtype=float)
    wall, tip = p["wall_m"], p["spout_tip_radius_m"]
    start, spread = p["spout_flare_start_m"], p["spout_half_angle_rad"]
    lid_radius, cutback = p["lid_radius_m"], p["lid_spout_cutback_radius_m"]
    if (
        profile.ndim != 2
        or profile.shape[1] != 2
        or len(profile) < 2
        or not np.isfinite(profile).all()
        or not np.all(np.diff(profile[:, 0]) > 0)
        or profile[0, 0] != 0
        or profile[-1, 0] != p["height_m"]
        or np.any(profile[:, 1] <= wall)
        or np.any(profile[:, 1] > p["radius_m"])
        or not profile[-1, 1] <= tip <= p["radius_m"]
        or not 0 < start < p["height_m"]
        or not 0 < spread < math.pi / 2
        or not p["lid_hole_radius_m"] < cutback < lid_radius < tip - wall
    ):
        raise ValueError("invalid pot profile, recessed spout or lid cutback")
    segments = 64
    angles = np.arange(segments) * 2 * math.pi / segments
    delta = np.abs(np.arctan2(np.sin(angles - math.pi), np.cos(angles - math.pi)))
    weight = np.where(delta < spread, (1 + np.cos(np.minimum(delta / spread, 1) * math.pi)) / 2, 0)
    outer = []
    inner = []
    for z, radius in profile:
        t = np.clip((z - start) / (p["height_m"] - start), 0, 1)
        radii = radius + (tip - profile[-1, 1]) * t * t * (3 - 2 * t) * weight
        outer.append(
            np.column_stack((radii * np.cos(angles), radii * np.sin(angles), np.full(segments, z)))
        )
        inner.append(
            np.column_stack(
                (
                    (radii - wall) * np.cos(angles),
                    (radii - wall) * np.sin(angles),
                    np.full(segments, z),
                )
            )
        )
    layers = len(profile)
    vertices = np.array([outer, inner])
    faces, patches, volumes = [], [], []

    def index(surface, level, angle):
        return (surface * layers + level) * segments + angle % segments

    def quad(a, b, c, d):
        faces.extend([(a, b, c), (a, c, d)])

    for level in range(layers - 1):
        for angle in range(segments):
            a, b = index(0, level, angle), index(0, level, angle + 1)
            c, d = index(0, level + 1, angle + 1), index(0, level + 1, angle)
            quad(a, b, c, d)
            a, b = index(1, level, angle), index(1, level + 1, angle)
            c, d = index(1, level + 1, angle + 1), index(1, level, angle + 1)
            quad(a, b, c, d)
            patch = vertices[:, level : level + 2, [angle, (angle + 1) % segments], :].reshape(
                -1, 3
            )
            patches.append(patch)
            volumes.append(ConvexHull(patch).volume)
    for angle in range(segments):
        quad(index(0, 0, angle), index(1, 0, angle), index(1, 0, angle + 1), index(0, 0, angle + 1))
        quad(
            index(0, layers - 1, angle),
            index(0, layers - 1, angle + 1),
            index(1, layers - 1, angle + 1),
            index(1, layers - 1, angle),
        )
    element(
        asset,
        "mesh",
        name="pot_glass_visual",
        vertex=numbers(vertices.ravel()),
        face=" ".join(str(v) for face in faces for v in face),
        smoothnormal="true",
    )
    element(
        pot,
        "geom",
        name="pot_glass_visual",
        type="mesh",
        mesh="pot_glass_visual",
        rgba=glass,
        mass=0,
        contype=0,
        conaffinity=0,
        group=1,
    )
    total_volume = sum(volumes)
    for i, (patch, volume) in enumerate(zip(patches, volumes)):
        name = f"pot_glass_collision_{i}"
        element(asset, "mesh", name=name, vertex=numbers(patch.ravel()))
        element(
            pot,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            group=3,
            mass=p["mass_kg"] * 0.455 * volume / total_volume,
            friction="0.5 0.005 0.0001",
            condim=4,
        )

    # Preserve the outlet hole, but scallop the front edge of the lid away from
    # the pouring channel. A full circular lid would cap the corrected spout.
    lid_radii = lid_radius - (lid_radius - cutback) * weight
    # Photo 102/104/106 shows an annular raised boss, not a flat hole edge.
    # Keep each radial band convex; a single hull would fill the central hole.
    boss = p.get("lid_boss_profile_radius_rise_m", [[p["lid_hole_radius_m"], 0]])
    if (
        len(boss) < 1
        or any(not math.isfinite(v) for point in boss for v in point)
        or abs(boss[0][0] - p["lid_hole_radius_m"]) > 1e-9
        or any(a[0] >= b[0] for a, b in zip(boss, boss[1:]))
        or boss[-1][0] >= min(lid_radii)
        or any(r <= 0 or z < 0 for r, z in boss)
    ):
        raise ValueError("invalid annular lid boss profile")
    lid_patches, lid_volumes = [], []
    for i in range(segments):
        for band in range(len(boss)):
            patch = []
            for thickness_offset in (-0.006, 0):
                for level in (band, band + 1):
                    for j in (i, (i + 1) % segments):
                        radius, rise = boss[level] if level < len(boss) else (lid_radii[j], 0)
                        patch.append(
                            (
                                radius * math.cos(angles[j]),
                                radius * math.sin(angles[j]),
                                p["height_m"] + rise + thickness_offset,
                            )
                        )
            lid_patches.append(patch)
            lid_volumes.append(ConvexHull(patch).volume)
    for i, (patch, volume) in enumerate(zip(lid_patches, lid_volumes)):
        name = f"pot_lid_segment_{i}"
        element(asset, "mesh", name=name, vertex=numbers(np.array(patch).ravel()))
        element(
            pot,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            rgba="0.9 0.88 0.8 1",
            mass=p["mass_kg"] * 0.15 * volume / sum(lid_volumes),
            friction="0.5 0.005 0.0001",
            condim=4,
        )
    element(
        pot,
        "site",
        name="pot_spout",
        pos=numbers([-tip, 0, p["height_m"]]),
        size="0.003",
        rgba="0.1 0.5 1 1",
    )


def pot_handle(asset, pot, p):
    """Photo-derived moulded loop; local +X points away from the pot.

    Keep the visible annulus separate from its convex collision strips: a single
    MuJoCo collision mesh would fill the grasp aperture with its convex hull.
    The historical 14 mm is the OUTER grip thickness, not both uprights.
    """
    import numpy as np
    from scipy.spatial import ConvexHull

    outer = p["handle_outer_to_center_m"]
    width, height, grip, depth = (
        p["handle_outer_width_m"],
        p["handle_outer_height_m"],
        p["handle_thickness_m"],
        p["handle_depth_m"],
    )
    inner = p["handle_inner_thickness_m"]
    rail = p["handle_rail_thickness_m"]
    bottom = p["handle_bottom_m"]
    bevel = p["handle_edge_bevel_m"]
    left_r, right_r = p["handle_root_radius_m"], p["handle_outer_radius_m"]
    hole_r = p["handle_aperture_radius_m"]
    lean = p["handle_bottom_inset_m"]
    values = [
        outer,
        width,
        height,
        grip,
        depth,
        inner,
        rail,
        bottom,
        bevel,
        left_r,
        right_r,
        hole_r,
        lean,
    ]
    if not all(math.isfinite(v) and v > 0 for v in values[:-1]) or not 0 <= lean < width:
        raise ValueError("invalid pot handle dimensions")
    hole_width, hole_height = width - inner - grip, height - 2 * rail
    if (
        hole_width <= 2 * hole_r
        or hole_height <= 2 * hole_r
        or left_r + right_r >= width
        or 2 * max(left_r, right_r) >= height
        or bevel >= min(left_r, right_r, inner / 2, grip / 2, rail / 2, depth / 2)
    ):
        raise ValueError("invalid pot handle aperture/radii/bevel")
    xmin, top = outer - width, bottom + height
    # Eight loops around the cross section: outer edge, two bevels and inner edge.
    layers = [
        (-depth / 2, bevel),
        (-depth / 2 + bevel, 0),
        (depth / 2 - bevel, 0),
        (depth / 2, bevel),
    ]
    loops = []
    for is_hole in (False, True):
        for y, inset in reversed(layers) if is_hole else layers:
            if is_hole:
                profile = rounded_profile(
                    xmin + inner - inset,
                    outer - grip + inset,
                    bottom + rail - inset,
                    top - rail + inset,
                    hole_r + inset,
                    hole_r + inset,
                )
            else:
                profile = rounded_profile(
                    xmin + inset,
                    outer - inset,
                    bottom + inset,
                    top - inset,
                    left_r - inset,
                    right_r - inset,
                )
            loops.append([(x - lean * (top - z) / height, y, z) for x, z in profile])
    vertices = np.array(loops)
    count = vertices.shape[1]
    faces = []
    for layer in range(8):
        for i in range(count):
            a, b = layer * count + i, layer * count + (i + 1) % count
            c, d = ((layer + 1) % 8) * count + i, ((layer + 1) % 8) * count + (i + 1) % count
            faces.extend([(a, c, b), (b, c, d)])
    element(
        asset,
        "mesh",
        name="handle_visual",
        vertex=numbers(vertices.ravel()),
        face=" ".join(str(v) for face in faces for v in face),
    )
    element(
        pot,
        "geom",
        name="handle_visual",
        type="mesh",
        mesh="handle_visual",
        rgba="0.94 0.93 0.86 1",
        group=1,
        contype=0,
        conaffinity=0,
        mass=0,
    )
    strips = [vertices[:, [i, (i + 1) % count], :].reshape(-1, 3) for i in range(count)]
    volumes = [ConvexHull(v).volume for v in strips]
    for i, (v, volume) in enumerate(zip(strips, volumes)):
        name = f"handle_collision_{i}"
        element(asset, "mesh", name=name, vertex=numbers(v.ravel()))
        element(
            pot,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            group=3,
            mass=p["mass_kg"] * 0.16 * volume / sum(volumes),
            friction="0.5 0.005 0.0001",
            condim=4,
        )

    # Neck root and lower strap lug meet the loop and glass instead of leaving
    # the old unsupported lower end. All coordinates are in the pot frame.
    def connector(name, profile, mass, half_depth=depth / 2, rgba="0.94 0.93 0.86 1"):
        coords = [(x, y, z) for y in (-half_depth, half_depth) for x, z in profile]
        element(asset, "mesh", name=name, vertex=numbers(np.array(coords).ravel()))
        element(pot, "geom", name=name, type="mesh", mesh=name, mass=mass, rgba=rgba)

    connector(
        "handle_upper_root",
        [
            (p["radius_m"] * 0.84, top),
            (xmin + left_r, top),
            (xmin + inner, top - rail - 0.004),
            (p["radius_m"] * 0.94, top - rail),
        ],
        p["mass_kg"] * 0.020,
    )
    connector(
        "handle_lower_root",
        [
            (p["radius_m"] * 0.96, bottom - 0.004),
            (xmin + inner - lean, bottom),
            (xmin + inner - lean, bottom + rail / 2),
            (p["radius_m"] * 0.96, bottom + rail / 2),
        ],
        p["mass_kg"] * 0.010,
    )
    band_height = p["handle_band_height_m"]
    band_z = bottom - band_height

    def glass_radius(z):
        return p["radius_m"] * (1 - 0.08 * z / (p["height_m"] - 0.008))

    # The inner web is backed against the glass, not a second free-standing
    # handle. Follow the existing tapered shell; this is a mounting proxy.
    connector(
        "handle_glass_backing",
        [
            (glass_radius(bottom) - 0.001, bottom),
            (xmin - lean + inner / 2, bottom),
            (xmin + inner / 2, top - rail),
            (glass_radius(top - rail) - 0.001, top - rail),
        ],
        p["mass_kg"] * 0.010,
        half_depth=depth / 2 - 0.002,
    )
    shell(
        asset,
        pot,
        "handle_retaining_band",
        [glass_radius(band_z) + 0.0007, glass_radius(bottom) + 0.0007],
        band_height,
        0.0014,
        z=band_z,
        rgba="0.55 0.57 0.59 1",
        mass=p["mass_kg"] * 0.01,
    )
    # Closed-lid hinge/lever proxies, as seen in Pot 104/107/108.
    hinge_x = xmin - 0.003
    for side in (-1, 1):
        geom(
            pot,
            f"handle_hinge_{side}",
            "box",
            [0.005, 0.002, 0.006],
            [hinge_x, side * (depth / 2 - 0.002), top + 0.005],
            rgba="0.94 0.93 0.86 1",
            mass=p["mass_kg"] * 0.0015,
        )
    geom(
        pot,
        "handle_hinge_pin",
        "cylinder",
        [0.0018, depth / 2 + 0.0002],
        [hinge_x, 0, top + 0.006],
        euler=numbers([math.pi / 2, 0, 0]),
        rgba="0.55 0.57 0.59 1",
        mass=0,
        contype=0,
        conaffinity=0,
    )
    connector(
        "handle_thumb_lever",
        [
            (p["radius_m"] * 0.6, top + 0.007),
            (xmin + 0.012, top + 0.004),
            (xmin + 0.010, top + 0.019),
            (p["radius_m"] * 0.6, top + 0.011),
        ],
        p["mass_kg"] * 0.002,
        half_depth=depth / 2 - 0.004,
    )
    # Steel strip wraps the outer grip and its rounded ends (Pot 104/107/108).
    # Visual only: its 0.2 mm offset is below the geometric uncertainty.
    outline = loops[1]
    trim_path = outline[3 * count // 4 :] + outline[: count // 4]
    trim_vertices = [
        (x + 0.0002, y, z)
        for x, _, z in trim_path
        for y in (-depth / 2 + bevel + 0.001, depth / 2 - bevel - 0.001)
    ]
    trim_faces = [(2 * i, 2 * i + 1, 2 * i + 2) for i in range(len(trim_path) - 1)]
    trim_faces += [(2 * i + 1, 2 * i + 3, 2 * i + 2) for i in range(len(trim_path) - 1)]
    element(
        asset,
        "mesh",
        name="handle_grip_trim",
        vertex=numbers(np.array(trim_vertices).ravel()),
        face=" ".join(str(v) for face in trim_faces for v in face),
    )
    element(
        pot,
        "geom",
        name="handle_grip_trim",
        type="mesh",
        mesh="handle_grip_trim",
        rgba="0.5 0.52 0.55 1",
        mass=0,
        contype=0,
        conaffinity=0,
    )
    element(
        pot,
        "site",
        name="pot_handle_grasp",
        pos=numbers([outer - grip / 2 - lean / 2, 0, bottom + height / 2]),
        size="0.003",
        rgba="1 0.4 0 1",
    )


def machine_head(asset, machine, config):
    """Solid head with a front-downward control slope, following photos 109–112."""
    depth, width, height = config["size_m"]
    bottom = config["head_bottom_m"]
    run, drop = config["control_slope_run_m"], config["control_slope_drop_m"]
    margin = config["control_panel_margin_m"]
    length = math.hypot(run, drop)
    if not (
        0 < run < depth
        and 0 < drop < height - bottom
        and 0 < margin < min(length, width) / 2 - 0.010
    ):
        raise ValueError("invalid machine control slope or panel margin")
    # Removing the upper-front corner is essential: a rotated panel over the
    # original box would leave an incorrect rectangular collision silhouette.
    profile = [
        (-depth / 2, bottom),
        (depth / 2, bottom),
        (depth / 2, height),
        (-depth / 2 + run, height),
        (-depth / 2, height - drop),
    ]
    vertices = [v for y in (-width / 2, width / 2) for x, z in profile for v in (x, y, z)]
    element(asset, "mesh", name="machine_head_mesh", vertex=numbers(vertices))
    element(
        machine,
        "geom",
        name="machine_head",
        type="mesh",
        mesh="machine_head_mesh",
        rgba="0.9 0.88 0.8 1",
    )
    mount = element(
        machine,
        "body",
        name="machine_controls_mount",
        pos=numbers([-depth / 2 + run / 2, 0, height - drop / 2]),
        euler=numbers([0, -math.atan2(drop, run), 0]),
    )
    # Local X follows the uphill direction and +Z is the slope's outward normal.
    # Both plates are convex rounded prisms and participate in contact.
    for name, inset, radius, zlo, zhi, color in [
        ("machine_controls_frame", margin, 0.008, 0, 0.001, "0.55 0.57 0.60 1"),
        ("machine_controls", margin + 0.002, 0.006, 0.001, 0.002, "0.035 0.04 0.05 1"),
    ]:
        outline = rounded_profile(
            -length / 2 + inset,
            length / 2 - inset,
            -width / 2 + inset,
            width / 2 - inset,
            radius,
            radius,
            steps=8,
        )
        vertices = [v for z in (zlo, zhi) for x, y in outline for v in (x, y, z)]
        element(asset, "mesh", name=name + "_mesh", vertex=numbers(vertices))
        element(mount, "geom", name=name, type="mesh", mesh=name + "_mesh", rgba=color)


def machine_pot_bay(asset, machine, m, pot):
    """Concave rear housing, open toward -X, with flared/chamfered arc ends.

    The inner circular face is concentric with the nominal pot belly. Each
    collision strip fills only the material *behind* the arc, never its cavity.
    """
    depth, width, _ = m["size_m"]
    cx, cy = m["pot_bay_x_m"], m["pot_bay_y_m"]
    clearance = m["pot_bay_clearance_m"]
    radius = pot["radius_m"] + clearance
    bevel = m["pot_bay_edge_chamfer_m"]
    half_angle = math.radians(m["pot_bay_wrap_deg"] / 2)
    low, high = m["platform_height_m"], m["head_bottom_m"]
    if not (
        clearance > 0 and 0 < bevel < radius / 4 and 0 < half_angle < math.pi / 2 and 0 < low < high
    ):
        raise ValueError("invalid pot bay clearance, arc, chamfer or height")
    core_angle = half_angle - bevel / radius
    if core_angle <= 0:
        raise ValueError("pot bay chamfer consumes arc")
    end_x = cx + (radius + bevel) * math.cos(half_angle)
    end_y = (radius + bevel) * math.sin(half_angle)
    if not (
        -width / 2 < cy - end_y < cy + end_y < width / 2
        and cx + radius < depth / 2
        and end_x < depth / 2
    ):
        raise ValueError("pot bay does not fit inside machine footprint")

    # Ordered front boundary (increasing Y). The tray wall and both chamfer
    # endpoints share one front plane; there is no recessed tray-side step.
    boundary = [
        (end_x, -width / 2),
        (end_x, cy - end_y),
    ]
    for i in range(65):
        angle = -core_angle + 2 * core_angle * i / 64
        boundary.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    boundary += [(end_x, cy + end_y), (end_x, width / 2)]
    vertices = [
        (x, y, z) for front_x, y in boundary for z in (low, high) for x in (front_x, depth / 2)
    ]
    # Per station: front-low, rear-low, front-high, rear-high.
    faces = []

    def quad(a, b, c, d):
        faces.extend([(a, b, c), (a, c, d)])

    for i in range(len(boundary) - 1):
        a, b, c, d = (4 * i + j for j in range(4))
        e, f, g, h = (4 * (i + 1) + j for j in range(4))
        quad(a, c, g, e)  # concave front
        quad(b, f, h, d)  # rear
        quad(c, d, h, g)  # top
        quad(a, e, f, b)  # bottom
        name = f"pot_bay_wall_collision_{i}"
        coords = [v for point in vertices[4 * i : 4 * i + 8] for v in point]
        element(asset, "mesh", name=name, vertex=numbers(coords))
        element(
            machine,
            "geom",
            name=name,
            type="mesh",
            mesh=name,
            group=3,
            friction="0.5 0.005 0.0001",
            condim=4,
        )
    quad(0, 1, 3, 2)
    last = 4 * (len(boundary) - 1)
    quad(last, last + 2, last + 3, last + 1)
    element(
        asset,
        "mesh",
        name="machine_back_mesh",
        vertex=numbers(v for point in vertices for v in point),
        face=" ".join(str(v) for face in faces for v in face),
    )
    element(
        machine,
        "geom",
        name="machine_back",
        type="mesh",
        mesh="machine_back_mesh",
        rgba="0.9 0.88 0.8 1",
        contype=0,
        conaffinity=0,
        mass=0,
        group=1,
    )
    return end_x


def add_scene(root, cfg):
    asset, world = root.find("asset"), root.find("worldbody")
    if asset is None:
        asset = element(root, "asset")
    if world is None:
        world = element(root, "worldbody")
    element(
        world,
        "light",
        name="key",
        pos="0 -1 3",
        dir="0 0 -1",
        diffuse="0.8 0.8 0.8",
        castshadow="true",
    )
    element(world, "light", name="fill", pos="2 2 2", dir="-1 -1 -1", diffuse="0.5 0.5 0.5")
    geom(world, "floor", "plane", [3, 3, 0.1], rgba="0.25 0.28 0.31 1")
    t = cfg["table"]
    tx, ty = t["center_xy_m"]
    top = t["top_height_m"]
    sx, sy, thick = t["size_m"]
    table = element(world, "body", name="table", pos=numbers([tx, ty, top]))
    geom(
        table,
        "tabletop",
        "box",
        [sx / 2, sy / 2, thick / 2],
        [0, 0, -thick / 2],
        rgba="0.83 0.83 0.85 1",
    )
    for i, (x, y) in enumerate(
        [(a * (sx / 2 - 0.04), b * (sy / 2 - 0.04)) for a in [-1, 1] for b in [-1, 1]]
    ):
        geom(
            table,
            f"table_leg{i}",
            "box",
            [0.018, 0.018, (top - thick) / 2],
            [x, y, -(top + thick) / 2],
            rgba="0.12 0.13 0.14 1",
        )

    m = cfg["machine"]
    mx, my = m["center_table_xy_m"]
    depth, width, height = m["size_m"]
    machine = element(table, "body", name="coffee_machine", pos=numbers([mx, my, 0]))
    platform = m["platform_height_m"]
    geom(
        machine,
        "machine_platform",
        "box",
        [depth / 2, width / 2, platform / 2],
        [0, 0, platform / 2],
    )
    back_front_x = machine_pot_bay(asset, machine, m, cfg["pot"])
    machine_head(asset, machine, m)
    bx, by = m["pot_bay_x_m"], m["pot_bay_y_m"]
    hotplate_radius = float(m.get("hotplate_radius_m", 0.074))
    if not math.isfinite(hotplate_radius) or not 0 < hotplate_radius < min(m["size_m"][:2]) / 2:
        raise ValueError("hotplate radius must fit within the machine platform")
    geom(
        machine,
        "hotplate",
        "cylinder",
        [hotplate_radius, 0.001],
        [bx, by, platform],
        rgba="0.08 0.08 0.09 1",
        solref="0.005 1",
        solimp="0.999 0.999 0.001",
        solmix="1000",
    )
    if "hotplate_recess" in m:
        cfg["platform_geometry"] = install_recessed_platform(asset, machine, m, cfg["pot"])
    # Trim the tray and its slot decoration at the now-flush wall rather than
    # leaving their rear sections buried inside the machine housing.
    tray_front, tray_back = bx - 0.065, min(bx + 0.065, back_front_x)
    geom(
        machine,
        "drip_tray",
        "box",
        [(tray_back - tray_front) / 2, 0.055, 0.002],
        [(tray_front + tray_back) / 2, -0.077, platform],
        rgba="0.5 0.52 0.55 1",
    )
    slot_front, slot_back = bx - 0.05, min(bx + 0.05, back_front_x - 0.001)
    for i in range(7):
        geom(
            machine,
            f"tray_slot{i}",
            "box",
            [(slot_back - slot_front) / 2, 0.0015, 0.0005],
            [(slot_front + slot_back) / 2, -0.12 + i * 0.014, platform + 0.0025],
            rgba="0.1 0.1 0.1 1",
            contype=0,
            conaffinity=0,
        )

    p = cfg["pot"]
    # A single cylinder/cylinder contact needs torsional friction to represent
    # resistance to spinning over a finite support patch. Keep legacy scenes
    # unchanged; candidate scenes may explicitly activate the existing spin
    # coefficient with condim=4. This coefficient is not hardware-calibrated.
    support_condim = p.get("support_contact_condim", 3)
    if type(support_condim) is not int or support_condim not in (3, 4):
        raise ValueError("pot support_contact_condim must be 3 or 4")
    seat_z = nominal_seated_height(m, p)
    pot_z = top + seat_z + 0.001
    pot = element(
        world,
        "body",
        name="pot",
        pos=numbers([tx + mx + bx, ty + my + by, pot_z]),
        euler=numbers([0, 0, p["yaw_rad"]]),
    )
    element(pot, "freejoint", name="pot_free")
    glass = "0.65 0.85 0.91 0.35"
    geom(
        pot,
        "pot_bottom",
        "cylinder",
        [p["radius_m"], p["wall_m"] / 2],
        [0, 0, p["wall_m"] / 2],
        rgba=glass,
        mass=p["mass_kg"] * 0.18,
        condim=support_condim,
        # Keep support penetration small compared with millimetre-scale relief.
        solref="0.005 1",
        solimp="0.999 0.999 0.001",
        solmix="1000",
    )
    pot_vessel(asset, pot, p, glass)
    pot_handle(asset, pot, p)

    o = cfg["outlet"]
    if "conical_hypothesis" in o:
        from .outlet import install_conical_outlet

        outlet = element(machine, "body", name="spring_outlet", pos=numbers([bx, by, 0]))
        cfg["outlet_spring_trial"] = install_conical_outlet(
            asset,
            outlet,
            cfg,
            lid_initial_z_m=seat_z
            + 0.001
            + p["height_m"]
            + p["lid_boss_profile_radius_rise_m"][0][1],
        )
    else:
        engagement = o.get("tip_engagement_below_lid_m")
        if engagement is None:
            outlet_z = seat_z + 0.001 + p["height_m"] - 0.003
        else:
            if not math.isfinite(engagement) or not 0 < engagement < o["tip_radius_m"]:
                raise ValueError("outlet engagement must remain on the rounded nose")
            lid_rise = max(z for _, z in p.get("lid_boss_profile_radius_rise_m", [[0, 0]]))
            # The lower capsule hemisphere cams upward over the lid. Its cylinder
            # must not enter the hole and form an unintended rigid lateral lock.
            outlet_z = seat_z + p["height_m"] + lid_rise + o["tip_radius_m"] - engagement
        outlet = element(machine, "body", name="spring_outlet", pos=numbers([bx, by, outlet_z]))
        element(
            outlet,
            "joint",
            name="outlet_compression",
            type="slide",
            axis="0 0 1",
            range=numbers([0, o["travel_m"]]),
            stiffness=o["spring_n_per_m"],
            damping="0.2",
            springref=-o["preload_n"] / o["spring_n_per_m"],
        )
        geom(
            outlet,
            "outlet_tip",
            "capsule",
            [o["tip_radius_m"], 0.008],
            [0, 0, 0.008],
            rgba="0.15 0.15 0.16 1",
            mass=0.02,
        )
    d = cfg["dispenser"]
    dx, dy = d["center_table_xy_m"]
    dispenser = element(table, "body", name="cup_dispenser", pos=numbers([dx, dy, 0]))
    dd, dw, dh = d["size_m"]
    geom(dispenser, "dispenser_base", "box", [dd / 2, dw / 2, 0.006], [0, 0, 0.006])
    geom(
        dispenser,
        "dispenser_back",
        "box",
        [0.012, dw / 2, d["ring_height_m"] / 2],
        [dd / 2 - 0.012, 0, d["ring_height_m"] / 2],
    )
    ring_z, rr = d["ring_height_m"], d["ring_radius_m"]
    shell(asset, dispenser, "dispenser_ring", [rr + 0.008, rr + 0.008], 0.015, 0.008, z=ring_z)
    shell(
        asset,
        dispenser,
        "dispenser_tube",
        [rr + 0.002, rr + 0.002],
        dh - ring_z - 0.015,
        0.0015,
        z=ring_z + 0.015,
        rgba="0.7 0.83 0.9 0.18",
    )
    latch_count = d.get("latch_count", 3)
    if type(latch_count) is not int or not 3 <= latch_count <= 8:
        raise ValueError("invalid dispenser latch count")
    if d.get("retention_mechanism") == "elastic_rim_tabs":
        # Reduced-order plastic flexure: each tab bends down independently.
        # A passive torsional spring models elastic bending; no actuator or weld.
        tab = d["elastic_tabs"]
        length, nose, width = tab["length_m"], tab["nose_radius_m"], tab["width_m"]
        roll = cfg["cup"].get("rolled_rim_radius_m", 0)
        wall_top = cfg["cup"]["rim_radius_m"] - roll
        inner = rr - length - nose
        if not (
            roll > 0
            and wall_top < inner < cfg["cup"]["rim_radius_m"]
            and 0 < tab["max_bend_rad"] < math.pi / 2
            and 0 < tab["thickness_m"] < nose
            and tab["stiffness_nm_per_rad"] > 0
            and tab["damping_nms_per_rad"] > 0
        ):
            raise ValueError("tabs must clear the cup wall and engage its rolled rim")
        min_pitch = (
            max(cfg["cup"]["wall_m"], roll)
            * (cfg["cup"]["height_m"] - roll)
            / (wall_top - cfg["cup"]["bottom_radius_m"])
        )
        if d["stack_pitch_m"] < min_pitch + 0.0002:
            raise ValueError("stack pitch interpenetrates the rigid tapered cup walls")
        tab_z = ring_z + tab["root_height_above_ring_m"]
        for i in range(latch_count):
            angle = 2 * math.pi * i / latch_count
            finger = element(
                dispenser,
                "body",
                name=f"retention_finger{i}",
                pos=numbers([rr * math.cos(angle), rr * math.sin(angle), tab_z]),
                euler=numbers([0, 0, angle]),
            )
            element(
                finger,
                "joint",
                name=f"finger_deflection{i}",
                type="hinge",
                axis="0 1 0",
                range=numbers([-tab["max_bend_rad"], 0]),
                springref="0",
                stiffness=tab["stiffness_nm_per_rad"],
                damping=tab["damping_nms_per_rad"],
                solreflimit="0.004 1",
                solimplimit="0.999 0.999 0.001",
            )
            common = dict(
                friction=f"{d['finger_friction']} 0.005 0.0001",
                condim="4",
                solref="0.005 1",
                solimp="0.999 0.999 0.001",
                solmix="1000",
            )
            geom(
                finger,
                f"retention_tab{i}",
                "box",
                [length / 2, width / 2, tab["thickness_m"] / 2],
                [-length / 2, 0, 0],
                mass=tab["mass_kg"] * 0.7,
                rgba="0.3 0.3 0.33 1",
                **common,
            )
            element(
                finger,
                "geom",
                name=f"retention_pad{i}",
                type="capsule",
                size=str(nose),
                fromto=numbers([-length, -width / 2, 0, -length, width / 2, 0]),
                mass=tab["mass_kg"] * 0.3,
                rgba="0.35 0.35 0.38 1",
                **common,
            )
        # Start just above the four noses. Gravity establishes support at the lip.
        first_rim_top = tab_z + nose + 2 * roll + 0.00002
    else:
        # Seed contact at the outer wall, never through the 1 mm thin shell.
        # A matching joint ref preserves the original zero-deflection rest geometry.
        contact_radius = (
            cfg["cup"]["rim_radius_m"]
            - (cfg["cup"]["rim_radius_m"] - cfg["cup"]["bottom_radius_m"])
            * 0.001
            / cfg["cup"]["height_m"]
        )
        pad_shape = d.get("retention_pad_shape", "box")
        if pad_shape not in ("box", "sphere"):
            raise ValueError("unknown retention pad shape")
        if pad_shape == "sphere":
            slope = (cfg["cup"]["rim_radius_m"] - cfg["cup"]["bottom_radius_m"]) / cfg["cup"][
                "height_m"
            ]
            # Same 33 mm inner rest radius and 8 mm height as the box. Rounded
            # contact avoids spurious normals at the shell's convex-wedge seams.
            contact_radius = (
                cfg["cup"]["rim_radius_m"]
                - slope * 0.005
                + 0.004 * (math.sqrt(1 + slope * slope) - 1)
            )
        initial_deflection = max(
            0.0, contact_radius - (rr - 0.008) + d.get("initial_finger_clearance_m", -0.00002)
        )
        if not math.isfinite(initial_deflection) or not 0 <= initial_deflection < 0.012:
            raise ValueError("invalid rack finger initial opening")
        for i in range(latch_count):
            a = i * 2 * math.pi / latch_count
            finger = element(
                dispenser,
                "body",
                name=f"retention_finger{i}",
                pos=numbers(
                    [
                        (rr + initial_deflection) * math.cos(a),
                        (rr + initial_deflection) * math.sin(a),
                        ring_z + 0.004,
                    ]
                ),
                euler=numbers([0, 0, a]),
            )
            element(
                finger,
                "joint",
                name=f"finger_deflection{i}",
                type="slide",
                axis="1 0 0",
                range="0 0.012",
                ref=initial_deflection,
                springref=0,
                stiffness=d["finger_stiffness_n_per_m"],
                damping=d.get("finger_damping_n_s_per_m", 0.1),
            )
            geom(
                finger,
                f"retention_pad{i}",
                pad_shape,
                [0.004] if pad_shape == "sphere" else [0.008, 0.004, 0.004],
                [-0.004, 0, 0] if pad_shape == "sphere" else [0, 0, 0],
                mass=0.002,
                rgba="0.3 0.3 0.33 1",
                friction=f"{d['finger_friction']} 0.005 0.0001",
                condim=4,
                solref="0.005 1",
                solimp="0.999 0.999 0.001",
                solmix="1000",
            )
        first_rim_top = ring_z + 0.009
    for i in range(d["stack_count"]):
        cup(
            asset,
            world,
            cfg,
            f"cup_{i}",
            [
                tx + dx,
                ty + dy,
                top + first_rim_top - cfg["cup"]["height_m"] + i * d["stack_pitch_m"],
            ],
        )
    place = cfg["task"]["cup_place_table_xyz_m"]
    element(
        table,
        "site",
        name="cup_placement_target",
        type="cylinder",
        size="0.045 0.001",
        pos=numbers(place),
        rgba="0.1 0.7 0.3 0.3",
    )
    # MuJoCo cameras look along local -Z with local +Y up.
    element(
        world,
        "camera",
        name="overview",
        pos="-0.5 -1.8 1.9",
        xyaxes="0.86 -0.5 0 0.25 0.43 0.87",
        fovy="50",
    )
    element(
        world,
        "camera",
        name="front",
        pos=numbers([tx - 0.55, 0, top + 0.25]),
        xyaxes="0 -1 0 0 0 1",
        fovy="65",
    )
    element(
        world,
        "camera",
        name="side",
        pos=numbers([tx + mx, ty + my + 0.85, top + height / 2]),
        # From world +Y, look toward -Y. The old +X right axis looked away.
        xyaxes="-1 0 0 0 0 1",
        fovy="42",
    )
    element(
        world,
        "camera",
        name="machine_detail",
        pos=numbers([tx + mx - 0.65, ty + my + 0.62, top + height + 0.17]),
        xyaxes="-0.690 -0.724 0 0.250 -0.239 0.938",
        fovy="45",
    )


def build_model(urdf: Path, scene_path: Path, output: Path) -> dict:
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    urdf, scene_path, output = urdf.resolve(), scene_path.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(scene_path.read_text())
    from .estimates import estimate_parameters

    estimates = estimate_parameters(cfg)
    if "conical_hypothesis" in cfg["outlet"]:
        hypothesis = cfg["outlet"]["conical_hypothesis"]
        cfg["outlet"]["spring_n_per_m"] = hypothesis["stiffness_n_per_m"]
        cfg["outlet"]["preload_n"] = hypothesis["preload_at_zero_n"]
        estimates["legacy_weight_based_spring_prior_superseded"] = True
        estimates["spring_stiffness_n_per_m"] = hypothesis["stiffness_n_per_m"]
        estimates["spring_reference_force_n"] = (
            hypothesis["preload_at_zero_n"]
            + hypothesis["stiffness_n_per_m"] * estimates["spring_reference_compression_m"]
        )
        estimates["spring_force_source"] = "explicit unmeasured conical outlet hypothesis"
    else:
        cfg["outlet"]["spring_n_per_m"] = estimates["spring_stiffness_n_per_m"]
    (output / "parameter_estimates.json").write_text(json.dumps(estimates, indent=2))
    (output / "scene_parameters.json").write_text(json.dumps(cfg, indent=2))
    tree = ET.parse(urdf)
    source = tree.getroot()
    mesh_dir = urdf.parent.parent / "meshes"
    for link in source.findall("link"):
        for tag in ("visual", "collision"):
            for i, item in enumerate(link.findall(tag)):
                item.set("name", f"{link.get('name')}_{tag}_{i}")
    mesh_manifest = {}
    for mesh in source.findall(".//mesh"):
        filename = mesh_dir / Path(mesh.get("filename")).name
        if not filename.is_file():
            raise FileNotFoundError(filename)
        mesh.set("filename", str(filename))
        if filename.name not in mesh_manifest:
            mesh_manifest[filename.name] = hashlib.sha256(filename.read_bytes()).hexdigest()
    import_opts = element(source, "mujoco")
    element(
        import_opts,
        "compiler",
        discardvisual="false",
        fusestatic="false",
        balanceinertia="true",
        strippath="false",
        meshdir=str(mesh_dir),
    )
    prepared = output / "resolved_robot.urdf"
    tree.write(prepared, encoding="utf-8", xml_declaration=True)
    imported = mujoco.MjModel.from_xml_path(str(prepared))
    converted = output / "robot_import.xml"
    mujoco.mj_saveLastXML(str(converted), imported)
    root = ET.parse(converted).getroot()
    root.set("model", "realbot01_coffee_prototype")
    root.find("compiler").set("autolimits", "true")
    root.find("compiler").set("boundmass", "0.0001")
    physics = cfg.get("contact_solver", {})
    timestep = float(physics.get("timestep_s", 0.002))
    if not math.isfinite(timestep) or not 0.000125 <= timestep <= 0.002:
        raise ValueError("simulation timestep must be in [0.000125, 0.002]")
    impratio = float(physics.get("friction_impedance_ratio", 1))
    if not math.isfinite(impratio) or not 1 <= impratio <= 100:
        raise ValueError("friction impedance ratio must be in [1, 100]")
    multiccd = physics.get("multiccd", False)
    if not isinstance(multiccd, bool):
        raise ValueError("multiccd must be a boolean")
    option = element(
        root,
        "option",
        timestep=timestep,
        integrator="implicitfast",
        cone="elliptic",
        impratio=impratio,
        solver="Newton",
        tolerance="1e-10",
    )
    if multiccd:
        element(option, "flag", multiccd="enable")
    if "ccd_tolerance" in physics:
        ccd_tolerance = float(physics["ccd_tolerance"])
        if not math.isfinite(ccd_tolerance) or not 1e-10 <= ccd_tolerance <= 1e-6:
            raise ValueError("CCD tolerance must be in [1e-10, 1e-6]")
        option.set("ccd_tolerance", str(ccd_tolerance))
    visual = element(root, "visual")
    element(visual, "global", offwidth="1280", offheight="960")
    equality, actuator = element(root, "equality"), element(root, "actuator")
    arm_joints = {f"{side}_joint_{i}" for side in ("l", "r") for i in range(1, 8)}
    frozen = []
    frozen_values = cfg["robot"].get("frozen_joints_rad", {})
    for body in root.findall(".//body"):
        body.set("gravcomp", "1")
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name in arm_joints:
                joint.set("damping", "2")
                joint.set("armature", "0.03")
                element(
                    actuator,
                    "position",
                    name=name + "_servo",
                    joint=name,
                    kp="200",
                    kv="25",
                    ctrlrange=joint.get("range"),
                    forcerange=joint.get("actuatorfrcrange", "-20 20"),
                )
            elif "tool_" in name:
                joint.set("damping", "0.03")
                joint.set("armature", "0.0001")
            else:
                # Base, torso, head, wheel and adjustable mounts are fixed at URDF zero.
                # This is a simulation choice, never a live zero-position assumption.
                frozen.append(name)
                angle = float(frozen_values.get(name, 0))
                limits = joint.get("range")
                if not math.isfinite(angle) or (
                    limits and not float(limits.split()[0]) <= angle <= float(limits.split()[1])
                ):
                    raise ValueError(f"invalid frozen URDF joint angle: {name}")
                if angle:
                    axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ")
                    delta = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle)
                    q = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
                    original = Rotation.from_quat(q[[1, 2, 3, 0]])
                    pivot = np.fromstring(joint.get("pos", "0 0 0"), sep=" ")
                    pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
                    body.set("pos", numbers(pos + original.apply(pivot - delta.apply(pivot))))
                    q = (original * delta).as_quat()
                    body.set("quat", numbers(q[[3, 0, 1, 2]]))
                body.remove(joint)
    unknown = set(frozen_values) - set(frozen)
    if unknown:
        raise ValueError(f"frozen_joints_rad includes unknown/non-frozen joints: {sorted(unknown)}")
    for j in source.findall("joint"):
        mimic = j.find("mimic")
        if mimic is not None:
            element(
                equality,
                "joint",
                name=j.get("name") + "_mimic",
                joint1=j.get("name"),
                joint2=mimic.get("joint"),
                polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0",
                # URDF mimic is a kinematic linkage, not a flexible finger.
                # Default soft equalities let the opposing jaw drift >10 deg
                # under handle load and invalidate the grasp-frame geometry.
                solref="0.004 1",
                solimp="0.999 0.999 0.001",
            )
    for side, arm in [("l", "left"), ("r", "right")]:
        driver = side + "tool_Left_1_Joint"
        servo = cfg.get("gripper_servos", {}).get(arm, {})
        kp, kv, effort = (
            float(servo.get(k, v))
            for k, v in [("kp_nm_per_rad", 4), ("kv_nms_per_rad", 0.2), ("effort_limit_nm", 1)]
        )
        if not all(math.isfinite(x) and x > 0 for x in (kp, kv, effort)):
            raise ValueError("invalid gripper servo parameters")
        element(
            actuator,
            "position",
            name=arm + "_gripper_servo",
            joint=driver,
            kp=kp,
            kv=kv,
            ctrlrange="-0.91 0",
            forcerange=numbers([-effort, effort]),
        )
        body = root.find(f".//body[@name='{side}_link_8']")
        element(body, "site", name=arm + "_tool_mount", size="0.004", rgba="1 0.2 0.2 1")
        pad = cfg.get("gripper_contact_proxy")
        if pad is not None:
            contact_time = float(pad.get("contact_time_constant_s", 0.02))
            contact_impedance = float(pad.get("contact_impedance", 0.99))
            if not math.isfinite(contact_impedance) or not 0.9 <= contact_impedance <= 0.9999:
                raise ValueError("pad contact impedance must be in [0.9, 0.9999]")
            contact_solimp = numbers([contact_impedance, contact_impedance, 0.001])
            contact_solmix = float(pad.get("contact_solmix", 1))
            if not math.isfinite(contact_solmix) or not 1 <= contact_solmix <= 10000:
                raise ValueError("pad contact solmix must be in [1, 10000]")
            if not math.isfinite(contact_time) or not 0.004 <= contact_time <= 0.02:
                raise ValueError("pad contact time must be in [0.004, 0.02] seconds")
            values = [pad[k] for k in ("thickness_m", "length_m", "width_m", "mass_kg", "friction")]
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError("invalid gripper contact proxy")
            for jaw in ("Left", "Right"):
                support = root.find(f".//body[@name='{side}tool_{jaw}_Support_Link']")
                if support is not None:
                    for support_geom in support.findall("geom"):
                        support_geom.set("solref", numbers([contact_time, 1]))
                        support_geom.set("solimp", contact_solimp)
                        support_geom.set("solmix", str(contact_solmix))
                body = root.find(f".//body[@name='{side}tool_{jaw}_Pad_Link']")
                if body is None:
                    raise ValueError("URDF missing pad frame")
                geom(
                    body,
                    f"{arm}_{jaw.lower()}_pad_contact",
                    "box",
                    [pad["thickness_m"] / 2, pad["length_m"] / 2, pad["width_m"] / 2],
                    mass=pad["mass_kg"],
                    friction=f"{pad['friction']} 0.005 0.0001",
                    condim="4",
                    solref=numbers([contact_time, 1]),
                    solimp=contact_solimp,
                    solmix=contact_solmix,
                    rgba="0.15 0.16 0.18 1",
                )
                # Both pad frames have +X pointing outward, away from the gap.
                element(
                    body,
                    "site",
                    name=f"{arm}_{jaw.lower()}_pad_surface",
                    pos=numbers([-pad["thickness_m"] / 2, 0, 0]),
                    size="0.002",
                    rgba="0.2 1 0.3 1",
                )
    base = root.find(".//body[@name='base_link']")
    base.set("pos", numbers([*cfg["robot"]["base_xy_m"], 0]))
    base.set(
        "quat",
        numbers(
            [math.cos(cfg["robot"]["yaw_rad"] / 2), 0, 0, math.sin(cfg["robot"]["yaw_rad"] / 2)]
        ),
    )
    add_scene(root, cfg)
    (output / "scene_parameters.json").write_text(json.dumps(cfg, indent=2))
    # Connected mechanism parts overlap by construction. Exclude only these
    # documented assembly neighbours, retaining robot/environment contacts.
    contact = element(root, "contact")
    exclusions = [("xb_link", f"{s}_link_1") for s in ("l", "r")]
    for s in ("l", "r"):
        exclusions += [
            (f"{s}tool_Left_Support_Link", f"{s}tool_Left_2_Link"),
            (f"{s}tool_Right_Support_Link", f"{s}tool_Right_2_Link"),
        ]
    exclusions += [("coffee_machine", "spring_outlet")]
    exclusions += [
        ("cup_dispenser", f"retention_finger{i}")
        for i in range(cfg["dispenser"].get("latch_count", 3))
    ]
    for a, b in exclusions:
        element(contact, "exclude", body1=a, body2=b)
    path = output / "scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    from .posture import upper_body_posture

    posture = upper_body_posture(
        model, maximum_tilt_deg=cfg["robot"].get("max_upper_body_tilt_deg")
    )
    report = {
        "scene": str(path),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "scene_config_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "source_meshes_sha256": mesh_manifest,
        "bodies": model.nbody,
        "joints": model.njnt,
        "geoms": model.ngeom,
        "actuators": model.nu,
        "frozen_urdf_joints": frozen,
        "frozen_urdf_positions_rad": {name: float(frozen_values.get(name, 0)) for name in frozen},
        "upper_body_posture": posture,
        "mimic_constraints": len(equality),
        "qualification": cfg["qualification"],
        "assembly_contact_exclusions": exclusions,
        "fluid_simulation": False,
        "real_motion": False,
        "gripper_contact_proxy": cfg.get("gripper_contact_proxy"),
        "limitations": [
            "URDF controller zero/sign mapping unverified",
            "torso and base frozen in this fixture",
            "tool sites are mounting frames, not calibrated grasp centers",
            "unmeasured masses/friction/springs",
            "convex robot mesh contacts plus source primitive contacts",
            "no thermal or fluid model",
        ],
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2))
    return report
