"""Recessed heating disk with an explicitly approximated rounded ramp."""

import math
import xml.etree.ElementTree as ET


def is_machine_support_geom(name):
    return name in {"hotplate", "machine_platform"} or name.startswith(
        ("machine_platform_outer_", "machine_platform_corner_", "machine_platform_ramp_")
    )


def profile_parameters(machine):
    p = machine["hotplate_recess"]
    depth, width = float(p["depth_m"]), float(p["ramp_width_m"])
    radius, height = float(machine["hotplate_radius_m"]), float(machine["platform_height_m"])
    segments, bands = p.get("segments", 64), p.get("radial_bands", 8)
    if not all(math.isfinite(x) for x in (depth, width, radius, height)):
        raise ValueError("platform dimensions must be finite")
    if not 0 < depth < height - 0.002 or not 0.003 <= width <= 0.03 or radius <= 0:
        raise ValueError("invalid recess or rounded-ramp dimensions")
    if type(segments) is not int or segments % 8 or not 32 <= segments <= 128:
        raise ValueError("need 32..128 angular segments divisible by eight")
    if type(bands) is not int or not 4 <= bands <= 32:
        raise ValueError("need 4..32 radial bands")
    if p.get("profile", "half_cosine") != "half_cosine":
        raise ValueError("unsupported rounded ramp profile")
    bx, by = machine["pot_bay_x_m"], machine["pot_bay_y_m"]
    sx, sy = machine["size_m"][:2]
    if abs(bx) + radius + width >= sx / 2 or abs(by) + radius + width >= sy / 2:
        raise ValueError("recess and ramp must fit inside the platform")
    return depth, width, radius, height, segments, bands


def support_height(machine, radial_distance):
    """Surface height above machine base; legacy raised scenes remain explicit."""
    if "hotplate_recess" not in machine:
        return machine["platform_height_m"] + 0.001
    depth, width, radius, height, _, _ = profile_parameters(machine)
    t = max(0.0, min(1.0, (radial_distance - radius) / width))
    return height - depth * (1 + math.cos(math.pi * t)) / 2


def nominal_seated_height(machine, pot):
    # Current pot floor is a flat cylinder. Do not silently shrink it to fit the disk.
    return support_height(machine, float(pot["radius_m"]))


def install_recessed_platform(asset, body, machine, pot):
    depth, width, radius, height, segments, bands = profile_parameters(machine)
    if (
        machine["hotplate_recess"].get("disk_at_least_pot_floor", False)
        and radius < pot["radius_m"]
    ):
        raise ValueError("operator-confirmed disk width must not be smaller than the pot floor")
    bx, by = machine["pot_bay_x_m"], machine["pot_bay_y_m"]
    sx, sy = machine["size_m"][:2]
    outer, floor = radius + width, height - depth - 0.002
    old_platform = body.find("geom[@name='machine_platform']")
    old_disk = body.find("geom[@name='hotplate']")
    if old_platform is None or old_disk is None:
        raise ValueError("expected existing platform and hotplate to replace")
    attrs = {
        k: v for k, v in old_platform.attrib.items() if k not in {"name", "type", "size", "pos"}
    }
    disk_attrs = dict(old_disk.attrib)
    disk_attrs.pop("mesh", None)
    disk_attrs["type"] = "cylinder"
    disk_attrs["rgba"] = "0.08 0.08 0.09 1"
    for node in list(body):
        if node.tag == "geom" and (
            is_machine_support_geom(node.get("name", "")) or node.get("name") == "hotplate_visual"
        ):
            body.remove(node)
    for node in list(asset):
        if node.tag == "mesh" and node.get("name", "").startswith("machine_platform_"):
            asset.remove(node)
    numbers = lambda vs: " ".join(format(float(v), ".12g") for v in vs)

    def box(name, bounds, lo, hi):
        x0, x1, y0, y1 = bounds
        ET.SubElement(
            body,
            "geom",
            name=name,
            type="box",
            size=numbers([(x1 - x0) / 2, (y1 - y0) / 2, (hi - lo) / 2]),
            pos=numbers([(x1 + x0) / 2, (y1 + y0) / 2, (hi + lo) / 2]),
            **attrs,
        )

    box("machine_platform", [-sx / 2, sx / 2, -sy / 2, sy / 2], 0, floor)
    bounds = [
        (-sx / 2, bx - outer, -sy / 2, sy / 2),
        (bx + outer, sx / 2, -sy / 2, sy / 2),
        (bx - outer, bx + outer, -sy / 2, by - outer),
        (bx - outer, bx + outer, by + outer, sy / 2),
    ]
    for i, bound in enumerate(bounds):
        box(f"machine_platform_outer_{i}", bound, floor, height)

    def patch(name, rs0, rs1, z0, z1, angles):
        xy = [
            (bx + r * math.cos(a), by + r * math.sin(a))
            for radii in (rs0, rs1)
            for r, a in zip(radii, angles)
        ]
        vertices = [v for z in (floor,) for x, y in xy for v in (x, y, z)]
        vertices += [v for (x, y), z in zip(xy, [z0, z0, z1, z1]) for v in (x, y, z)]
        ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices))
        ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, **attrs)

    for i in range(segments):
        angles = [2 * math.pi * i / segments, 2 * math.pi * (i + 1) / segments]
        square = [outer / max(abs(math.cos(a)), abs(math.sin(a))) for a in angles]
        patch(f"machine_platform_corner_{i:03d}", [outer] * 2, square, height, height, angles)
        for j in range(bands):
            ra, rb = radius + width * j / bands, radius + width * (j + 1) / bands
            patch(
                f"machine_platform_ramp_{j:02d}_{i:03d}",
                [ra] * 2,
                [rb] * 2,
                support_height(machine, ra),
                support_height(machine, rb),
                angles,
            )
    disk_attrs.update(size=numbers([radius, 0.001]), pos=numbers([bx, by, height - depth - 0.001]))
    # Finite convex faces avoid the native cylinder/cylinder EPA degeneracy at
    # nearly parallel flat surfaces. The visual disk remains an exact cylinder.
    visual = dict(disk_attrs, name="hotplate_visual", contype="0", conaffinity="0", group="1")
    ET.SubElement(body, "geom", **visual)
    disk_vertices = [
        v
        for z in [-0.001, 0.001]
        for i in range(segments)
        for v in (
            radius * math.cos(2 * math.pi * i / segments),
            radius * math.sin(2 * math.pi * i / segments),
            z,
        )
    ]
    ET.SubElement(asset, "mesh", name="machine_platform_disk", vertex=numbers(disk_vertices))
    disk_attrs.pop("size")
    disk_attrs.update(type="mesh", mesh="machine_platform_disk", rgba="0.08 0.08 0.09 0")
    ET.SubElement(body, "geom", **disk_attrs)
    return {
        "disk_below_platform_m": depth,
        "disk_top_z_m": height - depth,
        "platform_top_z_m": height,
        "nominal_pot_seat_z_m": nominal_seated_height(machine, pot),
        "ramp_width_m": width,
        "ramp_profile": "piecewise_convex_half_cosine",
        "ramp_width_and_curvature_measured": False,
        "pot_floor_radius_unchanged_m": pot["radius_m"],
        "segments": segments,
        "radial_bands": bands,
        "disk_collision": "convex_polygonal_prism",
        "disk_radial_polygon_error_m": radius * (1 - math.cos(math.pi / segments)),
    }


def install_convex_pot_floor(asset, pot_body, *, segments=128):
    """Replace a loaded pot's cylinder floor collision with a finite-faced prism.

    Explicit body inertia is required so faceting cannot change the workload.
    The exact cylinder is retained for rendering; all contact parameters survive.
    """
    if type(segments) is not int or not 64 <= segments <= 256 or segments % 8:
        raise ValueError("floor needs 64..256 segments divisible by eight")
    if pot_body.find("inertial") is None:
        raise ValueError("explicit pot inertia is required to preserve the payload")
    floor = pot_body.find("geom[@name='pot_bottom']")
    if floor is None or floor.get("type") != "cylinder":
        raise ValueError("expected the original cylinder pot floor")
    radius, half_height = map(float, floor.get("size").split())
    if not all(math.isfinite(x) and x > 0 for x in (radius, half_height)):
        raise ValueError("invalid floor dimensions")
    name = "pot_bottom_convex_proxy"
    if asset.find(f"mesh[@name='{name}']") is not None:
        raise ValueError("floor proxy already exists")
    visual = dict(
        floor.attrib, name="pot_bottom_visual", contype="0", conaffinity="0", mass="0", group="1"
    )
    visual.pop("density", None)
    ET.SubElement(pot_body, "geom", **visual)
    vertices = [
        v
        for z in [-half_height, half_height]
        for i in range(segments)
        for v in (
            radius * math.cos(2 * math.pi * i / segments),
            radius * math.sin(2 * math.pi * i / segments),
            z,
        )
    ]
    ET.SubElement(asset, "mesh", name=name, vertex=" ".join(format(v, ".12g") for v in vertices))
    floor.set("type", "mesh")
    floor.set("mesh", name)
    floor.attrib.pop("size")
    floor.set("rgba", "0 0 0 0")
    return {
        "shape": "convex_polygonal_prism",
        "segments": segments,
        "nominal_radius_m": radius,
        "half_height_m": half_height,
        "maximum_radial_error_m": radius * (1 - math.cos(math.pi / segments)),
        "explicit_body_inertia_preserved": True,
    }
