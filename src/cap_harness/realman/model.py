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
        [c["bottom_radius_m"], c["rim_radius_m"]],
        c["height_m"],
        c["wall_m"],
        mass=c["mass_kg"] * 0.8,
    )
    element(
        body,
        "site",
        name=name + "_rim",
        pos=numbers([0, 0, c["height_m"]]),
        size="0.002",
        rgba="0 0.6 0 1",
    )
    return body


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
    geom(
        machine,
        "machine_back",
        "box",
        [0.055, width / 2, height / 2],
        [depth / 2 - 0.055, 0, height / 2],
    )
    head_bottom = m["head_bottom_m"]
    geom(
        machine,
        "machine_head",
        "box",
        [depth / 2, width / 2, (height - head_bottom) / 2],
        [0, 0, (height + head_bottom) / 2],
    )
    geom(
        machine,
        "machine_controls",
        "box",
        [0.006, width * 0.46, 0.025],
        [-depth / 2 - 0.001, 0, height - 0.045],
        rgba="0.09 0.1 0.12 1",
    )
    bx, by = m["pot_bay_x_m"], m["pot_bay_y_m"]
    geom(
        machine, "hotplate", "cylinder", [0.074, 0.001], [bx, by, platform], rgba="0.08 0.08 0.09 1"
    )
    geom(
        machine,
        "drip_tray",
        "box",
        [0.065, 0.055, 0.002],
        [bx, -0.077, platform],
        rgba="0.5 0.52 0.55 1",
    )
    for i in range(7):
        geom(
            machine,
            f"tray_slot{i}",
            "box",
            [0.05, 0.0015, 0.0005],
            [bx, -0.12 + i * 0.014, platform + 0.0025],
            rgba="0.1 0.1 0.1 1",
            contype=0,
            conaffinity=0,
        )

    p = cfg["pot"]
    pot_z = top + platform + 0.002
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
    )
    shell(
        asset,
        pot,
        "pot_glass",
        [p["radius_m"], p["radius_m"] * 0.92],
        p["height_m"] - 0.008,
        p["wall_m"],
        rgba=glass,
        mass=p["mass_kg"] * 0.45,
    )
    # Annular lid retains a real opening for the outlet proxy.
    lid_r = p["radius_m"] * 0.94
    shell(
        asset,
        pot,
        "pot_lid",
        [lid_r, lid_r],
        0.006,
        lid_r - p["lid_hole_radius_m"],
        z=p["height_m"] - 0.006,
        mass=p["mass_kg"] * 0.15,
    )
    outer = p["handle_outer_to_center_m"]
    hw, hh, ht, hd = (
        p["handle_outer_width_m"],
        p["handle_outer_height_m"],
        p["handle_thickness_m"],
        p["handle_depth_m"],
    )
    hz = p["height_m"] * 0.52
    # Four bars preserve the handle aperture and provide two grasp contact faces.
    for side, x in [("outer", outer - ht / 2), ("inner", outer - hw + ht / 2)]:
        geom(
            pot,
            f"handle_{side}",
            "box",
            [ht / 2, hd / 2, hh / 2],
            [x, 0, hz],
            mass=p["mass_kg"] * 0.045,
        )
    for side, z in [("top", hz + hh / 2 - ht / 2), ("bottom", hz - hh / 2 + ht / 2)]:
        geom(
            pot,
            f"handle_{side}",
            "box",
            [hw / 2, hd / 2, ht / 2],
            [outer - hw / 2, 0, z],
            mass=p["mass_kg"] * 0.035,
        )
    geom(
        pot,
        "handle_attachment",
        "box",
        [0.016, hd / 2, 0.012],
        [outer - hw - 0.008, 0, hz + hh / 2 - 0.012],
        mass=p["mass_kg"] * 0.055,
    )
    element(
        pot,
        "site",
        name="pot_handle_grasp",
        pos=numbers([outer - ht / 2, 0, hz]),
        size="0.003",
        rgba="1 0.4 0 1",
    )
    element(
        pot,
        "site",
        name="pot_spout",
        pos=numbers([-p["radius_m"] - 0.007, 0, p["height_m"] - 0.01]),
        size="0.003",
        rgba="0.1 0.5 1 1",
    )
    geom(
        pot,
        "spout_proxy",
        "box",
        [0.012, 0.012, 0.003],
        [-p["radius_m"], 0, p["height_m"] - 0.01],
        rgba=glass,
        mass=p["mass_kg"] * 0.005,
    )

    o = cfg["outlet"]
    outlet_z = platform + 0.002 + p["height_m"] - 0.003
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
    for i in range(3):
        a = i * 2 * math.pi / 3
        finger = element(
            dispenser,
            "body",
            name=f"retention_finger{i}",
            pos=numbers([rr * math.cos(a), rr * math.sin(a), ring_z + 0.004]),
            euler=numbers([0, 0, a]),
        )
        element(
            finger,
            "joint",
            name=f"finger_deflection{i}",
            type="slide",
            axis="1 0 0",
            range="0 0.012",
            stiffness=d["finger_stiffness_n_per_m"],
            damping="0.1",
        )
        geom(
            finger,
            f"retention_pad{i}",
            "box",
            [0.008, 0.004, 0.004],
            mass=0.002,
            rgba="0.3 0.3 0.33 1",
            friction=f"{d['finger_friction']} 0.005 0.0001",
            condim=4,
            solref="0.005 1",
            solimp="0.99 0.99 0.001",
        )
    for i in range(d["stack_count"]):
        cup(
            asset,
            world,
            cfg,
            f"cup_{i}",
            [
                tx + dx,
                ty + dy,
                top + ring_z + 0.009 - cfg["cup"]["height_m"] + i * d["stack_pitch_m"],
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
    # Camera -Z looks along +X; camera +Y is up. These are synthetic cameras.
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
        pos=numbers([tx - 0.2, 1.4, top + 0.4]),
        xyaxes="1 0 0 0 0 1",
        fovy="50",
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
    element(root, "option", timestep="0.002", integrator="implicitfast", cone="elliptic")
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
            )
    for side, arm in [("l", "left"), ("r", "right")]:
        driver = side + "tool_Left_1_Joint"
        element(
            actuator,
            "position",
            name=arm + "_gripper_servo",
            joint=driver,
            kp="4",
            kv="0.2",
            ctrlrange="-0.91 0",
            forcerange="-1 1",
        )
        body = root.find(f".//body[@name='{side}_link_8']")
        element(body, "site", name=arm + "_tool_mount", size="0.004", rgba="1 0.2 0.2 1")
    base = root.find(".//body[@name='base_link']")
    base.set("pos", numbers([*cfg["robot"]["base_xy_m"], 0]))
    base.set(
        "quat",
        numbers(
            [math.cos(cfg["robot"]["yaw_rad"] / 2), 0, 0, math.sin(cfg["robot"]["yaw_rad"] / 2)]
        ),
    )
    add_scene(root, cfg)
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
    exclusions += [("cup_dispenser", f"retention_finger{i}") for i in range(3)]
    for a, b in exclusions:
        element(contact, "exclude", body1=a, body2=b)
    path = output / "scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
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
        "mimic_constraints": len(equality),
        "qualification": cfg["qualification"],
        "assembly_contact_exclusions": exclusions,
        "fluid_simulation": False,
        "real_motion": False,
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
