"""Offline, conditional wrist clearance review; never enables or commands a robot.

Sweeps the entire wrist rotation and uncertain gripper/mount angles. This only
reviews moving wrist geometry relative to fixed robot geometry. It is not a
general self-collision planner or a calibration of the supplied candidate map.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from cap_harness.realman.clearance_sampling import DistancePairs, body_pair_lever_bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--joint", type=int, choices=(6, 7), default=7)
    parser.add_argument("--joint6-span-deg", type=float, default=0.25)
    parser.add_argument("--joint6-grid-deg", type=float, default=0.25)
    parser.add_argument("--wrist-grid-deg", type=float, default=1.0)
    parser.add_argument("--fast", action="store_true", help="Conservative bounding-sphere culling")
    parser.add_argument("--body-bound-mode", choices=("legacy", "kinematic"), default="legacy")
    args = parser.parse_args()
    arm_prefix = "l" if args.arm == "left" else "r"
    if not 0 < args.joint6_span_deg <= 10 or not 0 < args.joint6_grid_deg <= 0.5:
        raise ValueError("offline review range/grid outside supported limits")
    if not 0.05 <= args.wrist_grid_deg <= 1.0:
        raise ValueError("wrist grid must be between 0.05 and 1 degree")
    wrist_intervals = int(np.ceil(360 / args.wrist_grid_deg))
    wrist_grid_deg = 360 / wrist_intervals
    args.output.mkdir(parents=True, exist_ok=False)
    urdf = Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf")
    model_path = Path("artifacts/realman/coffee/robot_import.xml")
    mapping_path = Path("configs/realman/controller_urdf_candidate.json")
    state = json.loads(args.state.read_text())
    mapping = json.loads(mapping_path.read_text())
    build = json.loads(Path("artifacts/realman/coffee/build_report.json").read_text())
    assert build["urdf_sha256"] == hashlib.sha256(urdf.read_bytes()).hexdigest()
    source = ET.parse(urdf).getroot()
    tree = ET.parse(model_path)
    root = tree.getroot()
    # The source URDF's pad links have no collision meshes. This deliberately
    # generous extension covers pads/tool variation for this empty-gripper test.
    body = root.find(f".//body[@name='{arm_prefix}_link_7']")
    ET.SubElement(
        body,
        "geom",
        name="unmodelled_pad_guard",
        type="cylinder",
        pos="0 0 0.26",
        size="0.065 0.14",
        rgba="1 0.4 0 0.2",
        mass="0",
    )
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mj.MjData(model)
    for side, prefix in [("left", "l"), ("right", "r")]:
        cfg = mapping["arms"][side]
        for i, q in enumerate(state["controller"][side]["joint_deg"]):
            value = cfg["sign"][i] * q + cfg["offset_deg"][i]
            data.qpos[model.joint(f"{prefix}_joint_{i + 1}").qposadr] = np.deg2rad(value)
    # Torso is near zero; any ordering/sign uncertainty is included in the
    # explicit model margin, not silently declared a calibrated body mapping.
    torso = np.asarray(state["ros"]["state"]["body"]["position"]) / 1000
    if np.sum(np.abs(torso)) > 0.5:
        raise ValueError("torso not near zero: this narrow review does not apply")
    # Even if all five body angles have unknown signs/order, deviation from
    # zero is bounded by their sum. Bound motion of EACH point by a 2.5 m lever
    # arm (larger than this robot), and add the two point displacements.
    torso_pair_bound = 5.0 * float(np.deg2rad(np.sum(np.abs(torso))))
    for i, name in enumerate(["jb_joint", "tb_joint"]):
        data.qpos[model.joint(name).qposadr] = np.deg2rad(
            state["ros"]["state"]["head"]["position"][i] / 1000
        )
    moving = {model.body(f"{arm_prefix}_link_{args.joint}").id}
    for _ in range(model.nbody):
        moving.update(i for i in range(1, model.nbody) if model.body_parentid[i] in moving)
    moving_geoms = [
        i for i in range(model.ngeom) if model.geom_bodyid[i] in moving and model.geom_contype[i]
    ]
    fixed_geoms = [
        i
        for i in range(model.ngeom)
        if model.geom_contype[i] and model.geom_bodyid[i] not in moving
    ]
    pairs_by_region = {"same_arm": [], "other_robot": []}
    for a in moving_geoms:
        for b in fixed_geoms:
            if (
                model.geom(a).name == f"{arm_prefix}_link_{args.joint}_collision_0"
                and model.geom_bodyid[b] == model.body(f"{arm_prefix}_link_{args.joint - 1}").id
            ):
                continue
            region = (
                "same_arm"
                if model.body(model.geom_bodyid[b]).name.startswith(f"{arm_prefix}_link_")
                else "other_robot"
            )
            pairs_by_region[region].append((a, b))
    pair_queries = {r: DistancePairs(model, pairs) for r, pairs in pairs_by_region.items()}
    body_bound_review = None
    if args.body_bound_mode == "kinematic":
        body_bound_review = body_pair_lever_bound(
            model,
            pairs_by_region["other_robot"],
            ["xt_joint_1", "xt_joint_2", "dt_joint", "yb_joint", "xb_joint"],
        )
        # Cover ALL near-zero body states admitted above, not one noisy readback.
        body_bound_review["sum_absolute_angle_budget_deg"] = 0.5
        torso_pair_bound = body_bound_review["maximum_pair_lever_m"] * np.deg2rad(0.5)
    moving_ids = np.asarray(moving_geoms)
    mimics = [
        (j.get("name"), j.find("mimic"))
        for j in source.findall("joint")
        if j.find("mimic") is not None
    ]
    closest = {"distance_m": 1.0}
    closest_by_region = {"same_arm": {"distance_m": 1.0}, "other_robot": {"distance_m": 1.0}}
    max_radius = 0.0
    max_wrist_radius = 0.0
    samples = 0
    q6_start = float(data.qpos[model.joint(f"{arm_prefix}_joint_6").qposadr][0])
    for aperture in np.linspace(-0.91, 0, 3):
        for side in ["l", "r"]:
            data.qpos[model.joint(side + "tool_Left_1_Joint").qposadr] = aperture
        for name, mimic in mimics:
            data.qpos[model.joint(name).qposadr] = data.qpos[
                model.joint(mimic.get("joint")).qposadr
            ] * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0"))
        for mount in np.linspace(0, 0.78, 5):
            data.qpos[model.joint(f"{arm_prefix}_camera_rojoint").qposadr] = mount
            offsets = (
                np.linspace(
                    -args.joint6_span_deg,
                    args.joint6_span_deg,
                    int(np.ceil(2 * args.joint6_span_deg / args.joint6_grid_deg)) + 1,
                )
                if args.joint == 6
                else [0]
            )
            for angle, offset in itertools.product(
                np.linspace(-np.pi, np.pi, wrist_intervals + 1), offsets
            ):
                data.qpos[model.joint(f"{arm_prefix}_joint_7").qposadr] = angle
                data.qpos[model.joint(f"{arm_prefix}_joint_6").qposadr] = q6_start + np.deg2rad(
                    offset
                )
                mj.mj_forward(model, data)
                samples += 1
                center = data.xpos[model.body(f"{arm_prefix}_link_{args.joint}").id]
                wrist_center = data.xpos[model.body(f"{arm_prefix}_link_7").id]
                # Bounding spheres give a conservative bound on point movement
                # between consecutive sampled wrist angles.
                if args.fast:
                    positions = data.geom_xpos[moving_ids]
                    radii = model.geom_rbound[moving_ids]
                    max_radius = max(
                        max_radius,
                        float(np.max(np.linalg.norm(positions - center, axis=1) + radii)),
                    )
                    max_wrist_radius = max(
                        max_wrist_radius,
                        float(np.max(np.linalg.norm(positions - wrist_center, axis=1) + radii)),
                    )
                    for region, query in pair_queries.items():
                        distance, pair = query.minimum(model, data)
                        record = {
                            "distance_m": distance,
                            "moving_geom": model.geom(pair[0]).name if pair else None,
                            "fixed_geom": model.geom(pair[1]).name if pair else None,
                            "wrist_deg": float(np.rad2deg(angle)),
                            "joint6_offset_deg": offset,
                            "gripper_rad": float(aperture),
                            "mount_rad": float(mount),
                        }
                        if distance < closest_by_region[region]["distance_m"]:
                            closest_by_region[region] = record
                        if distance < closest["distance_m"]:
                            closest = record
                    continue
                for a in moving_geoms:
                    max_radius = max(
                        max_radius,
                        float(np.linalg.norm(data.geom_xpos[a] - center) + model.geom_rbound[a]),
                    )
                    max_wrist_radius = max(
                        max_wrist_radius,
                        float(
                            np.linalg.norm(data.geom_xpos[a] - wrist_center) + model.geom_rbound[a]
                        ),
                    )
                    for b in fixed_geoms:
                        if (
                            model.geom(a).name == f"{arm_prefix}_link_{args.joint}_collision_0"
                            and model.geom_bodyid[b]
                            == model.body(f"{arm_prefix}_link_{args.joint - 1}").id
                        ):
                            continue  # Only the direct bearing/flange pair.
                        distance = float(mj.mj_geomDistance(model, data, a, b, 0.15, None))
                        record = {
                            "distance_m": distance,
                            "moving_geom": model.geom(a).name,
                            "fixed_geom": model.geom(b).name,
                            "wrist_deg": float(np.rad2deg(angle)),
                            "joint6_offset_deg": offset,
                            "gripper_rad": float(aperture),
                            "mount_rad": float(mount),
                        }
                        region = (
                            "same_arm"
                            if model.body(model.geom_bodyid[b]).name.startswith(
                                f"{arm_prefix}_link_"
                            )
                            else "other_robot"
                        )
                        if distance < closest_by_region[region]["distance_m"]:
                            closest_by_region[region] = record
                        if distance < closest["distance_m"]:
                            closest = record
    wrist_sampling_bound = 2 * max_wrist_radius * np.sin(np.deg2rad(wrist_grid_deg / 2) / 2)
    motion_sampling_bound = (
        2 * max_radius * np.sin(np.deg2rad(args.joint6_grid_deg / 2) / 2) if args.joint == 6 else 0
    )
    sampling_bound = wrist_sampling_bound + motion_sampling_bound
    # Body pose / arm-base registration cancels for two points on the same arm.
    # Do not charge that global error to the wrist-versus-parent-housing gap.
    # 5 mm inflation per CAD part is an explicit local engineering assumption,
    # NOT a measured tolerance. Other robot parts retain global allowances.
    local_cad_allowance = 0.01
    model_allowance = 0.03
    region_lower = {
        "same_arm": closest_by_region["same_arm"]["distance_m"]
        - sampling_bound
        - local_cad_allowance,
        "other_robot": closest_by_region["other_robot"]["distance_m"]
        - sampling_bound
        - model_allowance
        - torso_pair_bound,
    }
    lower = min(region_lower.values())
    report = {
        "arm": args.arm,
        "distance_sampling": "bounding_sphere_culling" if args.fast else "exhaustive",
        "mujoco_version": mj.__version__,
        "scope": f"{args.arm} J{args.joint} only; other joints held; full J7 sweep accounts for uncertain wrist mapping",
        "joint6_reviewed_offset_range_deg": [-args.joint6_span_deg, args.joint6_span_deg]
        if args.joint == 6
        else [0, 0],
        "qualified_for_general_motion": False,
        "qualified_for_physical_motion": False,
        "conditional_sample_clearance_positive": bool(lower > 0),
        "physical_gap_confirmation_required": True,
        "closest_sample": closest,
        "samples": samples,
        "closest_by_region": closest_by_region,
        "conditional_lower_by_region_m": region_lower,
        "wrist_full_rotation_swept": True,
        "wrist_grid_deg": wrist_grid_deg,
        "max_vertex_radius_bound_m": max_radius,
        "wrist_sampling_bound_m": float(wrist_sampling_bound),
        "joint6_sampling_bound_m": float(motion_sampling_bound),
        "body_mapping_pair_displacement_bound_m": torso_pair_bound,
        "body_bound_review": body_bound_review,
        "additional_engineering_allowance_m": model_allowance,
        "local_cad_pair_allowance_m": local_cad_allowance,
        "conditional_clearance_after_allowances_m": float(lower),
        "exclusions": [
            f"internal moving-assembly pairs (rigid under J{args.joint} motion)",
            f"{arm_prefix}_link_{args.joint}_collision_0 versus {arm_prefix}_link_{args.joint - 1} (direct bearing/flange interface only)",
        ],
        "unmodelled_pad_guard": {"radius_m": 0.065, "z_range_m": [0.12, 0.40]},
        "limitations": [
            "candidate mapping; photograph review is not metric calibration",
            "mount/gripper samples and assumed CAD allowances are not certified continuous uncertainty bounds",
            "cables, board and environment are not collision meshes",
            "no permission for full wrist rotation or any other joint motion",
        ],
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [urdf, model_path, mapping_path, args.state]
        },
    }
    (args.output / "review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
