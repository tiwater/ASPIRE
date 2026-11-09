"""Create a fresh loaded 135-degree scene with unchanged stock gripper pads.

Run from the repository root with PYTHONPATH=src. This has no hardware IO.
Only robot reset joints are imported; objects, velocities and time start fresh.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from cap_harness.realman.payload import (
    body_mass_properties,
    coffee_cavities,
    combine_mass_properties,
)
from cap_harness.realman.sim import CoffeeSimAdapter
from cap_harness.realman.workspace import robot_contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", type=Path, default=Path("configs/realman/coffee_workload.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workload = json.loads(args.workload.read_text())
    if workload["gripper_pad_modification_allowed"] is not False:
        raise ValueError("this builder only supports unchanged stock pads")
    source = Path(workload["source_review"])
    scene = source / "model/scene.xml"
    prior = json.loads((source / "result.json").read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(scene) != prior["model_sha256"]:
        raise ValueError("source review/model hash mismatch")
    config = json.loads((source / "scene_config.json").read_text())
    if workload.get("simulation_diagnostics"):
        config["simulation_diagnostics"] = workload["simulation_diagnostics"]
    base = mj.MjModel.from_xml_path(str(scene))
    cavities = coffee_cavities(config)
    request = workload["initial_pot_load"]
    equivalents = float(request["cup_equivalents"])
    fraction = float(request["fraction_of_cup_capacity_per_equivalent"])
    if not np.isfinite(equivalents) or equivalents <= 0 or not 0 < fraction <= 1:
        raise ValueError("invalid cup-equivalent load")
    if request["density_kg_m3"] != config["force_priors"]["water_density_kg_m3"]:
        raise ValueError("workload density must agree with the scene")
    if workload["served_cup_fill_fraction"] != config["task"]["fill_fraction"]:
        raise ValueError("served fill target must agree with the task")
    volume = equivalents * fraction * cavities["cup_0"].capacity_m3
    dry = body_mass_properties(base, "pot")
    liquid = cavities["pot"].properties(volume)
    loaded = combine_mass_properties(dry, liquid)
    tree = ET.parse(scene)
    body = tree.find(".//body[@name='pot']")
    if workload.get("hotplate_recess") is not None:
        from cap_harness.realman.platform import install_recessed_platform, nominal_seated_height

        config["machine"]["hotplate_recess"] = workload["hotplate_recess"]
        if "hotplate_radius_m" in workload:
            config["machine"]["hotplate_radius_m"] = workload["hotplate_radius_m"]
        config["platform_geometry"] = install_recessed_platform(
            tree.find("asset"),
            tree.find(".//body[@name='coffee_machine']"),
            config["machine"],
            config["pot"],
        )
        initial_position = list(map(float, body.get("pos").split()))
        initial_position[2] = (
            config["table"]["top_height_m"]
            + nominal_seated_height(config["machine"], config["pot"])
            + 0.001
        )
        body.set("pos", " ".join(map(str, initial_position)))
    if body.find("inertial") is not None:
        raise ValueError("expected original dry pot with geometry-derived inertia")
    inertia = loaded.inertia_kg_m2
    ET.SubElement(
        body,
        "inertial",
        mass=str(loaded.mass_kg),
        pos=" ".join(map(str, loaded.center_m)),
        fullinertia=" ".join(
            map(
                str,
                [
                    inertia[0, 0],
                    inertia[1, 1],
                    inertia[2, 2],
                    inertia[0, 1],
                    inertia[0, 2],
                    inertia[1, 2],
                ],
            )
        ),
    )
    if "floor_collision_segments" in workload:
        from cap_harness.realman.platform import install_convex_pot_floor

        config["pot_floor_collision"] = install_convex_pot_floor(
            tree.find("asset"), body, segments=workload["floor_collision_segments"]
        )
    spring_scale = float(workload.get("outlet_spring_force_scale", 1.0))
    if not np.isfinite(spring_scale) or not 0 < spring_scale <= 1:
        raise ValueError("outlet spring force scale must be in (0, 1]")
    outlet = tree.find(".//joint[@name='outlet_compression']")
    source_stiffness = float(outlet.get("stiffness"))
    outlet.set("stiffness", str(source_stiffness * spring_scale))
    # Keep spring reference position: both elastic force and any preload scale.
    # Damping, travel, geometry, friction and outlet mass are unchanged.
    config["outlet"]["spring_n_per_m"] = source_stiffness * spring_scale
    config["outlet"]["preload_n"] *= spring_scale
    config["force_priors"]["spring_full_pot_weight_fraction"] *= spring_scale
    spring_trial = {
        "scale_relative_to_source_review": spring_scale,
        "source_stiffness_n_per_m": source_stiffness,
        "stiffness_n_per_m": source_stiffness * spring_scale,
        "spring_reference_position_m": float(outlet.get("springref", "0")),
        "damping_unchanged": True,
        "hardware_measured": False,
    }
    config["outlet_spring_trial"] = spring_trial
    if workload.get("outlet_conical_hypothesis") is not None:
        from cap_harness.realman.outlet import install_conical_outlet

        if spring_scale != 1:
            raise ValueError("use explicit conical stiffness, not simultaneous spring scaling")
        config["outlet"]["conical_hypothesis"] = workload["outlet_conical_hypothesis"]
        reference_data = mj.MjData(base)
        mj.mj_forward(base, reference_data)
        outlet_body = tree.find(".//body[@name='spring_outlet']")
        lid_z = (
            float(body.get("pos").split()[2])
            - reference_data.body("coffee_machine").xpos[2]
            + config["pot"]["height_m"]
            + config["pot"]["lid_boss_profile_radius_rise_m"][0][1]
        )
        spring_trial = install_conical_outlet(
            tree.find("asset"), outlet_body, config, lid_initial_z_m=float(lid_z)
        )
        config["outlet_spring_trial"] = spring_trial
    servo = tree.find(".//position[@name='left_gripper_servo']")
    effort = float(workload["left_gripper_effort_limit_nm"])
    prior_limit = min(abs(float(v)) for v in servo.get("forcerange").split())
    if not np.isfinite(effort) or not 0 < effort <= prior_limit:
        raise ValueError("load-only trials may reduce, not increase, the source effort limit")
    servo.set("forcerange", f"{-effort} {effort}")
    option = tree.find("option")
    timestep = float(workload["contact_solver"]["timestep_s"])
    if not np.isfinite(timestep) or not 0.000125 <= timestep <= 0.001:
        raise ValueError("invalid diagnostic timestep")
    option.set("timestep", str(timestep))
    if "ccd_tolerance" in workload["contact_solver"]:
        ccd_tolerance = float(workload["contact_solver"]["ccd_tolerance"])
        if not np.isfinite(ccd_tolerance) or not 1e-10 <= ccd_tolerance <= 1e-6:
            raise ValueError("CCD tolerance must be in [1e-10, 1e-6]")
        option.set("ccd_tolerance", str(ccd_tolerance))
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    if workload["contact_solver"]["multiccd"] is not True:
        raise ValueError("this candidate requires the verified multiccd option")
    flag.set("multiccd", "enable")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    review = root / "review"
    (review / "model").mkdir(parents=True)
    new_scene = review / "model/scene.xml"
    tree.write(new_scene, encoding="unicode")
    adapter = CoffeeSimAdapter(new_scene)
    try:
        model, data = adapter.model, adapter.data
        # Verify actual compiled stock geometry, material proxies and pad mass.
        for arm in ("left", "right"):
            for side in ("left", "right"):
                name = f"{arm}_{side}_pad_contact"
                for field in ("size", "pos", "quat", "friction", "solref", "solimp", "solmix"):
                    np.testing.assert_array_equal(
                        getattr(model.geom(name), field), getattr(base.geom(name), field)
                    )
                pad = f"{arm[0]}tool_{side.capitalize()}_Pad_Link"
                for field in ("mass", "inertia", "ipos", "iquat"):
                    np.testing.assert_array_equal(
                        getattr(model.body(pad), field), getattr(base.body(pad), field)
                    )
        prior_q = np.asarray(prior["qpos"])
        for joint in range(model.njnt):
            name = model.joint(joint).name
            if name.startswith(("l_joint_", "r_joint_", "ltool_", "rtool_")):
                data.qpos[model.jnt_qposadr[joint]] = prior_q[base.joint(name).qposadr[0]]
        for arm in adapter.arms:
            data.ctrl[adapter.actuators[arm]] = data.qpos[adapter.qadr[arm]]
            data.ctrl[model.actuator(arm + "_gripper_servo").id] = data.qpos[
                model.joint(arm[0] + "tool_Left_1_Joint").qposadr[0]
            ]
        mj.mj_forward(model, data)
        if robot_contacts(model, data):
            raise ValueError("initial robot reset is in collision")
        properties = {
            "simulation_only": True,
            "fluid_simulated": False,
            "mode": "rigid_upright_fill_attached_to_vessel",
            "pot_initial_ml": volume * 1e6,
            "cup_initial_ml": 0,
            "cup_capacity_ml": cavities["cup_0"].capacity_m3 * 1e6,
            "cup_equivalents": equivalents,
            "fraction_per_equivalent": fraction,
            "stock_pads_verified_unchanged": True,
            "outlet_spring_trial": spring_trial,
            "source_model_sha256": prior["model_sha256"],
            "model_sha256": digest(new_scene),
            "dry": {name: body_mass_properties(base, name).as_dict() for name in ("pot", "cup_0")},
            "pot_liquid": liquid.as_dict(),
            "pot_combined": loaded.as_dict(),
            "workload": workload,
        }
        config["payload"] = properties
        config["gripper_servos"]["left"]["effort_limit_nm"] = effort
        config["contact_solver"].update(workload["contact_solver"])
        (review / "scene_config.json").write_text(json.dumps(config, indent=2) + "\n")
        plan = {
            **prior,
            "model_sha256": digest(new_scene),
            "source_model_sha256": prior["model_sha256"],
            "qpos": data.qpos.tolist(),
            "payload": properties,
        }
        (review / "result.json").write_text(json.dumps(plan, indent=2) + "\n")
        (root / "payload.json").write_text(json.dumps(properties, indent=2) + "\n")
        (root / "prepare_source.py").write_bytes(Path(__file__).read_bytes())
        state = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
        mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
        checkpoint = root / "initial_pregrasp.npz"
        np.savez_compressed(checkpoint, state=state)
        checkpoint.with_suffix(".json").write_text(
            json.dumps(
                {
                    "model_sha256": digest(new_scene),
                    "npz_sha256": digest(checkpoint),
                    "time_s": 0,
                    "allowed_contacts": [],
                    "simulation_only": True,
                    "initialization": "fresh model objects; reviewed robot joints only; zero velocity/time",
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "pot_initial_ml": volume * 1e6,
                    "pot_total_mass_kg": loaded.mass_kg,
                    "stock_pads_verified_unchanged": True,
                    "model_sha256": digest(new_scene),
                }
            )
        )
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
