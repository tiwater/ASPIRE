"""Passive cup-dispenser test with a bounded virtual grip, never robot success.

No object pose writes, welds, disabled collisions or retention forces. Only cup_0
receives the explicitly logged test-grip wrench; all other cups and tabs are free.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .sim_evaluation import CoffeeGeometryEvaluator


def run_dispenser_bench(
    model_path, output, *, hold_seconds=10.0, cup_friction=None, stiffness_scale=1.0
):
    model_path, output = Path(model_path), Path(output)
    source_bytes = Path(__file__).read_bytes()
    if not np.isfinite([hold_seconds, stiffness_scale]).all() or not (
        0.3 <= hold_seconds <= 300 and 0 <= stiffness_scale <= 10
    ):
        raise ValueError("invalid hold time or stiffness scale")
    if cup_friction is not None and (not np.isfinite(cup_friction) or not 0 <= cup_friction <= 2):
        raise ValueError("invalid cup friction")
    cfg = json.loads(model_path.with_name("scene_parameters.json").read_text())
    if cfg["dispenser"].get("retention_mechanism") != "elastic_rim_tabs":
        raise ValueError("bench requires elastic rim tabs")
    output.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    cups = [f"cup_{i}" for i in range(cfg["dispenser"]["stack_count"])]
    if len(cups) < 2:
        raise ValueError("bench requires at least two cups")
    # Explicit diagnostic overrides are saved in the report; source XML is unchanged.
    for g in range(m.ngeom):
        if cup_friction is not None and m.body(m.geom_bodyid[g]).name in cups:
            m.geom_friction[g, 0] = cup_friction
    joints = [m.joint(f"finger_deflection{i}").id for i in range(cfg["dispenser"]["latch_count"])]
    m.jnt_stiffness[joints] *= stiffness_scale
    mujoco.mj_forward(m, d)
    evaluator = CoffeeGeometryEvaluator(SimpleNamespace(model=m, data=d, mj=mujoco), cfg)
    initial = {n: d.body(n).xpos.copy() for n in cups}
    target_body = m.body(cups[0]).id
    # Position error is at the free-body origin, not its offset centre of mass.
    dof = m.joint(cups[0] + "_free").dofadr[0]
    # 80 mm leaves lip/bottom contact with the next cup; 90 mm hits the rack
    # foot. 85 mm fits this nominal fixture with ~5 mm upper and ~1 mm lower
    # clearance. This bench is not a general-purpose extraction planner.
    settle_s, distance, speed, force_limit = 2.0, 0.085, 0.01, 8.0
    pull_end = settle_s + distance / speed
    end = pull_end + hold_seconds
    peak_force, max_displacement, max_tilt = 0.0, 0.0, 0.0
    min_angles = np.zeros(len(joints))
    samples, origin, initial_hold, hold_valid = [], None, None, True
    evidence = evaluator.update()
    sample_steps = max(1, round(0.02 / m.opt.timestep))
    log_steps = max(1, round(0.1 / m.opt.timestep))
    for step in range(round(end / m.opt.timestep)):
        if d.time >= settle_s:
            if origin is None:
                origin = d.body(cups[0]).xpos.copy()
                initial_hold = all(
                    np.linalg.norm(d.body(n).xpos - initial[n]) < 0.003
                    and d.body(n).xmat[8] > np.cos(np.deg2rad(3))
                    for n in cups
                )
            goal = origin + [0, 0, -min(distance, (d.time - settle_s) * speed)]
            force = np.clip(
                1000 * (goal - d.body(cups[0]).xpos) - 5 * d.qvel[dof : dof + 3],
                -force_limit,
                force_limit,
            )
            rotation_error = Rotation.from_matrix(d.body(cups[0]).xmat.reshape(3, 3).T).as_rotvec()
            torque = np.clip(
                0.02 * rotation_error
                - 0.001 * (d.body(cups[0]).xmat.reshape(3, 3) @ d.qvel[dof + 3 : dof + 6]),
                -0.02,
                0.02,
            )
            d.xfrc_applied[target_body] = np.r_[force, torque]
            peak_force = max(peak_force, -float(force[2]))
        mujoco.mj_step(m, d)
        angles = d.qpos[m.jnt_qposadr[joints]]
        min_angles = np.minimum(min_angles, angles)
        if step % sample_steps == 0:
            mujoco.mj_forward(m, d)
            evidence = evaluator.update()
            displacements = [np.linalg.norm(d.body(n).xpos - initial[n]) for n in cups[1:]]
            tilts = [np.degrees(np.arccos(np.clip(d.body(n).xmat[8], -1, 1))) for n in cups[1:]]
            max_displacement = max(max_displacement, *displacements)
            max_tilt = max(max_tilt, *tilts)
            if d.time >= pull_end:
                hold_valid &= evidence["evidence"]["single_cup_separated"].value is True
                hold_valid &= not evidence["failures"]
        if step % log_steps == 0:
            samples.append(
                {
                    "time_s": float(d.time),
                    "cup_positions_m": {n: d.body(n).xpos.tolist() for n in cups},
                    "tab_angles_rad": angles.tolist(),
                    "test_grip_wrench_world": d.xfrc_applied[target_body].tolist(),
                    "remaining_cups_retained": evidence["evidence"][
                        "remaining_cups_retained"
                    ].value,
                    "single_cup_separated": evidence["evidence"]["single_cup_separated"].value,
                }
            )
    mujoco.mj_forward(m, d)
    evidence = evaluator.update()
    final_angles = d.qpos[m.jnt_qposadr[joints]]
    recovered = bool(np.all(np.abs(final_angles) < 0.15))
    result = {
        "simulation_only": True,
        "physical_motion_allowed": False,
        "purpose": "virtual compliant grip mechanism bench, not robot policy",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "overrides": {"cup_friction": cup_friction, "stiffness_scale": stiffness_scale},
        "parameters": {
            "settle_seconds": settle_s,
            "pull_distance_m": distance,
            "pull_speed_m_s": speed,
            "force_limit_per_axis_n": force_limit,
            "hold_seconds": hold_seconds,
            "sample_period_s": sample_steps * m.opt.timestep,
        },
        "initial_stack_held": bool(initial_hold),
        "continuous_final_hold_passed": bool(hold_valid),
        "tabs_recovered": recovered,
        "peak_downward_test_force_n": peak_force,
        "remaining_max_displacement_m": max_displacement,
        "remaining_max_tilt_deg": max_tilt,
        "minimum_tab_angles_rad": min_angles.tolist(),
        "final_tab_angles_rad": final_angles.tolist(),
        "final_displacements_m": {n: (d.body(n).xpos - initial[n]).tolist() for n in cups},
        "warnings": d.warning.number.tolist(),
        "bench_success": bool(
            initial_hold
            and hold_valid
            and recovered
            and not np.any(d.warning.number)
            and evidence["evidence"]["single_cup_separated"].value
        ),
        "coffee_task_success": None,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "trace.json").write_text(json.dumps(samples, indent=2) + "\n")
    (output / "scene_parameters.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "bench_source.py").write_bytes(source_bytes)
    return result
