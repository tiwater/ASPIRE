"""Bounded virtual-grip diagonal withdrawal diagnostic; never robot-policy success.

Only an explicit external wrench acts on the pot. No pose writes, attachments,
collision exclusions or spring overrides are used. The robot is not controlled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def diagonal_withdraw_delta(config):
    """Nominal table axes: +X away from robot, +Y robot left, +Z up."""
    task = config["task"]
    direction = np.asarray(task["withdraw_table_direction"], dtype=float)
    distance = float(task["withdraw_distance_m"])
    if (
        direction.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isfinite(distance)
        or not 0 < distance <= 0.5
        or not direction[0] < 0 < direction[1]
        or abs(direction[2]) > 1e-9
        or not np.isclose(np.linalg.norm(direction), 1, atol=1e-6)
    ):
        raise ValueError("pot requires a unit horizontal left-back diagonal and positive distance")
    return direction * distance


def run_pot_bench(model_path, output):
    model_path, output = Path(model_path), Path(output)
    cfg = json.loads(model_path.with_name("scene_parameters.json").read_text())
    delta = diagonal_withdraw_delta(cfg)
    output.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b, dof = m.body("pot").id, m.joint("pot_free").dofadr[0]
    qadr = m.joint("pot_free").qposadr[0]
    direction, distance = delta / np.linalg.norm(delta), float(np.linalg.norm(delta))
    settle_s, speed, hold_s, force_limit = 2.0, 0.01, 1.0, 30.0
    duration = settle_s + distance / speed + hold_s
    origin, rotation = None, None
    peak_pull, peak_compression, max_tilt = 0.0, 0.0, 0.0
    samples, contact_peaks = [], {}
    failure = None
    for step in range(round(duration / m.opt.timestep)):
        if d.time >= settle_s:
            if origin is None:
                origin = d.body(b).xpos.copy()
                rotation = d.body(b).xmat.reshape(3, 3).copy()
            r = d.body(b).xmat.reshape(3, 3)
            target = origin + direction * min(distance, (d.time - settle_s) * speed)
            force = np.clip(
                2000 * (target - d.body(b).xpos)
                - 20 * d.qvel[dof : dof + 3]
                + [0, 0, m.body_mass[b] * 9.81],
                -force_limit,
                force_limit,
            )
            torque = np.clip(
                20 * Rotation.from_matrix(rotation @ r.T).as_rotvec()
                - 0.2 * (r @ d.qvel[dof + 3 : dof + 6]),
                -1,
                1,
            )
            d.xfrc_applied[b] = np.r_[force, torque]
            peak_pull = max(peak_pull, float(force @ direction))
        mujoco.mj_step(m, d)
        # Check the integrated orientation every physics step; a 0.1 s log
        # interval can miss the onset of a contact-induced tip/instability.
        quat = d.qpos[qadr + 3 : qadr + 7]
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))))
        max_tilt = max(max_tilt, tilt)
        if max_tilt >= 3 or np.any(d.warning.number):
            failure = "virtual grip lost upright constraint or simulation warning"
        peak_compression = max(peak_compression, float(d.joint("outlet_compression").qpos[0]))
        if failure or step % max(1, round(0.1 / m.opt.timestep)) == 0:
            mujoco.mj_forward(m, d)
            max_tilt = max(
                max_tilt, float(np.degrees(np.arccos(np.clip(d.body(b).xmat[8], -1, 1))))
            )
            contacts = []
            for i, c in enumerate(d.contact[: d.ncon]):
                if b not in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]):
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, wrench)
                if wrench[0] <= 0.02:
                    continue
                pair = [m.geom(c.geom1).name, m.geom(c.geom2).name]
                contact_peaks[" / ".join(pair)] = max(
                    contact_peaks.get(" / ".join(pair), 0), float(wrench[0])
                )
                contacts.append(
                    {
                        "pair": pair,
                        "normal_force_n": float(wrench[0]),
                        "normal_world": c.frame[:3].tolist(),
                        "penetration_m": -float(c.dist),
                    }
                )
            samples.append(
                {
                    "time_s": float(d.time),
                    "pot_xyz_m": d.body(b).xpos.tolist(),
                    "test_grip_wrench_world": d.xfrc_applied[b].tolist(),
                    "compression_m": float(d.joint("outlet_compression").qpos[0]),
                    "contacts": contacts,
                }
            )
            if failure:
                break
    mujoco.mj_forward(m, d)
    error = None if origin is None else float(np.linalg.norm(d.body(b).xpos - origin - delta))
    contacts_at_end = any(
        m.geom_bodyid[c.geom1] == b or m.geom_bodyid[c.geom2] == b for c in d.contact[: d.ncon]
    )
    result = {
        "simulation_only": True,
        "physical_motion_allowed": False,
        "purpose": "bounded virtual grip; does not validate robot grasp or policy",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mujoco_version": mujoco.__version__,
        "parameters": {
            "delta_table_m": delta.tolist(),
            "speed_m_s": speed,
            "force_limit_per_axis_n": force_limit,
            "torque_limit_per_axis_nm": 1,
            "settle_s": settle_s,
            "hold_s": hold_s,
            "kp_n_m": 2000,
            "kd_ns_m": 20,
            "orientation_kp_nm_rad": 20,
            "orientation_kd_nms_rad": 0.2,
        },
        "peak_pull_n": peak_pull,
        "max_spring_compression_m": peak_compression,
        "max_tilt_deg": max_tilt,
        "error": failure,
        "simulation_seconds": float(d.time),
        "final_position_error_m": error,
        "displacement_m": None if origin is None else (d.body(b).xpos - origin).tolist(),
        "sampled_contact_peak_normal_n": contact_peaks,
        "final_pot_contacts_present": contacts_at_end,
        "warnings": d.warning.number.tolist(),
        "bench_success": bool(
            failure is None
            and error is not None
            and error < 0.003
            and max_tilt < 3
            and not contacts_at_end
            and not np.any(d.warning.number)
        ),
        "coffee_task_success": None,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "trace.json").write_text(json.dumps(samples, indent=2) + "\n")
    (output / "bench_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "scene_parameters.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return result
