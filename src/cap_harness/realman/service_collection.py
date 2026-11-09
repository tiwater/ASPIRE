"""Explicit simulated staff pickup of completed cups from the work table.

This is an environmental event, not a robot action or a model of human motion.
Each collected cup remains a distinct, loaded body in an off-workspace area.
The robot, other objects, payload ledger, mass properties and clock are preserved.
"""

from __future__ import annotations

import copy

import numpy as np

from .service_evaluation import next_table_slot


def fixed_delivery_slot(config):
    """Reuse the previously qualified first-cup position after staff pickup."""
    return next_table_slot(config, 0)


class StaffCupCollection:
    def __init__(self, adapter, config, payload, *, simulation_only):
        if simulation_only is not True:
            raise ValueError("automatic staff pickup is a simulation assumption only")
        self.adapter = adapter
        self.config = config
        self.payload = payload
        self.events = []

    def state(self):
        return {
            "mode": "explicit_simulated_staff_collection",
            "simulation_only": True,
            "human_motion_simulated": False,
            "cup_identities_and_payload_preserved": True,
            "events": copy.deepcopy(self.events),
        }

    def collect(self, cup, *, staff_confirmed):
        if staff_confirmed is not True:
            raise ValueError("staff collection requires explicit confirmation")
        a = self.adapter
        m, d, mj = a.model, a.data, a.mj
        evaluator = a.task_evaluator
        if cup not in evaluator.served or cup in evaluator.collected_cups:
            raise ValueError("collect a completed, not previously collected cup")
        payload_before = self.payload.state()
        volume = payload_before["vessel_volumes_m3"][cup]
        target = .8 * self.config["payload"]["cup_capacity_ml"] / 1e6
        if not np.isfinite(volume) or abs(volume - target) > 1e-12:
            raise ValueError("collection requires a completed 80-percent serving")
        body = m.body(cup).id
        pos = d.body(cup).xpos.copy()
        half = np.asarray(self.config["table"]["size_m"][:2]) / 2
        center = np.asarray(self.config["table"]["center_xy_m"])
        margin = float(np.min(half - np.abs(pos[:2] - center)))
        margin -= self.config["cup"]["bottom_radius_m"]
        velocity = np.zeros(6)
        mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
        up = float(d.body(cup).xmat[8])
        support = 0.0
        unexpected = []
        right_busy = False
        vessels = set(payload_before["vessel_volumes_m3"])
        for i, contact in enumerate(d.contact):
            force = np.zeros(6)
            mj.mj_contactForce(m, d, i, force)
            if not np.isfinite(force).all():
                raise ValueError("nonfinite pickup contact evidence")
            if force[0] <= .02:
                continue
            g1, g2 = contact.geom1, contact.geom2
            b1, b2 = m.body(m.geom_bodyid[g1]).name, m.body(m.geom_bodyid[g2]).name
            right_busy |= (b1.startswith("rtool_") and b2 in vessels) or (
                b2.startswith("rtool_") and b1 in vessels
            )
            if cup not in {b1, b2}:
                continue
            other = g2 if b1 == cup else g1
            if m.geom(other).name == "tabletop":
                support += float(force[0])
            elif m.geom_bodyid[other] != body:
                unexpected.append(m.geom(other).name)
        if (
            not np.isfinite(pos).all()
            or margin < .02
            or up <= np.cos(np.deg2rad(8))
            or np.linalg.norm(velocity[:3]) >= .15
            or np.linalg.norm(velocity[3:]) >= .015
            or support <= .02
            or unexpected
            or right_busy
            or np.linalg.norm(pos - evaluator.served[cup]) >= .01
        ):
            raise ValueError("cup must be stable on the table and released before staff pickup")
        joint = int(m.body_jntadr[body])
        if joint < 0 or m.jnt_type[joint] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError("collection requires an independently movable cup body")
        qadr, vadr = int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint])
        q_before, v_before = d.qpos.copy(), d.qvel.copy()
        masses = m.body_mass.copy()
        time = float(d.time)
        # Off-workspace bookkeeping: staff carries the cup out of the scene.
        # Keep a physical body and its contents; no cup/rack reset or deletion.
        parked = np.array([-1.5, -1.5 - .15 * len(self.events), .0001])
        receipt = {
            "event": "external_staff_cup_collection",
            "event_id": len(self.events),
            "actor": "simulated_staff",
            "confirmation_source": "operator_assumption",
            "cup_id": cup,
            "time_s": time,
            "volume_ml": volume * 1e6,
            "before_position_m": pos.tolist(),
            "before_quaternion_wxyz": q_before[qadr + 3:qadr + 7].tolist(),
            "after_position_m": parked.tolist(),
            "after_quaternion_wxyz": [1, 0, 0, 0],
            "observed_before": {
                "table_support_force_n": support,
                "table_margin_m": margin,
                "upright_cosine": up,
                "angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
                "linear_speed_m_s": float(np.linalg.norm(velocity[3:])),
                "right_gripper_empty": not right_busy,
                "other_contacts": unexpected,
            },
            "physical_robot_motion_executed": False,
            "human_motion_simulated": False,
        }
        d.qpos[qadr:qadr + 3] = parked
        d.qpos[qadr + 3:qadr + 7] = [1, 0, 0, 0]
        d.qvel[vadr:vadr + 6] = 0
        d.qacc_warmstart[vadr:vadr + 6] = 0
        mj.mj_forward(m, d)
        qi = np.ones(m.nq, dtype=bool); qi[qadr:qadr + 7] = False
        vi = np.ones(m.nv, dtype=bool); vi[vadr:vadr + 6] = False
        if (
            not np.array_equal(d.qpos[qi], q_before[qi])
            or not np.array_equal(d.qvel[vi], v_before[vi])
            or not np.array_equal(m.body_mass, masses)
            or float(d.time) != time
            or self.payload.state() != payload_before
        ):
            raise RuntimeError("staff pickup changed unrelated robot/object/payload state")
        evaluator.collected_cups[cup] = parked.copy()
        self.events.append(receipt)
        return copy.deepcopy(receipt)
