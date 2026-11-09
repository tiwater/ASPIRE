"""Object-level simulation evidence. Ground truth is a grader, never real perception.

Fluid predicates remain unknown; this evaluator cannot certify coffee volume or
whole-task success. Contact grasps require a sustained hold after displacement.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .diagnostics import SupportContactContinuity
from .grasp_geometry import grasp_geometry
from .platform import is_machine_support_geom
from .task import STAGES, Evidence


class CoffeeGeometryEvaluator:
    def __init__(self, adapter, scene_config, *, hold_s=0.3, max_sample_gap_s=0.05):
        if not np.isfinite([hold_s, max_sample_gap_s]).all() or not 0 < max_sample_gap_s <= hold_s:
            raise ValueError("invalid evidence window")
        self.adapter = adapter
        self.config = scene_config
        self.hold_s, self.max_sample_gap_s = hold_s, max_sample_gap_s
        self.cups = [f"cup_{i}" for i in range(scene_config["dispenser"]["stack_count"])]
        if not self.cups:
            raise ValueError("need a target cup")
        self.initial = {n: adapter.data.body(n).xpos.copy() for n in ["pot", *self.cups]}
        self.since = {}
        self.relative = {}
        self.last_time = None
        support = scene_config.get("simulation_diagnostics", {}).get("rack_support")
        self.support_continuity = SupportContactContinuity(**support) if support else None
        self.retention_established = False

    def update(self):
        a, cfg = self.adapter, self.config
        m, d, mj = a.model, a.data, a.mj
        now = float(d.time)
        if self.last_time is not None and now < self.last_time:
            raise ValueError("simulation reset requires a new evaluator")
        if self.last_time is not None and now - self.last_time > self.max_sample_gap_s + 1e-9:
            self.since.clear()
            self.relative.clear()
            self.retention_established = False
            if self.support_continuity:
                self.support_continuity.clear()
        self.last_time = now
        evidence = {
            name: Evidence(None, now, "simulation:unimplemented_or_unobserved")
            for _, names in STAGES
            for name in names
        }

        def put(name, value):
            evidence[name] = Evidence(
                bool(value), now, "simulation:object_geometry_and_contact_history"
            )

        def sustained(name, value):
            if not value:
                self.since.pop(name, None)
                return False
            self.since.setdefault(name, now)
            return now - self.since[name] >= self.hold_s - 1e-9

        def upright(name):
            return d.body(name).xmat.reshape(3, 3)[2, 2] > np.cos(np.deg2rad(8))

        def slow(name):
            velocity = np.zeros(6)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, m.body(name).id, velocity, 0)
            return np.linalg.norm(velocity[:3]) < 0.15 and np.linalg.norm(velocity[3:]) < 0.015

        contacts = []
        graph = {}
        for index, c in enumerate(d.contact[: d.ncon]):
            force = np.zeros(6)
            mj.mj_contactForce(m, d, index, force)
            if force[0] <= 0.001:
                continue
            g1, g2 = m.geom(c.geom1).name, m.geom(c.geom2).name
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            contacts.append((g1, g2, b1, b2))
            graph.setdefault(b1, set()).add(b2)
            graph.setdefault(b2, set()).add(b1)

        def geom_contact(body, geom):
            return any(
                (b1 == body and g2 == geom) or (b2 == body and g1 == geom)
                for g1, g2, b1, b2 in contacts
            )

        def tool_touch(body, arm):
            # All tool links, not just pads, count against release.
            prefix = arm[0] + "tool_"
            return any(
                (b1 == body and b2.startswith(prefix)) or (b2 == body and b1.startswith(prefix))
                for _, _, b1, b2 in contacts
            )

        def held(body, arm, handle_only=False):
            pads = set()
            for g1, g2, b1, b2 in contacts:
                for obj, other, object_geom in [(b1, g2, g1), (b2, g1, g2)]:
                    if obj == body and (
                        not handle_only or object_geom.startswith("handle_collision_")
                    ):
                        if other in {f"{arm}_left_pad_contact", f"{arm}_right_pad_contact"}:
                            pads.add(other)
            frame = np.asarray(grasp_geometry(m, d, arm)["T_world_grasp"])
            relative = np.eye(4)
            relative[:3, :3] = frame[:3, :3].T @ d.body(body).xmat.reshape(3, 3)
            relative[:3, 3] = frame[:3, :3].T @ (d.body(body).xpos - frame[:3, 3])
            key = body + "_held"
            old = self.relative.get(key)
            stable = old is None or (
                np.linalg.norm(relative[:3, 3] - old[:3, 3]) < 0.003
                and np.linalg.norm(
                    Rotation.from_matrix(relative[:3, :3] @ old[:3, :3].T).as_rotvec()
                )
                < np.deg2rad(3)
            )
            displaced = np.linalg.norm(d.body(body).xpos - self.initial[body]) > 0.03
            contact_valid = len(pads) == 2
            if handle_only and cfg.get("grasp_evidence", {}).get("left_pot") == "upper_bar_bearing":
                from .upper_handle import upper_handle_loads

                contact_valid = upper_handle_loads(m, d, cfg["pot"], arm=arm)["opposed_contact"]
            valid = contact_valid and stable and displaced
            if not valid:
                self.relative.pop(key, None)
            elif old is None:
                self.relative[key] = relative
            return sustained(key, valid)

        target = self.cups[0]
        remaining = self.cups[1:]
        retained = True
        support_rows = {}
        for name in remaining:
            seen, todo = {target}, [name]
            anchored = False
            while todo:
                body = todo.pop()
                if body in seen:
                    continue
                seen.add(body)
                if body.startswith("retention_finger") or body == "cup_dispenser":
                    anchored = True
                    break
                # Support must pass through retained cups, not the removed one or robot.
                todo.extend(
                    n
                    for n in graph.get(body, ())
                    if n in remaining or n.startswith("retention_finger") or n == "cup_dispenser"
                )
            geometry_valid = (
                upright(name)
                and slow(name)
                and np.linalg.norm(d.body(name).xpos - self.initial[name]) < 0.015
            )
            if self.support_continuity:
                status = self.support_continuity.update(
                    name,
                    now,
                    d.body(name).xpos,
                    anchored=anchored,
                    geometry_valid=geometry_valid,
                    established=self.retention_established,
                )
            else:
                status = {
                    "retained": bool(anchored and geometry_valid),
                    "inferred": False,
                    "gap_s": 0.0 if anchored else None,
                }
            support_rows[name] = {
                **status,
                "anchored_now": bool(anchored),
                "geometry_valid": bool(geometry_valid),
            }
            retained &= status["retained"]
        put("remaining_cups_retained", sustained("retained", retained))
        self.retention_established = evidence["remaining_cups_retained"].value is True
        if self.retention_established and any(s["inferred"] for s in support_rows.values()):
            evidence["remaining_cups_retained"] = Evidence(
                True, now, "simulation:bounded_support_state_inference_not_current_contact"
            )
        rack_contacts = any(
            n in remaining or n.startswith("retention_finger") or n == "cup_dispenser"
            for n in graph.get(target, ())
        )
        separated = (
            not rack_contacts and np.linalg.norm(d.body(target).xpos - self.initial[target]) > 0.04
        )
        put("single_cup_separated", separated and evidence["remaining_cups_retained"].value)
        put("cup_in_right_gripper", held(target, "right"))
        put("pot_handle_in_left_gripper", held("pot", "left", handle_only=True))
        put("cup_upright", upright(target))
        put("pot_upright", upright("pot"))
        table_support = geom_contact(target, "tabletop") and slow(target) and upright(target)
        put("cup_supported_by_table", sustained("cup_table", table_support))
        disk_contact = geom_contact("pot", "hotplate")
        seat_contact = disk_contact
        if "hotplate_recess" in cfg["machine"]:
            # Legacy predicate denotes seating in the bay, not necessarily direct
            # disk contact: a wider flat pot floor can bridge a recessed disk.
            seat_contact = any(
                (b1 == "pot" and is_machine_support_geom(g2))
                or (b2 == "pot" and is_machine_support_geom(g1))
                for g1, g2, b1, b2 in contacts
            )
        hotplate = seat_contact and slow("pot") and upright("pot")
        hotplate &= np.linalg.norm(d.body("pot").xpos[:2] - d.geom("hotplate").xpos[:2]) < 0.015
        put("pot_on_hotplate", sustained("pot_hotplate", hotplate))
        put(
            "right_gripper_released",
            evidence["cup_supported_by_table"].value and not tool_touch(target, "right"),
        )
        put(
            "left_gripper_released",
            evidence["pot_on_hotplate"].value and not tool_touch("pot", "left"),
        )
        put(
            "pot_stable_after_release",
            sustained("pot_released", hotplate and not tool_touch("pot", "left")),
        )
        pot_neighbors = graph.get("pot", set())
        delta = d.body("pot").xpos[:2] - d.body("spring_outlet").xpos[:2]
        put(
            "pot_clear_of_outlet",
            "spring_outlet" not in pot_neighbors and np.linalg.norm(delta) > 0.02,
        )
        put(
            "pot_clear_of_machine",
            not {"coffee_machine", "spring_outlet"}.intersection(pot_neighbors)
            and np.linalg.norm(delta) > cfg["pot"]["radius_m"] + 0.02,
        )
        lid_rise = max(
            rise for _, rise in cfg["pot"].get("lid_boss_profile_radius_rise_m", [[0, 0]])
        )
        lid = d.body("pot").xpos + d.body("pot").xmat.reshape(3, 3) @ [
            0,
            0,
            cfg["pot"]["height_m"] + lid_rise,
        ]
        outlet = d.body("spring_outlet").xpos
        put(
            "lid_outlet_aligned",
            np.linalg.norm((lid - outlet)[:2]) < 0.004
            and abs(lid[2] - outlet[2]) < 0.01
            and upright("pot"),
        )
        spout_local = d.body(target).xmat.reshape(3, 3).T @ (
            d.site("pot_spout").xpos - d.body(target).xpos
        )
        inside = (
            np.linalg.norm(spout_local[:2])
            < cfg["cup"]["rim_radius_m"]
            - max(cfg["cup"]["wall_m"], 2 * cfg["cup"].get("rolled_rim_radius_m", 0))
            - 0.003
        )
        clearance = float(spout_local[2] - cfg["cup"]["height_m"])
        put("spout_over_cup", inside and upright(target))
        put(
            "pour_clearance",
            inside
            and upright(target)
            and -0.005 <= clearance <= 0.04
            and target not in pot_neighbors,
        )
        failures = []
        collected = getattr(self, "collected_cups", {})
        if any(
            d.body(n).xpos[2] < cfg["table"]["top_height_m"] - 0.04
            for n in ["pot", *(cup for cup in self.cups if cup not in collected)]
        ):
            failures.append("object_below_table")
        if (
            not np.isfinite(d.qpos).all()
            or not np.isfinite(d.qvel).all()
            or np.any(d.warning.number)
        ):
            failures.append("simulation_invalid")
        return {
            "evidence": evidence,
            "failures": failures,
            "simulation_only": True,
            "oracle_grader": True,
            "heating_disk_contact_now": bool(disk_contact),
            "rack_support_diagnostics": {
                "cups": support_rows,
                "all_supported_now": all(
                    s["anchored_now"] and s["geometry_valid"] for s in support_rows.values()
                ),
                "inference_enabled": self.support_continuity is not None,
            },
            "fluid_mode": "not_simulated",
            "coffee_task_success": None,
            "spout_height_above_rim_m": clearance,
            "threshold_basis": "engineering acceptance priors; not measured hardware tolerances",
        }
