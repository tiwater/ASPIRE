"""Keep cup identities and served-object invariants across multiple servings."""

from __future__ import annotations

import numpy as np

from .sim_evaluation import CoffeeGeometryEvaluator
from .task import Evidence


def rack_support_paths(graph, cups):
    """Require a path to the rack using only the explicitly eligible cup set.

    Before extraction the bottom cup is eligible; after extraction the original
    evaluator excludes it. Neither robot nor table contacts can anchor a stack.
    """
    eligible = set(cups)
    result = {}
    for name in sorted(eligible):
        seen, todo = set(), [name]
        anchored = False
        while todo:
            body = todo.pop()
            if body in seen:
                continue
            seen.add(body)
            if body.startswith("retention_finger") or body == "cup_dispenser":
                anchored = True
                break
            todo.extend(
                n
                for n in graph.get(body, ())
                if n in eligible or n.startswith("retention_finger") or n == "cup_dispenser"
            )
        result[name] = anchored
    return result


def next_table_slot(config, served_count):
    """Fill the robot-right slots first; reserve the pouring corridor for last.

    The near-centre slot intersects the pot's route on subsequent servings.
    Use it only for the rack's last cup, after that cup's pot transport ends.
    Restocking requires clearing the served row before further service.
    """
    if isinstance(served_count, bool) or not isinstance(served_count, int) or served_count < 0:
        raise ValueError("invalid served cup count")
    if served_count >= config["dispenser"]["stack_count"]:
        raise ValueError(
            "no remaining qualified table slot; clear the served row before restocking"
        )
    center = np.asarray(config["table"]["center_xy_m"], dtype=float)
    half = np.asarray(config["table"]["size_m"][:2], dtype=float) / 2
    clearance = config["cup"]["bottom_radius_m"] + 0.035
    pitch = 2 * config["cup"]["rim_radius_m"] + 0.02
    slot = (served_count + 1) % config["dispenser"]["stack_count"]
    target = center + [-half[0] + clearance, -pitch * slot]
    if np.any(np.abs(target - center) + clearance > half + 1e-12):
        raise ValueError("no remaining qualified table slot")
    return target


class ServiceGeometryEvaluator(CoffeeGeometryEvaluator):
    def __init__(self, adapter, scene_config, **kwargs):
        super().__init__(adapter, scene_config, **kwargs)
        self.all_cups = tuple(self.cups)
        self.served = {}
        self.collected_cups = {}
        self.inventory_since = {}

    def select_cup(self, name):
        if name not in self.all_cups or name in self.served:
            raise ValueError("target must be an unserved cup")
        # Each successful withdrawal advances the retained stack by one nesting
        # pitch. Establish a new per-withdrawal reference only from independently
        # verified support, rather than accumulating that intended travel against
        # the original single-cup 15 mm displacement bound.
        result = self.update()
        if result["evidence"]["rack_inventory_supported"].value is not True:
            raise ValueError("cannot select next cup before rack retention is verified")
        remaining = [n for n in self.all_cups if n not in self.served and n != name]
        for cup in [name, *remaining]:
            self.initial[cup] = self.adapter.data.body(cup).xpos.copy()
        self.cups = [name, *remaining]
        # Retain pot grasp history and original rack reference positions.
        for key in ["cup_table", name + "_held"]:
            self.since.pop(key, None)
        self.relative.pop(name + "_held", None)
        # Retained cups form a subset of the previous retained set. Preserve
        # their support histories; update() still checks every remaining member.

    def mark_served(self):
        result = self.update()
        needed = ["cup_supported_by_table", "cup_upright", "right_gripper_released"]
        if not all(result["evidence"][key].value is True for key in needed):
            raise ValueError("cup placement/release not verified")
        name = self.cups[0]
        if name in self.served:
            raise ValueError("cup already served")
        self.served[name] = self.adapter.data.body(name).xpos.copy()

    def update(self):
        previous_time = self.last_time
        result = super().update()
        a = self.adapter
        m, d, mj = a.model, a.data, a.mj
        now = float(d.time)
        if previous_time is not None and now - previous_time > self.max_sample_gap_s + 1e-9:
            self.inventory_since.clear()
        contacts = set()
        graph = {}
        tool_bodies = {arm: set() for arm in ["left", "right"]}
        for i, c in enumerate(d.contact):
            f = np.zeros(6)
            mj.mj_contactForce(m, d, i, f)
            if f[0] <= 0.001:
                continue
            g1, g2 = m.geom(c.geom1).name, m.geom(c.geom2).name
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            graph.setdefault(b1, set()).add(b2)
            graph.setdefault(b2, set()).add(b1)
            if f[0] <= 0.02:
                continue
            if g1 == "tabletop":
                contacts.add(b2)
            if g2 == "tabletop":
                contacts.add(b1)
            for arm, touched in tool_bodies.items():
                prefix = arm[0] + "tool_"
                if b1.startswith(prefix):
                    touched.add(b2)
                if b2.startswith(prefix):
                    touched.add(b1)
        unserved = set(self.all_cups) - set(self.served)
        anchored = rack_support_paths(graph, unserved)
        inventory = {}
        for name in sorted(unserved):
            velocity = np.zeros(6)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, m.body(name).id, velocity, 0)
            geometry_valid = bool(
                d.body(name).xmat[8] > np.cos(np.deg2rad(8))
                and np.linalg.norm(velocity[:3]) < 0.15
                and np.linalg.norm(velocity[3:]) < 0.015
                and np.linalg.norm(d.body(name).xpos - self.initial[name]) < 0.015
            )
            fresh = anchored[name] and geometry_valid
            if fresh:
                self.inventory_since.setdefault(name, now)
            else:
                self.inventory_since.pop(name, None)
            stable = fresh and now - self.inventory_since[name] >= self.hold_s - 1e-9
            inventory[name] = dict(
                anchored_now=anchored[name],
                geometry_valid=geometry_valid,
                supported=bool(stable),
                inferred=False,
            )
        result["rack_inventory_support_diagnostics"] = {
            "cups": inventory,
            "all_supported_now": all(
                x["anchored_now"] and x["geometry_valid"] for x in inventory.values()
            ),
        }
        # The active cup remains selected until the next serving. Once staff
        # has collected it, its off-workspace bookkeeping body is intentionally
        # below the table. Recompute this one failure over every service object,
        # excluding only served, explicitly collected cups still outside the
        # workspace. Never exempt the pot or an uncollected cup.
        half = np.asarray(self.config["table"]["size_m"][:2]) / 2
        center = np.asarray(self.config["table"]["center_xy_m"])
        collected_outside = {
            name for name in self.all_cups
            if name in self.served and name in self.collected_cups
            and np.any(np.abs(d.body(name).xpos[:2] - center) > half + .25)
        }
        below = [
            name for name in ["pot", *self.all_cups]
            if name not in collected_outside
            and d.body(name).xpos[2] < self.config["table"]["top_height_m"] - .04
        ]
        result["failures"] = [f for f in result["failures"] if f != "object_below_table"]
        if below:
            result["failures"].append("object_below_table")
        result["object_height_diagnostics"] = {
            "below_table_objects": below,
            "collected_outside_workspace": sorted(collected_outside),
        }
        result["evidence"]["rack_inventory_supported"] = Evidence(
            all(x["supported"] for x in inventory.values()),
            now,
            "simulation:all_unserved_cups_to_rack_contact_paths_and_pose_history",
        )
        preserved = True
        for name, position in self.served.items():
            if name in self.collected_cups:
                # An explicit staff event has completed delivery. Keep checking
                # that this identified body stays outside the work table and
                # away from the robot, rather than requiring table support.
                half = np.asarray(self.config["table"]["size_m"][:2]) / 2
                center = np.asarray(self.config["table"]["center_xy_m"])
                outside = np.any(np.abs(d.body(name).xpos[:2] - center) > half + .25)
                preserved &= bool(outside) and all(
                    name not in touched for touched in tool_bodies.values()
                )
                continue
            velocity = np.zeros(6)
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, m.body(name).id, velocity, 0)
            half = np.array(self.config["table"]["size_m"][:2]) / 2
            center = np.array(self.config["table"]["center_xy_m"])
            margin = (
                np.min(half - np.abs(d.body(name).xpos[:2] - center))
                - self.config["cup"]["bottom_radius_m"]
            )
            valid = (
                name in contacts
                and margin >= 0.02
                and d.body(name).xmat[8] > np.cos(np.deg2rad(8))
                and np.linalg.norm(d.body(name).xpos - position) < 0.01
                and np.linalg.norm(velocity[:3]) < 0.15
                and np.linalg.norm(velocity[3:]) < 0.015
                and all(name not in touched for touched in tool_bodies.values())
            )
            preserved &= valid
        vessels = {"pot", *self.all_cups}
        right_empty = not vessels.intersection(tool_bodies["right"])
        both_empty = right_empty and not vessels.intersection(tool_bodies["left"])
        for name, value in [
            ("previous_served_cups_preserved", preserved),
            ("right_gripper_empty", right_empty),
            ("both_grippers_empty", both_empty),
        ]:
            result["evidence"][name] = Evidence(
                bool(value), float(d.time), "simulation:identified_object_contacts_and_pose"
            )
        if not preserved:
            result["failures"].append("previous_served_cup_relation_lost")
        result["active_cup"] = self.cups[0]
        result["served_cups"] = list(self.served)
        result["collected_cups"] = list(self.collected_cups)
        result["evidence"]["served_cups_collected"] = Evidence(
            bool(self.served) and set(self.served) == set(self.collected_cups),
            now,
            "simulation:explicit_staff_collection_ledger_and_outside_workspace",
        )
        return result


def selected_cup_withdraw_clearance(data, config, target, remaining):
    cup = config["cup"]
    body = data.body(target)
    up = float(body.xmat[8])
    top = body.xpos[2] + cup["height_m"] * up + cup["rim_radius_m"] * np.sqrt(max(0, 1 - up * up))
    bottoms = []
    for name in remaining:
        other = data.body(name)
        up = float(other.xmat[8])
        bottoms.append(other.xpos[2] - cup["bottom_radius_m"] * np.sqrt(max(0, 1 - up * up)))
    return min(bottoms) - top if bottoms else float("inf")
