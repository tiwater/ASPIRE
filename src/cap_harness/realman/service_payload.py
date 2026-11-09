"""Conserved prescribed payload for distinct cups across serving/refill cycles."""

from __future__ import annotations

import copy

import numpy as np

from .payload import (
    MassProperties,
    _principal_inertia,
    body_mass_properties,
    coffee_cavities,
    combine_mass_properties,
    refresh_inertial_constants,
    set_body_properties,
)


class ServicePayload:
    def __init__(self, model, data, config):
        self.model, self.data = model, data
        self.cups = tuple(f"cup_{i}" for i in range(config["dispenser"]["stack_count"]))
        self.active_cup = self.cups[0]
        cavities = coffee_cavities(config)
        self.cavities = {"pot": cavities["pot"], **dict.fromkeys(self.cups, cavities["cup_0"])}
        dry = config["payload"]["dry"]
        self.dry = {}
        for name in ["pot", *self.cups]:
            x = dry["pot" if name == "pot" else "cup_0"]
            self.dry[name] = MassProperties(
                x["mass_kg"], np.array(x["center_m"]), np.array(x["inertia_kg_m2"])
            )
        self.initial_volume_m3 = config["payload"]["pot_initial_ml"] / 1e6
        self.refilled_volume_m3 = 0.0
        self.volumes = {"pot": self.initial_volume_m3, **dict.fromkeys(self.cups, 0.0)}
        self.events = []
        for name, volume in self.volumes.items():
            expected = self._properties(name, volume)
            actual = body_mass_properties(model, name)
            if (
                not np.isclose(actual.mass_kg, expected.mass_kg, rtol=0, atol=1e-10)
                or not np.allclose(actual.center_m, expected.center_m, rtol=0, atol=1e-10)
                or not np.allclose(actual.inertia_kg_m2, expected.inertia_kg_m2, rtol=0, atol=1e-10)
            ):
                raise ValueError(f"initial payload mismatch: {name}")

    def _properties(self, name, volume):
        return combine_mass_properties(self.dry[name], self.cavities[name].properties(volume))

    @property
    def cup_volume_m3(self):
        return self.volumes[self.active_cup]

    @property
    def pot_volume_m3(self):
        return self.volumes["pot"]

    def select_cup(self, name):
        if name not in self.cups or self.volumes[name] != 0:
            raise ValueError("select a distinct empty cup")
        self.active_cup = name

    def _apply(self, updates):
        properties = {n: self._properties(n, v) for n, v in updates.items()}
        for n, value in properties.items():
            if self.model.body_simple[self.model.body(n).id] != 0:
                raise ValueError("payload update requires general bodies")
            _principal_inertia(value)
        for n, value in properties.items():
            set_body_properties(self.model, n, value)
        refresh_inertial_constants(self.model, self.data)
        self.volumes.update(updates)

    def transfer(self, amount_m3, *, geometry_eligible=False):
        if not geometry_eligible or not np.isfinite(amount_m3) or amount_m3 <= 0:
            raise ValueError("positive prescribed transfer requires current geometric eligibility")
        if amount_m3 > self.pot_volume_m3:
            raise ValueError("insufficient pot volume")
        self._apply(
            {"pot": self.pot_volume_m3 - amount_m3, self.active_cup: self.cup_volume_m3 + amount_m3}
        )

    def refill(self, amount_m3, *, staff_confirmed=False, pot_supported_released=False):
        if not staff_confirmed or not pot_supported_released:
            raise ValueError("external refill requires confirmation and independent pot support")
        if not np.isfinite(amount_m3) or amount_m3 <= 0:
            raise ValueError("invalid refill amount")
        self._apply({"pot": self.pot_volume_m3 + amount_m3})
        self.refilled_volume_m3 += amount_m3
        self.events.append(
            {
                "event": "external_staff_refill",
                "time_s": float(self.data.time),
                "added_m3": amount_m3,
                "flow_simulated": False,
            }
        )

    def state(self):
        total = sum(self.volumes.values())
        if abs(total - self.initial_volume_m3 - self.refilled_volume_m3) > 1e-12:
            raise RuntimeError("multi-cup payload conservation failed")
        return {
            "mode": "prescribed_mass_transfer_rigid_payload_multi_cup",
            "active_cup": self.active_cup,
            "pot_volume_m3": self.pot_volume_m3,
            "cup_volume_m3": self.cup_volume_m3,
            "vessel_volumes_m3": self.volumes.copy(),
            "initial_volume_m3": self.initial_volume_m3,
            "external_refill_volume_m3": self.refilled_volume_m3,
            "body_properties": {n: body_mass_properties(self.model, n).as_dict() for n in self.dry},
            "external_events": self.events.copy(),
            "actual_volume_observed": False,
            "free_surface_and_flow_simulated": False,
            "spill_free_verified": False,
            "momentum_flux_modeled": False,
        }

    def restore_payload_state(self, state):
        """Restore a diagnostic checkpoint's loads without creating a transfer.

        The caller verifies the model and checkpoint hashes and restores the
        matching integration state separately. Validate the complete volume,
        external refill ledger and inertia records before changing any body.
        This does not qualify a resumed run as a continuous service.
        """
        mutable = {
            "active_cup",
            "pot_volume_m3",
            "cup_volume_m3",
            "vessel_volumes_m3",
            "external_refill_volume_m3",
            "body_properties",
            "external_events",
        }
        for key, expected in self.state().items():
            if key not in mutable and state.get(key) != expected:
                raise ValueError(f"payload checkpoint mismatch: {key}")
        volumes = state["vessel_volumes_m3"]
        if set(volumes) != set(self.volumes) or state["active_cup"] not in self.cups:
            raise ValueError("payload checkpoint vessel mismatch")
        volumes = {name: float(value) for name, value in volumes.items()}
        refill = float(state["external_refill_volume_m3"])
        if not np.isfinite([*volumes.values(), refill]).all() or min(*volumes.values(), refill) < 0:
            raise ValueError("invalid checkpoint volume")
        if (
            abs(sum(volumes.values()) - self.initial_volume_m3 - refill) > 1e-12
            or state["pot_volume_m3"] != volumes["pot"]
            or state["cup_volume_m3"] != volumes[state["active_cup"]]
        ):
            raise ValueError("payload checkpoint does not conserve volume")
        events = copy.deepcopy(state["external_events"])
        previous_time, added = -1.0, 0.0
        for event in events:
            amount, time = float(event["added_m3"]), float(event["time_s"])
            if (
                event["event"] != "external_staff_refill"
                or event["flow_simulated"] is not False
                or not np.isfinite([amount, time]).all()
                or amount <= 0
                or time < 0
                or time < previous_time
            ):
                raise ValueError("invalid checkpoint refill event")
            added += amount
            previous_time = time
        if abs(added - refill) > 1e-12:
            raise ValueError("checkpoint refill ledger mismatch")
        properties = {name: self._properties(name, volume) for name, volume in volumes.items()}
        recorded = state["body_properties"]
        if set(recorded) != set(properties):
            raise ValueError("payload checkpoint inertia vessel mismatch")
        for name, value in properties.items():
            for key, expected in value.as_dict().items():
                actual = np.asarray(recorded[name][key], dtype=float)
                if actual.shape != np.asarray(expected).shape or not np.allclose(
                    actual, expected, rtol=0, atol=1e-10
                ):
                    raise ValueError(f"payload checkpoint inertia mismatch: {name}/{key}")
        self._apply(volumes)
        self.active_cup = state["active_cup"]
        self.refilled_volume_m3 = refill
        self.events = events
        return self.state()
