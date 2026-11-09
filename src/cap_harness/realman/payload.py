"""Rigid equivalent liquid payloads, not free-surface or flow simulation.

The upright fill distribution stays attached to the vessel when tilted. This
provides mass/COM/inertia loads; it does not predict spillage or actual volume.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    center_m: np.ndarray
    inertia_kg_m2: np.ndarray

    def as_dict(self):
        return {
            "mass_kg": float(self.mass_kg),
            "center_m": self.center_m.tolist(),
            "inertia_kg_m2": self.inertia_kg_m2.tolist(),
        }


class AxisymmetricCavity:
    """Integrate horizontal disks in a piecewise linear *inner* radius profile."""

    def __init__(self, profile_z_radius_m, density_kg_m3=1000.0):
        profile = np.asarray(profile_z_radius_m, dtype=float)
        if (
            profile.ndim != 2
            or profile.shape[1] != 2
            or len(profile) < 2
            or not np.isfinite(profile).all()
            or np.any(profile[:, 1] <= 0)
            or np.any(np.diff(profile[:, 0]) <= 0)
            or not np.isfinite(density_kg_m3)
            or density_kg_m3 <= 0
        ):
            raise ValueError("need increasing finite heights, positive radii and density")
        self.profile = profile.copy()
        self.density = float(density_kg_m3)
        self.segments = []
        self.capacity_m3 = 0.0
        for (z0, r0), (z1, r1) in zip(profile, profile[1:]):
            h = z1 - z0
            r = Polynomial([r0, (r1 - r0) / h])
            z = Polynomial([z0, 1])
            integrals = [(p * np.pi).integ() for p in [r * r, z * r * r, z * z * r * r, r**4]]
            total = np.array([p(h) - p(0) for p in integrals])
            self.segments.append((h, integrals, total))
            self.capacity_m3 += total[0]

    def properties(self, volume_m3):
        if not np.isfinite(volume_m3) or volume_m3 < 0 or volume_m3 > self.capacity_m3 + 1e-14:
            raise ValueError("volume outside cavity")
        if volume_m3 == 0:
            return MassProperties(0.0, np.zeros(3), np.zeros((3, 3)))
        remaining = min(float(volume_m3), self.capacity_m3)
        moments = np.zeros(4)
        for height, integrals, total in self.segments:
            if remaining >= total[0]:
                moments += total
                remaining -= total[0]
            else:
                h = brentq(
                    lambda h: integrals[0](h) - integrals[0](0) - remaining, 0, height, xtol=1e-14
                )
                moments += [p(h) - p(0) for p in integrals]
                break
        v, z1, z2, radial4 = moments
        mass = self.density * v
        center_z = z1 / v
        transverse = self.density * (radial4 / 4 + z2 - z1 * z1 / v)
        axial = self.density * radial4 / 2
        return MassProperties(
            float(mass), np.array([0.0, 0.0, center_z]), np.diag([transverse, transverse, axial])
        )


def coffee_cavities(config):
    """Use the same cavity bounds/radii as estimate_parameters."""
    p, c = config["pot"], config["cup"]
    profile = np.asarray(p["body_profile_z_radius_m"], dtype=float)
    low, high = p["wall_m"], p["height_m"] - 0.006
    heights = np.r_[low, profile[(profile[:, 0] > low) & (profile[:, 0] < high), 0], high]
    radii = np.interp(heights, profile[:, 0], profile[:, 1]) - p["wall_m"]
    pot = np.column_stack((heights, radii))
    roll = c.get("rolled_rim_radius_m", 0)
    top, rtop, bottom, wall = (
        c["height_m"] - roll,
        c["rim_radius_m"] - roll,
        c["bottom_radius_m"],
        c["wall_m"],
    )
    cup = [[wall, bottom - wall + (rtop - bottom) * wall / top], [top, rtop - wall]]
    rho = config["force_priors"]["water_density_kg_m3"]
    return {"pot": AxisymmetricCavity(pot, rho), "cup_0": AxisymmetricCavity(cup, rho)}


def combine_mass_properties(*parts):
    if not parts or any(not np.isfinite(p.mass_kg) or p.mass_kg < 0 for p in parts):
        raise ValueError("need nonnegative finite masses")
    mass = sum(p.mass_kg for p in parts)
    if mass <= 0:
        raise ValueError("total mass must be positive")
    center = sum((p.mass_kg * p.center_m for p in parts), np.zeros(3)) / mass
    inertia = np.zeros((3, 3))
    for p in parts:
        delta = p.center_m - center
        inertia += p.inertia_kg_m2 + p.mass_kg * (
            np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)
        )
    return MassProperties(mass, center, inertia)


def body_mass_properties(model, name):
    body = model.body(name)
    rotation = Rotation.from_quat(body.iquat[[1, 2, 3, 0]]).as_matrix()
    return MassProperties(
        float(body.mass[0]), body.ipos.copy(), rotation @ np.diag(body.inertia) @ rotation.T
    )


def _principal_inertia(properties):
    if (
        not np.isfinite(properties.mass_kg)
        or properties.mass_kg <= 0
        or np.shape(properties.center_m) != (3,)
        or np.shape(properties.inertia_kg_m2) != (3, 3)
        or not np.isfinite(properties.center_m).all()
        or not np.isfinite(properties.inertia_kg_m2).all()
        or not np.allclose(properties.inertia_kg_m2, properties.inertia_kg_m2.T, rtol=0, atol=1e-12)
    ):
        raise ValueError("invalid mass, center or inertia tensor")
    eigenvalues, vectors = np.linalg.eigh(properties.inertia_kg_m2)
    if (
        not np.isfinite(eigenvalues).all()
        or np.any(eigenvalues <= 0)
        or eigenvalues[-1] > eigenvalues[0] + eigenvalues[1] + 1e-12
    ):
        raise ValueError("invalid physical inertia")
    if np.linalg.det(vectors) < 0:
        vectors[:, 0] *= -1
    return eigenvalues, vectors


def set_body_properties(model, name, properties):
    """Set inertial inputs only; caller must rebuild constants before stepping."""
    eigenvalues, vectors = _principal_inertia(properties)
    b = model.body(name)
    if model.body_simple[b.id] != 0:
        raise ValueError("payload update requires a general body; recompile simple bodies")
    b.mass[:] = properties.mass_kg
    b.ipos[:] = properties.center_m
    b.inertia[:] = eigenvalues
    b.iquat[:] = Rotation.from_matrix(vectors).as_quat()[[3, 0, 1, 2]]
    # Compiler shortcuts referring to the old inertia frame must not move
    # collision geometry/sites when that frame changes. Body-relative poses
    # are unchanged. This implementation targets the general free vessels.
    model.body_sameframe[b.id] = 0
    for prefix in ["geom", "site"]:
        owners = getattr(model, prefix + "_bodyid")
        flags = getattr(model, prefix + "_sameframe")
        flags[(owners == b.id) & np.isin(flags, [2, 4])] = 0
    _refit_body_bvh(model, b.id)


def _refit_body_bvh(model, body_id):
    """Refit collision bounds in the new inertia frame, without changing geoms.

    MuJoCo 3.3 uses xipos/ximat for intermediate body BVH nodes. mj_setConst
    does not refit these boxes. Stale boxes can cull real pad contacts even
    though geom_xpos and the mass matrix remain correct. Refit from leaf
    geometry each time to avoid accumulating conservative-box inflation.
    """
    start, count = int(model.body_bvhadr[body_id]), int(model.body_bvhnum[body_id])
    if not count:
        return
    nodes = model.bvh_nodeid[start : start + count]
    leaves = np.flatnonzero(nodes >= 0)
    geom_ids = nodes[leaves]
    body = model.body(body_id)
    inertia_rotation = Rotation.from_quat(body.iquat[[1, 2, 3, 0]]).as_matrix()
    geom_rotations = Rotation.from_quat(model.geom_quat[geom_ids][:, [1, 2, 3, 0]]).as_matrix()
    boxes = model.geom_aabb[geom_ids]
    centers = model.geom_pos[geom_ids] + np.einsum("nij,nj->ni", geom_rotations, boxes[:, :3])
    relative_rotations = np.einsum("ij,njk->nik", inertia_rotation.T, geom_rotations)
    model.bvh_aabb[start + leaves, :3] = (centers - body.ipos) @ inertia_rotation
    model.bvh_aabb[start + leaves, 3:] = (
        np.einsum("nij,nj->ni", np.abs(relative_rotations), boxes[:, 3:]) + 1e-12
    )
    # Compiler preorder guarantees children follow parents; child indices are
    # relative to this body's root. Tree topology and collision masks stay intact.
    for i in range(count - 1, -1, -1):
        if nodes[i] >= 0:
            continue
        children = model.bvh_child[start + i]
        bounds = model.bvh_aabb[start + children[children >= 0]]
        low = np.min(bounds[:, :3] - bounds[:, 3:], axis=0)
        high = np.max(bounds[:, :3] + bounds[:, 3:], axis=0)
        model.bvh_aabb[start + i, :3] = (low + high) / 2
        model.bvh_aabb[start + i, 3:] = (high - low) / 2


def refresh_inertial_constants(model, data):
    """mj_setConst uses scratch data; preserve the live integration state exactly.

    Do not reset the running simulator or reuse a checkpoint with different
    model/payload metadata. Parameter changes here omit fluid momentum flux.
    """
    import mujoco

    scratch = mujoco.MjData(model)
    mujoco.mj_setConst(model, scratch)
    mujoco.mj_forward(model, data)


class PrescribedPayloadTransfer:
    """Conserved load transfer for simulation, explicitly not a flow sensor.

    The caller must establish geometric eligibility before every small transfer.
    Inertia and COM are recomputed from the remaining volumes. This deliberately
    omits free-surface motion, slosh, jet reaction and momentum-flux dynamics.
    A target reached here must never establish actual-pour or no-spill success.
    """

    def __init__(self, model, data, config, dry_properties, pot_volume_m3):
        self.model, self.data = model, data
        self.cavities = coffee_cavities(config)
        self.dry = {
            name: MassProperties(
                float(p["mass_kg"]), np.array(p["center_m"]), np.array(p["inertia_kg_m2"])
            )
            for name, p in dry_properties.items()
        }
        if set(self.dry) != {"pot", "cup_0"}:
            raise ValueError("need both dry vessel inertias")
        self.cavities["pot"].properties(pot_volume_m3)
        self.initial_volume_m3 = float(pot_volume_m3)
        self.cup_volume_m3 = 0.0
        # Validate the model matches the declared initial payload; do not
        # silently add a second load to an already-loaded XML model.
        for name, volume in [("pot", pot_volume_m3), ("cup_0", 0.0)]:
            expected = combine_mass_properties(
                self.dry[name], self.cavities[name].properties(volume)
            )
            actual = body_mass_properties(model, name)
            if (
                not np.isclose(actual.mass_kg, expected.mass_kg, rtol=0, atol=1e-10)
                or not np.allclose(actual.center_m, expected.center_m, rtol=0, atol=1e-10)
                or not np.allclose(actual.inertia_kg_m2, expected.inertia_kg_m2, rtol=0, atol=1e-10)
            ):
                raise ValueError("initial payload metadata/model mismatch")

    def transfer(self, amount_m3, *, geometry_eligible=False):
        if not geometry_eligible:
            raise ValueError("prescribed transfer requires current geometric eligibility")
        if not np.isfinite(amount_m3) or amount_m3 <= 0:
            raise ValueError("need a positive finite transfer")
        cup = self.cup_volume_m3 + amount_m3
        pot = self.initial_volume_m3 - cup
        # Compute/validate both results before any mutation.
        properties = {
            name: combine_mass_properties(self.dry[name], self.cavities[name].properties(volume))
            for name, volume in [("pot", pot), ("cup_0", cup)]
        }
        for name, value in properties.items():
            if self.model.body_simple[self.model.body(name).id] != 0:
                raise ValueError("payload requires general bodies")
            _principal_inertia(value)
        for name, value in properties.items():
            set_body_properties(self.model, name, value)
        refresh_inertial_constants(self.model, self.data)
        self.cup_volume_m3 = cup
        return self.state()

    def restore_payload_state(self, state):
        """Restore a recorded load without claiming a new transfer occurred.

        The caller must separately verify the model/checkpoint hashes and restore
        the matching integration state. Validate conservation and both recorded
        inertias against the configured vessels before changing either body.
        """
        for key, expected in self.state().items():
            if key not in {"pot_volume_m3", "cup_volume_m3", "body_properties"}:
                if state.get(key) != expected:
                    raise ValueError(f"payload checkpoint mismatch: {key}")
        cup, pot = float(state["cup_volume_m3"]), float(state["pot_volume_m3"])
        if not np.isfinite([cup, pot]).all() or not np.isclose(
            cup + pot, self.initial_volume_m3, rtol=0, atol=1e-12
        ):
            raise ValueError("payload checkpoint does not conserve volume")
        properties = {
            name: combine_mass_properties(self.dry[name], self.cavities[name].properties(volume))
            for name, volume in [("pot", pot), ("cup_0", cup)]
        }
        recorded = state["body_properties"]
        if set(recorded) != set(properties):
            raise ValueError("payload checkpoint vessel mismatch")
        for name, value in properties.items():
            for key, expected in value.as_dict().items():
                actual = np.asarray(recorded[name][key], dtype=float)
                if actual.shape != np.asarray(expected).shape or not np.allclose(
                    actual, expected, rtol=0, atol=1e-10
                ):
                    raise ValueError(f"payload checkpoint inertia mismatch: {name}/{key}")
            if self.model.body_simple[self.model.body(name).id] != 0:
                raise ValueError("payload requires general bodies")
            _principal_inertia(value)
        for name, value in properties.items():
            set_body_properties(self.model, name, value)
        refresh_inertial_constants(self.model, self.data)
        self.cup_volume_m3 = cup
        return self.state()

    def state(self):
        return {
            "mode": "prescribed_mass_transfer_rigid_payload",
            "pot_volume_m3": self.initial_volume_m3 - self.cup_volume_m3,
            "cup_volume_m3": self.cup_volume_m3,
            "initial_volume_m3": self.initial_volume_m3,
            "body_properties": {
                name: body_mass_properties(self.model, name).as_dict() for name in self.dry
            },
            "actual_volume_observed": False,
            "free_surface_and_flow_simulated": False,
            "spill_free_verified": False,
            "momentum_flux_modeled": False,
        }
