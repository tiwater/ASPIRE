"""Static contact-wrench feasibility, not an actuator or material certification.

Elliptic friction rays give an inner approximation; a box relaxation supplies
an optimistic lower force bound. Contact
positions, normals and friction are supplied measurements/model assumptions.
Joint torques, compliance, pressure limits and dynamic stability are not modeled.
"""

import numpy as np
from scipy.optimize import linprog


def minimum_pad_load(
    contacts, external_wrench, reference_m, *, friction_model="elliptic_inscribed"
):
    """Minimize the largest pad's summed normal load at fixed contact geometry.

    Wrenches are world-frame [force N, moment Nm] about reference_m. Each contact
    supplies a frame whose rows are the normal on the object and two tangents.
    Torsional friction is a length (Nm per N), not a sliding coefficient.
    A feasible result concerns only this idealized static contact model.
    """
    wrench, reference = np.asarray(external_wrench, float), np.asarray(reference_m, float)
    if (
        wrench.shape != (6,)
        or reference.shape != (3,)
        or not np.isfinite(np.r_[wrench, reference]).all()
    ):
        raise ValueError("need finite wrench and reference point")
    if not contacts:
        raise ValueError("need contact geometry")
    pads = sorted({c["pad"] for c in contacts})
    if friction_model == "outer_box":
        directions = np.array([[x, y, z] for x in [-1, 1] for y in [-1, 1] for z in [-1, 1]])
    elif friction_model == "elliptic_inscribed":
        directions = np.array(
            [
                [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)]
                for phi in np.linspace(-np.pi / 3, np.pi / 3, 5)
                for theta in np.arange(8) * np.pi / 4
            ]
            + [[0, 0, -1], [0, 0, 1]]
        )
    else:
        raise ValueError("unknown friction model")
    rays = len(directions)
    count = len(contacts) * rays + 1
    equilibrium = np.zeros((6, count))
    limits, rhs = [], []
    normals = {pad: np.zeros(count) for pad in pads}
    for i, c in enumerate(contacts):
        frame, position = np.asarray(c["frame"], float), np.asarray(c["position_m"], float)
        mu, spin = float(c["sliding_friction"]), float(c["torsional_friction_m"])
        if (
            frame.shape != (3, 3)
            or position.shape != (3,)
            or not np.isfinite(np.r_[frame.ravel(), position, mu, spin]).all()
            or mu < 0
            or spin < 0
            or not np.allclose(frame @ frame.T, np.eye(3), atol=1e-8)
        ):
            raise ValueError("invalid contact frame or friction")
        offset = rays * i
        for j, direction in enumerate(directions):
            force = frame[0] + mu * (direction[0] * frame[1] + direction[1] * frame[2])
            moment = np.cross(position - reference, force) + spin * direction[2] * frame[0]
            equilibrium[:, offset + j] = np.r_[force, moment]
            normals[c["pad"]][offset + j] = 1
    for normal in normals.values():
        bound = normal.copy()
        bound[-1] = -1
        limits.append(bound)
        rhs.append(0)
    objective = np.zeros(count)
    objective[-1] = 1
    fit = linprog(
        objective,
        A_ub=limits,
        b_ub=rhs,
        A_eq=equilibrium,
        b_eq=-wrench,
        bounds=(0, None),
        method="highs",
    )
    result = {
        "static_model_feasible": bool(fit.success),
        "friction_model": friction_model,
        "force_bound": "optimistic_lower"
        if friction_model == "outer_box"
        else "inner_approximation_upper",
        "solver_message": fit.message,
        "hardware_qualified": False,
        "dynamic_stability_verified": False,
        "contact_count": len(contacts),
    }
    if fit.success:
        result.update(
            {
                "minimum_max_pad_normal_n": float(fit.x[-1]),
                "pad_normal_n": {pad: float(row @ fit.x) for pad, row in normals.items()},
                "equilibrium_residual": (equilibrium @ fit.x + wrench).tolist(),
            }
        )
    return result
