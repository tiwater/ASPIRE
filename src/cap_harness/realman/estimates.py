"""Geometric/load priors, not qualified live force limits or fluid predictions."""

from __future__ import annotations

import math


def cavity_volume_m3(bottom_radius, top_radius, height, wall, bottom_thickness):
    """Circular frustum volume above the floor, matching the model's linear wall."""
    values = (bottom_radius, top_radius, height, wall, bottom_thickness)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("cavity dimensions must be positive and finite")
    if wall >= min(bottom_radius, top_radius) or bottom_thickness >= height:
        raise ValueError("wall/floor consumes cavity")
    h = height - bottom_thickness
    rb = bottom_radius - wall + (top_radius - bottom_radius) * bottom_thickness / height
    rt = top_radius - wall
    return math.pi * h * (rb * rb + rb * rt + rt * rt) / 3


def estimate_parameters(config):
    c, p, priors = config["cup"], config["pot"], config["force_priors"]
    rho, g = priors["water_density_kg_m3"], priors["gravity_m_s2"]
    if not all(math.isfinite(v) and v > 0 for v in (rho, g, p["mass_kg"])):
        raise ValueError("invalid mass/density/gravity")
    fraction = config["task"]["fill_fraction"]
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("invalid fill fraction")
    cup = cavity_volume_m3(
        c["bottom_radius_m"], c["rim_radius_m"], c["height_m"], c["wall_m"], c["wall_m"]
    )
    pot = cavity_volume_m3(
        p["radius_m"], p["radius_m"] * 0.92, p["height_m"] - 0.008, p["wall_m"], p["wall_m"]
    )
    full_pot_mass = p["mass_kg"] + rho * pot
    spring = priors["spring_full_pot_weight_fraction"] * full_pot_mass * g
    pull = priors["dispenser_full_cups_of_water"] * rho * cup * g
    reference, preload = priors["spring_reference_compression_m"], config["outlet"]["preload_n"]
    if (
        not all(math.isfinite(v) for v in (spring, pull, reference, preload))
        or not 0 < reference <= config["outlet"]["travel_m"]
        or not 0 <= preload < spring
        or pull <= 0
    ):
        raise ValueError("invalid force prior or spring reference displacement")
    return {
        "cup_capacity_ml": cup * 1e6,
        "cup_target_ml": cup * fraction * 1e6,
        "target_water_mass_g": cup * fraction * rho * 1000,
        "pot_capacity_ml": pot * 1e6,
        "pot_empty_mass_kg": p["mass_kg"],
        "pot_full_reference_mass_kg": full_pot_mass,
        "spring_reference_force_n": spring,
        "spring_reference_compression_m": reference,
        "spring_stiffness_n_per_m": (spring - preload) / reference,
        "dispenser_reference_pull_force_n": pull,
        "capacity_source": "circular_frustum_photo_geometry_with_wall_and_floor_subtracted",
        "force_source": "user_approved_initial_estimates",
        "hardware_force_limit_qualified": False,
        "actual_pour_volume_observed": False,
    }


def pour_envelope(config, *, spout_in_cup_m, tilt_rate_rad_s):
    """Check a measured spout point in cup axes (+Z up, origin at cup bottom).

    Full-body clearance and fill feedback must be established separately.
    This pure check never commands or stops hardware.
    """
    if len(spout_in_cup_m) != 3 or not all(
        math.isfinite(v) for v in (*spout_in_cup_m, tilt_rate_rad_s)
    ):
        raise ValueError("need finite spout position and tilt rate")
    c, rule = config["cup"], config["pour_geometry"]
    x, y, z = spout_in_cup_m
    allowed_radius = c["rim_radius_m"] - c["wall_m"] - rule["rim_clearance_m"]
    gap = z - c["height_m"]
    return {
        "spout_projection_inside_rim": math.hypot(x, y) <= allowed_radius,
        "spout_height_within_band": rule["min_tip_above_rim_m"]
        <= gap
        <= rule["max_tip_above_rim_m"],
        "tilt_rate_within_limit": abs(tilt_rate_rad_s) <= math.radians(rule["max_tilt_rate_deg_s"]),
        "tip_above_rim_m": gap,
        "full_body_clearance_checked": False,
        "spill_free_verified": False,
        "fill_target_verified": False,
    }
