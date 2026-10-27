import json
import math
from pathlib import Path

import pytest

from cap_harness.realman.estimates import cavity_volume_m3, estimate_parameters, pour_envelope


@pytest.fixture
def config():
    return json.loads(Path("configs/realman/coffee_scene.json").read_text())


def test_cavity_subtracts_floor_and_wall():
    assert cavity_volume_m3(0.04, 0.04, 0.1, 0.002, 0.003) == pytest.approx(
        math.pi * 0.038**2 * 0.097
    )
    with pytest.raises(ValueError):
        cavity_volume_m3(0.04, 0.04, 0.1, 0.05, 0.003)


def test_user_load_priors_and_units(config):
    e = estimate_parameters(config)
    assert 210 < e["cup_capacity_ml"] < 213
    assert e["cup_target_ml"] == pytest.approx(e["cup_capacity_ml"] * 0.8)
    assert e["target_water_mass_g"] == pytest.approx(e["cup_target_ml"])
    assert e["pot_empty_mass_kg"] == 0.396
    assert e["spring_reference_force_n"] == pytest.approx(
        0.5 * e["pot_full_reference_mass_kg"] * 9.80665
    )
    assert e["spring_stiffness_n_per_m"] * 0.008 == pytest.approx(e["spring_reference_force_n"])
    assert e["dispenser_reference_pull_force_n"] == pytest.approx(
        2 * e["cup_capacity_ml"] / 1000 * 9.80665
    )
    assert not e["hardware_force_limit_qualified"]


def test_force_is_not_stiffness(config):
    before = estimate_parameters(config)
    config["force_priors"]["spring_reference_compression_m"] /= 2
    after = estimate_parameters(config)
    assert after["spring_reference_force_n"] == before["spring_reference_force_n"]
    assert after["spring_stiffness_n_per_m"] == 2 * before["spring_stiffness_n_per_m"]


def test_pour_checks_do_not_claim_volume_or_collision_success(config):
    good = pour_envelope(config, spout_in_cup_m=[0, 0, 0.095], tilt_rate_rad_s=0.05)
    assert (
        good["spout_projection_inside_rim"]
        and good["spout_height_within_band"]
        and good["tilt_rate_within_limit"]
    )
    assert not good["fill_target_verified"] and not good["full_body_clearance_checked"]
    bad = pour_envelope(config, spout_in_cup_m=[0.04, 0, 0.15], tilt_rate_rad_s=0.3)
    assert (
        not bad["spout_projection_inside_rim"]
        and not bad["spout_height_within_band"]
        and not bad["tilt_rate_within_limit"]
    )
