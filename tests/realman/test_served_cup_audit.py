"""Reject disturbed served cups even if a run's reported predicates are true."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[2] / "tools/realman/audit_served_cups.py"
spec = importlib.util.spec_from_file_location("served_cup_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

CONFIG = {
    "table": {"center_xy_m": [1, 0], "size_m": [0.4, 0.8, 0.02]},
    "cup": {"bottom_radius_m": 0.025},
}
BODIES = {"cup_base": "cup_0", "tabletop": "table", "pot_glass": "pot"}
SUPPORT = {"geom1": "tabletop", "geom2": "cup_base", "normal_force_n": 1.6}


def inspect(position=(0.87, -0.09, 0.75), up=1, contacts=None):
    return module.inspect_sample(
        position,
        up,
        [0.87, -0.09, 0.75],
        [SUPPORT] if contacts is None else contacts,
        BODIES,
        "cup_0",
        CONFIG,
    )


def test_table_supported_stationary_cup_passes():
    assert inspect()["passed"]


def test_pot_strike_is_rejected_despite_table_support_and_no_displacement():
    strike = {"geom1": "pot_glass", "geom2": "cup_base", "normal_force_n": 0.56}
    result = inspect(contacts=[SUPPORT, strike])
    assert result["checks"]["table_support"]
    assert result["checks"]["position_preserved"]
    assert not result["passed"]
    assert result["unexpected_contacts"] == [strike]


@pytest.mark.parametrize(
    "kwargs,failed",
    [
        ({"contacts": []}, "table_support"),
        ({"position": [0.87, -0.102, 0.75]}, "position_preserved"),
        ({"position": [0.83, -0.09, 0.75]}, "footprint_margin"),
        ({"up": np.cos(np.deg2rad(9))}, "upright"),
        ({"position": [np.nan, -0.09, 0.75]}, "finite_pose"),
    ],
)
def test_lost_relation_fails_independently(kwargs, failed):
    result = inspect(**kwargs)
    assert not result["passed"]
    assert not result["checks"][failed]
