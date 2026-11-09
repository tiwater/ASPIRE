import json
from pathlib import Path

import numpy as np
import pytest

from cap_harness.realman.service_evaluation import next_table_slot


def test_front_row_slots_follow_translated_table_and_leave_rim_spacing():
    config = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    config["table"]["center_xy_m"] = [1.1110533060313998, 0.2]
    config["table"]["size_m"] = [0.6, 1.2, 0.035]
    slots = np.array([next_table_slot(config, i) for i in range(4)])
    center = np.array(config["table"]["center_xy_m"])
    half = np.array(config["table"]["size_m"][:2]) / 2
    margin = half - np.abs(slots - center) - config["cup"]["bottom_radius_m"]
    assert margin.min() >= 0.035 - 1e-12
    assert slots[:, 0].min() > 0.8110533060313998
    # Keep the pot's near-centre transit corridor empty until the last cup.
    pitch = 2 * config["cup"]["rim_radius_m"] + 0.02
    np.testing.assert_allclose(slots[:, 1], center[1] - pitch * np.array([1, 2, 3, 0]))
    assert (
        np.linalg.norm(np.diff(slots, axis=0), axis=1).min()
        >= 2 * config["cup"]["rim_radius_m"] + 0.019
    )
    with pytest.raises(ValueError, match="no remaining"):
        next_table_slot(config, 4)
