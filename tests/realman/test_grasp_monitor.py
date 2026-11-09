import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cap_harness.realman.grasp_monitor import HandleGraspMonitor


def test_rotation_about_handle_does_not_become_fictitious_handle_translation():
    handle = np.array([0.116, 0, 0.081])
    reference = np.eye(4)
    reference[:3, 3] = -handle
    rotated = reference.copy()
    rotated[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    rotated[:3, 3] = -rotated[:3, :3] @ handle
    monitor = HandleGraspMonitor(reference, handle)
    result = monitor.update(0, rotated, [25, 29])
    assert result["pot_origin_translation_m"] > 0.003
    assert result["handle_translation_m"] < 1e-12
    assert result["relative_rotation_deg"] == pytest.approx(2)
    assert result["failures"] == []


def test_real_handle_translation_and_rotation_are_independently_limited():
    monitor = HandleGraspMonitor(np.eye(4), [0, 0, 0])
    shifted = np.eye(4)
    shifted[0, 3] = 0.006
    assert monitor.update(0, shifted, [25, 29])["failures"] == ["handle_translation"]
    turned = np.eye(4)
    turned[:3, :3] = Rotation.from_euler("z", 6, degrees=True).as_matrix()
    assert monitor.update(0.001, turned, [25, 29])["failures"] == ["relative_rotation"]


def test_transient_unload_recovers_but_alternating_unloaded_pads_do_not_reset_timer():
    monitor = HandleGraspMonitor(np.eye(4), [0, 0, 0])
    assert monitor.update(0, np.eye(4), [25, 0])["failures"] == []
    assert monitor.update(0.001, np.eye(4), [25, 29])["failures"] == []
    for i in range(2, 52):
        result = monitor.update(i / 1000, np.eye(4), [25, 0] if i % 2 else [0, 29])
        assert result["failures"] == []
    result = monitor.update(0.052, np.eye(4), [25, 0])
    assert result["failures"] == ["persistent_unilateral_unload"]


def test_both_pads_unloaded_stop_earlier_and_missing_observations_are_not_success():
    monitor = HandleGraspMonitor(np.eye(4), [0, 0, 0])
    for i in range(10):
        assert monitor.update(i / 1000, np.eye(4), [0, 0])["failures"] == []
    assert monitor.update(0.010, np.eye(4), [0, 0])["failures"] == ["persistent_bilateral_unload"]
    assert monitor.update(0.020, np.eye(4), [25, 29])["failures"] == ["observation_gap"]
    with pytest.raises(ValueError):
        monitor.update(0.019, np.eye(4), [25, 29])


def test_nonfinite_force_cannot_appear_to_recover_contact():
    monitor = HandleGraspMonitor(np.eye(4), [0, 0, 0])
    with pytest.raises(ValueError):
        monitor.update(0, np.eye(4), [np.nan, 29])
