import numpy as np
import pytest

from cap_harness.realman.tracking import bounded_rotation_integral, bumpless_rotation_limit


def test_long_settle_inside_pose_acceptance_does_not_wind_up_grip():
    feedback = np.zeros(3)
    for _ in range(1500):
        feedback = bounded_rotation_integral(feedback, np.deg2rad([0.2, 0.2, 0]))
    np.testing.assert_array_equal(feedback, np.zeros(3))


def test_persistent_tracking_error_cannot_accumulate_to_five_degree_grasp_limit():
    feedback = np.zeros(3)
    for _ in range(1000):
        feedback = bounded_rotation_integral(feedback, np.deg2rad([1, 1, 0]))
    assert np.linalg.norm(feedback) == pytest.approx(np.deg2rad(1))
    assert np.linalg.norm(feedback) < np.deg2rad(5)
    held = bounded_rotation_integral(feedback, np.zeros(3))
    np.testing.assert_array_equal(held, feedback)


def test_invalid_feedback_cannot_reach_control():
    with pytest.raises(ValueError):
        bounded_rotation_integral(np.zeros(3), [0, np.nan, 0])


@pytest.mark.parametrize(
    "recorded_deg,requested_deg,minimum_old_jump_deg",
    [
        ([-3.35266517992681, -2.7187669212639767, 0.07770229455073899], 2, 2),
        ([1.9769383230881867, 2.001167140497084, -0.32504423181225556], 1, 1.8),
    ],
)
def test_support_transition_preserves_recorded_compensation_then_unwinds_gradually(
    recorded_deg, requested_deg, minimum_old_jump_deg
):
    # Recorded pre-lift and nozzle-recontact corrections. Losing platform support
    # must not abruptly delete established compensation when changing bounds.
    previous = np.deg2rad(recorded_deg)
    clipped = bounded_rotation_integral(previous, np.zeros(3), maximum_deg=requested_deg)
    assert np.rad2deg(np.linalg.norm(previous - clipped)) > minimum_old_jump_deg
    bound = bumpless_rotation_limit(previous, requested_deg=requested_deg)
    held = bounded_rotation_integral(previous, np.zeros(3), maximum_deg=bound)
    np.testing.assert_allclose(held, previous, atol=1e-15)
    error = -previous / np.linalg.norm(previous) * np.deg2rad(0.6)
    for _ in range(100):
        bound = bumpless_rotation_limit(held, requested_deg=requested_deg)
        updated = bounded_rotation_integral(held, error, maximum_deg=bound)
        assert np.rad2deg(np.linalg.norm(updated - held)) <= 0.024 + 1e-12
        assert np.linalg.norm(updated) <= np.linalg.norm(held)
        held = updated
    assert np.rad2deg(np.linalg.norm(held)) < requested_deg


def test_invalid_transition_cannot_supply_a_controller_bound():
    with pytest.raises(ValueError):
        bumpless_rotation_limit([0, np.nan, 0], requested_deg=2)
    with pytest.raises(ValueError):
        bumpless_rotation_limit(np.zeros(3), requested_deg=-1)
