import numpy as np
import pytest

from cap_harness.realman.supported_slide import supported_slide_settled, supported_yaw_integral


def sample(**overrides):
    values = dict(
        position_error_m=0.0005,
        yaw_error_deg=0.2,
        tilt_deg=1.5,
        linear_speed_m_s=0.0001,
        angular_speed_rad_s=0.001,
        upward_support_n=4.0,
    )
    return supported_slide_settled(**(values | overrides))


def test_actual_platform_support_allows_small_tilt_but_not_free_space():
    assert sample()
    assert not sample(upward_support_n=0)


@pytest.mark.parametrize(
    "measurement,value",
    [
        ("position_error_m", 0.003),
        ("yaw_error_deg", 0.6),
        ("tilt_deg", 2.1),
        ("linear_speed_m_s", 0.004),
        ("angular_speed_rad_s", 0.04),
    ],
)
def test_support_does_not_hide_other_waypoint_failures(measurement, value):
    assert not sample(**{measurement: value})


def test_invalid_support_is_not_accepted():
    with pytest.raises(ValueError):
        sample(upward_support_n=float("nan"))


def test_heading_correction_preserves_support_constrained_tilt_and_is_bounded():
    original = np.deg2rad([-1.35, -1.44, 0.25])
    correction = original.copy()
    for _ in range(500):
        correction = supported_yaw_integral(correction, np.deg2rad(0.7))
    np.testing.assert_array_equal(correction[:2], original[:2])
    assert correction[2] == pytest.approx(np.deg2rad(1))
    np.testing.assert_array_equal(supported_yaw_integral(correction, 0.001), correction)


def test_explicit_larger_tilt_window_still_requires_support_and_heading():
    assert not sample(tilt_deg=2.005)
    assert sample(tilt_deg=2.005, max_tilt_deg=5)
    assert sample(tilt_deg=4.99, max_tilt_deg=5)
    assert not sample(tilt_deg=5, max_tilt_deg=5)
    assert not sample(tilt_deg=2.005, max_tilt_deg=5, upward_support_n=0)
    assert not sample(tilt_deg=2.005, max_tilt_deg=5, yaw_error_deg=0.6)
    assert not sample(tilt_deg=2.005, max_tilt_deg=5, position_error_m=0.003)


@pytest.mark.parametrize("limit", [0, -1, 8.1, float("nan"), float("inf")])
def test_invalid_tilt_limit(limit):
    with pytest.raises(ValueError):
        sample(max_tilt_deg=limit)


@pytest.mark.parametrize("direction", [-1, 1])
def test_larger_yaw_authority_is_gradual_bounded_and_preserves_tilt(direction):
    initial = np.deg2rad([-1.2, -0.8, direction])
    first = supported_yaw_integral(initial, direction * np.deg2rad(1.08), maximum_deg=2)
    assert abs(first[2] - initial[2]) == pytest.approx(0.04 * np.deg2rad(1.08))
    result = first
    for _ in range(200):
        result = supported_yaw_integral(result, direction * np.deg2rad(1.08), maximum_deg=2)
    np.testing.assert_array_equal(result[:2], initial[:2])
    assert result[2] == pytest.approx(direction * np.deg2rad(2))
    opposite = supported_yaw_integral(result, -direction * np.deg2rad(1), maximum_deg=2)
    assert abs(opposite[2]) < abs(result[2])


@pytest.mark.parametrize("limit", [0, -1, 5.1, float("nan"), float("inf")])
def test_invalid_yaw_authority(limit):
    with pytest.raises(ValueError):
        supported_yaw_integral(np.zeros(3), 0, maximum_deg=limit)
