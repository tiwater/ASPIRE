import numpy as np
import pytest

from cap_harness.realman.coffee_service import MotionTiming


@pytest.mark.parametrize("baseline,scale", [(10, 4), (2, 2), (0.1, 4), (10, 1)])
def test_timing_keeps_endpoints_monotonicity_and_requested_peak_speed(baseline, scale):
    plan = MotionTiming(baseline, scale)
    t = np.linspace(0, plan.duration_seconds, 10001)
    progress = np.array([plan.fraction(float(x)) for x in t])
    assert progress[0] == 0 and progress[-1] == 1
    assert np.all(np.diff(progress) >= 0)
    velocity = np.diff(progress) / np.diff(t)
    assert velocity.max() <= scale / baseline + 1e-9
    if scale > 1:
        assert plan.duration_seconds > baseline / scale
        assert velocity[0] < velocity.max() * 0.001
        assert velocity[-1] < velocity.max() * 0.001
    else:
        np.testing.assert_allclose(progress, t / baseline)
