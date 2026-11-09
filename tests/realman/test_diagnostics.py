import numpy as np
import pytest

from cap_harness.realman.diagnostics import PadLoadLimits, PadLoadMonitor, SupportContactContinuity


def test_modest_bilateral_excess_warns_without_becoming_grasp_success():
    result = PadLoadMonitor().update(0, [20.8, 25.05])
    assert result == {
        "warning": True,
        "overload_duration_s": 0,
        "asymmetric_duration_s": 0,
        "failures": [],
    }


def test_peak_and_asymmetric_loads_stop_without_delay():
    assert "pad_peak_load" in PadLoadMonitor().update(0, [20, 35.01])["failures"]
    assert "pad_asymmetric_overload" in PadLoadMonitor().update(0, [30.6, 0])["failures"]


def test_sustained_overload_and_hysteresis_cannot_chatter_away_failure():
    monitor = PadLoadMonitor()
    for i in range(100):
        r = monitor.update(i / 1000, [20, 31 if i % 2 == 0 else 29.5])
        assert not r["failures"]
    assert "pad_sustained_load" in monitor.update(0.100, [20, 31])["failures"]
    assert not monitor.update(0.101, [20, 29])["failures"]
    assert monitor.update(0.102, [20, 31])["overload_duration_s"] == 0


def test_missing_or_invalid_force_samples_are_not_tolerated():
    monitor = PadLoadMonitor()
    monitor.update(0, [20, 26])
    assert "force_observation_gap" in monitor.update(0.01, [20, 26])["failures"]
    with pytest.raises(ValueError):
        monitor.update(0.02, [np.nan, 26])


def observe(monitor, time, *, anchored, established=True, valid=True, position=(0, 0, 0)):
    return monitor.update(
        "cup", time, position, anchored=anchored, geometry_valid=valid, established=established
    )


def test_contact_gap_is_limited_in_time_distance_and_prior_establishment():
    m = SupportContactContinuity(0.04, 0.0005)
    assert not observe(m, 0, anchored=False)["retained"]
    assert observe(m, 0.01, anchored=True)["retained"]
    assert not observe(m, 0.02, anchored=False, established=False)["retained"]
    assert observe(m, 0.03, anchored=False)["inferred"]
    assert not observe(m, 0.04, anchored=False, position=(0, 0, -0.0006))["retained"]
    assert not observe(m, 0.051, anchored=False)["retained"]
    assert observe(m, 0.06, anchored=True)["retained"]
    assert not observe(m, 0.07, anchored=False, valid=False)["retained"]
    assert not observe(m, 0.08, anchored=False)["retained"]
    observe(m, 0.09, anchored=True)
    m.clear()
    assert not observe(m, 0.10, anchored=False)["retained"]


def test_measured_short_pad_gap_recovers_but_persistent_asymmetry_stops():
    m = PadLoadMonitor(PadLoadLimits(asymmetric_s=0.01))
    assert not m.update(0, [28, 0])["failures"]
    assert not m.update(0.001, [28, 0])["failures"]
    assert not m.update(0.002, [23, 24])["failures"]
    for i in range(3, 13):
        assert not m.update(i / 1000, [28, 0])["failures"]
    assert "pad_asymmetric_overload" in m.update(0.013, [28, 0])["failures"]
    assert (
        "pad_peak_load"
        in PadLoadMonitor(PadLoadLimits(asymmetric_s=0.01)).update(0, [36, 0])["failures"]
    )
