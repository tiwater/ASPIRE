"""Opt-in simulation diagnostics, not qualified hardware overload limits."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PadLoadLimits:
    warning_n: float = 25.0
    sustained_n: float = 30.0
    peak_n: float = 35.0
    sustained_s: float = 0.1
    clear_n: float = 29.0
    minimum_pad_n: float = 0.02
    maximum_sample_gap_s: float = 0.002
    asymmetric_s: float = 0.0

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for k, v in vars(self).items() if k != "asymmetric_s"):
            raise ValueError("force diagnostic parameters must be finite and positive")
        if not np.isfinite(self.asymmetric_s) or not 0 <= self.asymmetric_s <= 0.05:
            raise ValueError("invalid asymmetric load observation window")
        if not self.minimum_pad_n < self.warning_n < self.clear_n < self.sustained_n < self.peak_n:
            raise ValueError("invalid force diagnostic ordering")


class PadLoadMonitor:
    def __init__(self, limits=None):
        self.limits = limits or PadLoadLimits()
        self.last_time = None
        self.over_since = None
        self.asymmetric_since = None

    def update(self, time_s, forces):
        f = np.asarray(forces, dtype=float)
        if (
            not np.isfinite(time_s)
            or f.shape != (2,)
            or not np.isfinite(f).all()
            or np.any(f < 0)
            or (self.last_time is not None and time_s <= self.last_time)
        ):
            raise ValueError("need increasing finite time and two nonnegative forces")
        p = self.limits
        failures = []
        if self.last_time is not None and time_s - self.last_time > p.maximum_sample_gap_s + 1e-9:
            failures.append("force_observation_gap")
        self.last_time = time_s
        high = float(f.max())
        if high > p.sustained_n and self.over_since is None:
            self.over_since = time_s
        elif high <= p.clear_n:
            self.over_since = None
        elapsed = 0 if self.over_since is None else time_s - self.over_since
        if high > p.peak_n:
            failures.append("pad_peak_load")
        if self.over_since is not None and elapsed >= p.sustained_s - 1e-9:
            failures.append("pad_sustained_load")
        if high > p.warning_n and float(f.min()) <= p.minimum_pad_n:
            if self.asymmetric_since is None:
                self.asymmetric_since = time_s
            if time_s - self.asymmetric_since >= p.asymmetric_s - 1e-9:
                failures.append("pad_asymmetric_overload")
        else:
            self.asymmetric_since = None
        return {
            "warning": high > p.warning_n,
            "overload_duration_s": elapsed,
            "asymmetric_duration_s": 0
            if self.asymmetric_since is None
            else time_s - self.asymmetric_since,
            "failures": failures,
        }


class SupportContactContinuity:
    """Bounded state inference after established support, never a new contact claim."""

    def __init__(self, maximum_gap_s, maximum_excursion_m):
        if not (
            np.isfinite(maximum_gap_s)
            and 0 < maximum_gap_s <= 0.05
            and np.isfinite(maximum_excursion_m)
            and 0 < maximum_excursion_m <= 0.001
        ):
            raise ValueError("invalid bounded support continuity window")
        self.maximum_gap_s = maximum_gap_s
        self.maximum_excursion_m = maximum_excursion_m
        self.last = {}

    def clear(self):
        self.last.clear()

    def update(self, name, now, position, *, anchored, geometry_valid, established):
        position = np.asarray(position, dtype=float)
        if not np.isfinite(now) or position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("invalid support observation")
        previous = self.last.get(name)
        if previous and now < previous[0]:
            raise ValueError("support time went backward")
        if not geometry_valid:
            self.last.pop(name, None)
            return {"retained": False, "inferred": False, "gap_s": None}
        if anchored:
            self.last[name] = (now, position.copy())
            return {"retained": True, "inferred": False, "gap_s": 0.0}
        gap = None if previous is None else now - previous[0]
        valid = (
            established
            and previous is not None
            and gap <= self.maximum_gap_s + 1e-9
            and np.linalg.norm(position - previous[1]) <= self.maximum_excursion_m
        )
        return {"retained": bool(valid), "inferred": bool(valid), "gap_s": gap}
