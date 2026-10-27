"""Object-level acceptance contract for coffee serving, independent of execution backend."""

from __future__ import annotations

from dataclasses import dataclass
import math

STAGES = (
    (
        "localize_station",
        (
            "base_stationary",
            "table_frame_current",
            "objects_localized",
            "cup_capacity_estimate_available",
        ),
    ),
    ("take_cup_right", ("cup_in_right_gripper", "single_cup_separated", "remaining_cups_retained")),
    ("grasp_pot_left", ("pot_handle_in_left_gripper", "cup_in_right_gripper")),
    (
        "withdraw_pot",
        (
            "pot_clear_of_outlet",
            "pot_clear_of_machine",
            "pot_upright",
            "pot_handle_in_left_gripper",
            "cup_in_right_gripper",
        ),
    ),
    (
        "align_for_pour",
        ("spout_over_cup", "pour_clearance", "pot_handle_in_left_gripper", "cup_in_right_gripper"),
    ),
    (
        "pour",
        (
            "fill_target_reached",
            "flow_stopped",
            "no_spill",
            "pot_handle_in_left_gripper",
            "cup_in_right_gripper",
        ),
    ),
    (
        "place_cup",
        (
            "cup_supported_by_table",
            "cup_upright",
            "right_gripper_released",
            "pot_handle_in_left_gripper",
            "flow_stopped",
        ),
    ),
    (
        "return_pot",
        (
            "pot_on_hotplate",
            "lid_outlet_aligned",
            "pot_upright",
            "pot_handle_in_left_gripper",
            "cup_supported_by_table",
        ),
    ),
    (
        "release_pot",
        (
            "pot_stable_after_release",
            "left_gripper_released",
            "cup_supported_by_table",
            "cup_upright",
        ),
    ),
)


@dataclass(frozen=True)
class Evidence:
    value: bool | None
    observed_s: float
    source: str

    def __post_init__(self):
        if self.value is not None and type(self.value) is not bool:
            raise ValueError("evidence value must be true, false or unknown")
        if not math.isfinite(self.observed_s) or self.observed_s < 0 or not self.source.strip():
            raise ValueError("evidence needs a valid timestamp and traceable source")


class CoffeeProgress:
    """A supervisor must supply visual/force/object evidence, not a bare motor ACK.

    Each stage needs fresh evidence. Re-localize after base/body movement; keep
    the stage active on unknown/false predicates. This class does not fabricate
    evidence from simulator ground truth or execute any motion.
    """

    def __init__(self):
        self.index = 0
        self.stage_started_s = 0.0
        self.history = []

    @property
    def complete(self):
        return self.index == len(STAGES)

    @property
    def stage(self):
        return "complete" if self.complete else STAGES[self.index][0]

    def advance(self, evidence: dict[str, Evidence], *, now_s: float, max_age_s: float) -> dict:
        if not math.isfinite(now_s) or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("invalid evidence time budget")
        if self.complete:
            return {"advanced": False, "stage": self.stage, "unresolved": []}
        missing = []
        for name in STAGES[self.index][1]:
            item = evidence.get(name)
            if (
                item is None
                or item.value is not True
                or not self.stage_started_s <= item.observed_s <= now_s
                or now_s - item.observed_s > max_age_s
            ):
                missing.append(name)
        if missing:
            return {"advanced": False, "stage": self.stage, "unresolved": missing}
        self.history.append(
            {
                "stage": self.stage,
                "verified_s": now_s,
                "sources": {k: evidence[k].source for k in STAGES[self.index][1]},
            }
        )
        self.index += 1
        self.stage_started_s = now_s
        return {"advanced": True, "stage": self.stage, "unresolved": []}


def fill_target_ml(capacity_ml: float, fraction=0.8) -> float:
    if (
        not math.isfinite(capacity_ml)
        or capacity_ml <= 0
        or not math.isfinite(fraction)
        or not 0 < fraction < 1
    ):
        raise ValueError("need positive capacity and a fraction between zero and one")
    return capacity_ml * fraction
