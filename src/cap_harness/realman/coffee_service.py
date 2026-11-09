"""Object-evidence transitions for the simulation coffee service.

This supervisor selects business stages; it neither moves a robot nor establishes
task success from an action receipt. Volumes are the simulation's prescribed
payload inventory, not observations of real coffee flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math


class CoffeeAction(str, Enum):
    PREPARE = "prepare"
    TAKE_POT = "take_pot"
    SERVE_CUP = "serve_cup"
    RETURN_POT = "return_pot"
    WAIT_REFILL = "wait_refill"
    EXIT = "exit"
    IDLE = "idle"

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class ServiceEvidence:
    predicates: dict[str, bool | None]
    remaining_pot_ml: float | None = None
    cup_id: str | None = None
    cup_volume_ml: float | None = None

    def require(self, *names):
        missing = [name for name in names if self.predicates.get(name) is not True]
        if missing:
            raise ValueError(f"unverified service relations: {missing}")


@dataclass
class CoffeeService:
    capacity_ml: float
    remaining_pot_ml: float
    pending_cups: int = 0
    state: str = "standing"
    exit_requested: bool = False
    served_cups: dict[str, float] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self._volume(self.capacity_ml, positive=True)
        self._volume(self.remaining_pot_ml)
        if self.state != "standing" or self.pending_cups != 0 or self.served_cups:
            raise ValueError("new service starts standing with no completed actions")

    @staticmethod
    def _volume(value, *, positive=False):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("finite observed simulation volume required")
        if value < 0 or (positive and value <= 0):
            raise ValueError("invalid volume")
        return float(value)

    def request_cups(self, count):
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("request a positive integer number of cups")
        if self.exit_requested:
            raise ValueError("exit requested; create a new session after exit")
        self.pending_cups += count

    def request_exit(self):
        self.exit_requested = True

    @property
    def enough_for_one_cup(self):
        # Reserve uses full geometric capacity, while each serving fills to 80%.
        # Exact equality is sufficient; tolerance only covers float roundoff.
        return self.remaining_pot_ml >= self.capacity_ml - 1e-9

    def next_action(self):
        if self.state == "standing":
            return (
                CoffeeAction.PREPARE
                if self.pending_cups and not self.exit_requested
                else CoffeeAction.IDLE
            )
        if self.state == "holding_pot":
            if self.exit_requested or not self.pending_cups or not self.enough_for_one_cup:
                return CoffeeAction.RETURN_POT
            return CoffeeAction.SERVE_CUP
        if self.exit_requested:
            return CoffeeAction.EXIT
        if not self.enough_for_one_cup:
            return CoffeeAction.WAIT_REFILL
        return CoffeeAction.TAKE_POT if self.pending_cups else CoffeeAction.IDLE

    def complete(self, action: CoffeeAction, evidence: ServiceEvidence):
        action = CoffeeAction(action)
        if action != self.next_action() or action in {CoffeeAction.IDLE, CoffeeAction.WAIT_REFILL}:
            raise ValueError("stage completion is out of order")
        old_state = self.state
        if action == CoffeeAction.PREPARE:
            evidence.require(
                "working_body_pose",
                "arms_pregrasp",
                "both_grippers_empty",
                "pot_on_hotplate",
                "rack_inventory_supported",
            )
            self.state = "ready"
        elif action == CoffeeAction.TAKE_POT:
            evidence.require(
                "pot_handle_in_left_gripper",
                "pot_clear_of_machine",
                "pot_clear_of_outlet",
                "pot_upright",
                "right_gripper_empty",
                "rack_inventory_supported",
            )
            self.state = "holding_pot"
        elif action == CoffeeAction.SERVE_CUP:
            evidence.require(
                "pot_handle_in_left_gripper",
                "pot_upright",
                "cup_supported_by_table",
                "cup_upright",
                "right_gripper_released",
                "remaining_cups_retained",
                "previous_served_cups_preserved",
            )
            if not evidence.cup_id or evidence.cup_id in self.served_cups:
                raise ValueError("serving requires a new identified cup")
            volume = self._volume(evidence.cup_volume_ml)
            remaining = self._volume(evidence.remaining_pot_ml)
            if abs(volume - self.capacity_ml * 0.8) > 1e-6:
                raise ValueError("serving has not reached 80% prescribed capacity")
            if abs(self.remaining_pot_ml - remaining - volume) > 1e-6:
                raise ValueError("payload volume is not conserved")
            self.served_cups[evidence.cup_id] = volume
            self.remaining_pot_ml = remaining
            self.pending_cups -= 1
        elif action == CoffeeAction.RETURN_POT:
            evidence.require(
                "pot_on_hotplate",
                "lid_outlet_aligned",
                "pot_stable_after_release",
                "pot_upright",
                "both_grippers_empty",
                "previous_served_cups_preserved",
            )
            self.state = "ready" if self.enough_for_one_cup else "waiting_refill"
        elif action == CoffeeAction.EXIT:
            evidence.require(
                "standing_body_pose",
                "arms_hanging",
                "both_grippers_empty",
                "pot_stable_after_release",
                "previous_served_cups_preserved",
            )
            self.state = "standing"
        self.events.append(
            {
                "action": action.value,
                "from": old_state,
                "to": self.state,
                "remaining_pot_ml": self.remaining_pot_ml,
                "pending_cups": self.pending_cups,
                "cup_id": evidence.cup_id,
            }
        )

    def acknowledge_refill(self, remaining_ml, evidence: ServiceEvidence):
        """Explicit external refill event; never silently top up a held pot."""
        if self.state not in {"ready", "waiting_refill"} or self.exit_requested:
            raise ValueError("refill requires a supported, released pot before exit")
        evidence.require(
            "staff_refill_confirmed",
            "pot_on_hotplate",
            "pot_stable_after_release",
            "both_grippers_empty",
            "previous_served_cups_preserved",
        )
        remaining = self._volume(remaining_ml)
        if remaining < self.remaining_pot_ml:
            raise ValueError("refill cannot reduce the inventory")
        self.remaining_pot_ml = remaining
        self.state = "ready" if self.enough_for_one_cup else "waiting_refill"
        self.events.append({"action": "external_refill", "remaining_pot_ml": remaining})


@dataclass(frozen=True)
class MotionSpeedProfile:
    """Scale motion only. Physical timestep, guard thresholds and holds stay fixed."""

    free_transport: float = 1.0
    contact_motion: float = 1.0

    def __post_init__(self):
        for value, maximum in [(self.free_transport, 4), (self.contact_motion, 2)]:
            if isinstance(value, bool) or not math.isfinite(value) or not 1 <= value <= maximum:
                raise ValueError("speed factor outside requested experiment envelope")

    def factor(self, *, independently_clear: bool):
        return self.free_transport if independently_clear else self.contact_motion


@dataclass(frozen=True)
class MotionTiming:
    """Accelerated reference with bounded speed and finite acceleration ramps.

    Scale is the peak reference speed relative to the original segment. Ramps
    add time; it is not a claim that total elapsed time is divided by scale.
    """

    baseline_seconds: float
    scale: float = 1.0
    requested_ramp_seconds: float = 0.5

    def __post_init__(self):
        if not math.isfinite(self.baseline_seconds) or self.baseline_seconds <= 0:
            raise ValueError("positive baseline duration required")
        if not math.isfinite(self.scale) or not 1 <= self.scale <= 4:
            raise ValueError("invalid motion speed scale")
        if not math.isfinite(self.requested_ramp_seconds) or self.requested_ramp_seconds < 0:
            raise ValueError("nonnegative acceleration ramp required")

    @property
    def nominal_seconds(self):
        return self.baseline_seconds / self.scale

    @property
    def ramp_seconds(self):
        return min(self.requested_ramp_seconds, self.nominal_seconds) if self.scale > 1 else 0.0

    @property
    def duration_seconds(self):
        return self.nominal_seconds + self.ramp_seconds

    def fraction(self, elapsed_seconds):
        if not math.isfinite(elapsed_seconds):
            raise ValueError("finite elapsed time required")
        t = max(0.0, min(self.duration_seconds, elapsed_seconds))
        nominal, ramp = self.nominal_seconds, self.ramp_seconds
        if not ramp:
            return t / nominal
        if t < ramp:
            return t * t / (2 * ramp * nominal)
        if t <= nominal:
            return (t - ramp / 2) / nominal
        return 1 - (self.duration_seconds - t) ** 2 / (2 * ramp * nominal)
