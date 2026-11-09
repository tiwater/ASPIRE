import pytest

from cap_harness.realman.coffee_service import (
    CoffeeAction as A,
)
from cap_harness.realman.coffee_service import (
    CoffeeService,
    MotionSpeedProfile,
    ServiceEvidence,
)


def evidence(*names, **kwargs):
    return ServiceEvidence(dict.fromkeys(names, True), **kwargs)


PREPARED = evidence(
    "working_body_pose",
    "arms_pregrasp",
    "both_grippers_empty",
    "pot_on_hotplate",
    "rack_inventory_supported",
)
TAKEN = evidence(
    "pot_handle_in_left_gripper",
    "pot_clear_of_machine",
    "pot_clear_of_outlet",
    "pot_upright",
    "right_gripper_empty",
    "rack_inventory_supported",
)
RETURNED = evidence(
    "pot_on_hotplate",
    "lid_outlet_aligned",
    "pot_stable_after_release",
    "pot_upright",
    "both_grippers_empty",
    "previous_served_cups_preserved",
)


def served(cup, remaining, volume=160):
    return evidence(
        "pot_handle_in_left_gripper",
        "pot_upright",
        "cup_supported_by_table",
        "cup_upright",
        "right_gripper_released",
        "remaining_cups_retained",
        "previous_served_cups_preserved",
        cup_id=cup,
        cup_volume_ml=volume,
        remaining_pot_ml=remaining,
    )


def holding(requests=1, remaining=400):
    s = CoffeeService(200, remaining)
    s.request_cups(requests)
    s.complete(A.PREPARE, PREPARED)
    s.complete(A.TAKE_POT, TAKEN)
    return s


def test_repeated_serving_requires_new_cups_and_explicit_refill():
    s = holding(requests=3)
    s.complete(A.SERVE_CUP, served("cup_0", 240))
    assert s.next_action() == A.SERVE_CUP
    s.complete(A.SERVE_CUP, served("cup_1", 80))
    assert s.next_action() == A.RETURN_POT
    s.complete(A.RETURN_POT, RETURNED)
    assert s.next_action() == A.WAIT_REFILL and s.pending_cups == 1
    with pytest.raises(ValueError):
        s.complete(A.TAKE_POT, TAKEN)
    s.acknowledge_refill(
        400,
        evidence(
            "staff_refill_confirmed",
            "pot_on_hotplate",
            "pot_stable_after_release",
            "both_grippers_empty",
            "previous_served_cups_preserved",
        ),
    )
    s.complete(A.TAKE_POT, TAKEN)
    s.complete(A.SERVE_CUP, served("cup_2", 240))
    s.complete(A.RETURN_POT, RETURNED)
    s.request_exit()
    s.complete(
        A.EXIT,
        evidence(
            "standing_body_pose",
            "arms_hanging",
            "both_grippers_empty",
            "pot_stable_after_release",
            "previous_served_cups_preserved",
        ),
    )
    assert s.state == "standing" and s.next_action() == A.IDLE
    assert list(s.served_cups) == ["cup_0", "cup_1", "cup_2"]


def test_exit_while_holding_requires_supported_release_first():
    s = holding()
    s.request_exit()
    assert s.next_action() == A.RETURN_POT
    with pytest.raises(ValueError):
        s.complete(A.EXIT, ServiceEvidence({}))
    s.complete(A.RETURN_POT, RETURNED)
    assert s.next_action() == A.EXIT


def test_exit_from_refill_wait_preserves_unserved_order_and_low_inventory():
    s = holding(requests=3)
    s.complete(A.SERVE_CUP, served("cup_0", 240))
    s.complete(A.SERVE_CUP, served("cup_1", 80))
    s.complete(A.RETURN_POT, RETURNED)
    assert s.next_action() == A.WAIT_REFILL and s.pending_cups == 1
    s.request_exit()
    assert s.next_action() == A.EXIT
    with pytest.raises(ValueError, match="unverified service relations"):
        s.complete(A.EXIT, evidence("standing_body_pose", "arms_hanging"))
    assert s.state == "waiting_refill" and s.pending_cups == 1
    s.complete(A.EXIT, evidence(
        "standing_body_pose", "arms_hanging", "both_grippers_empty",
        "pot_stable_after_release", "previous_served_cups_preserved",
    ))
    assert s.state == "standing" and s.next_action() == A.IDLE
    assert s.pending_cups == 1 and s.remaining_pot_ml == 80
    assert list(s.served_cups) == ["cup_0", "cup_1"]
    assert not any(e["action"] == "external_refill" for e in s.events)


def test_rejected_evidence_does_not_consume_request_or_inventory():
    s = holding(requests=2)
    for e in [served("cup_0", 200), served("cup_0", 240, volume=150), ServiceEvidence({})]:
        with pytest.raises(ValueError):
            s.complete(A.SERVE_CUP, e)
    assert s.pending_cups == 2 and s.remaining_pot_ml == 400 and not s.served_cups
    s.complete(A.SERVE_CUP, served("cup_0", 240))
    with pytest.raises(ValueError):
        s.complete(A.SERVE_CUP, served("cup_0", 80))
    assert s.pending_cups == 1 and s.remaining_pot_ml == 240


def test_refill_requires_external_confirmation_and_supported_pot():
    s = holding()
    with pytest.raises(ValueError):
        s.acknowledge_refill(1000, ServiceEvidence({"staff_refill_confirmed": True}))
    s.request_exit()
    s.complete(A.RETURN_POT, RETURNED)
    with pytest.raises(ValueError):
        s.acknowledge_refill(1000, RETURNED)


@pytest.mark.parametrize("remaining,action", [(200, A.SERVE_CUP), (199.99, A.WAIT_REFILL)])
def test_full_cup_reserve_is_distinct_from_eighty_percent_serving(remaining, action):
    s = CoffeeService(200, remaining)
    s.request_cups(1)
    s.complete(A.PREPARE, PREPARED)
    if remaining == 200:
        s.complete(A.TAKE_POT, TAKEN)
    assert s.next_action() == action


def test_speed_profile_keeps_contact_motion_separate():
    p = MotionSpeedProfile(4, 2)
    assert p.factor(independently_clear=True) == 4
    assert p.factor(independently_clear=False) == 2
    for free, contact in [(5, 2), (4, 4), (float("nan"), 1)]:
        with pytest.raises(ValueError):
            MotionSpeedProfile(free, contact)


def test_intact_inventory_evidence_cannot_replace_post_extraction_retention():
    s = holding()
    # Before cup extraction all cups may be supported through the bottom cup.
    # Serving still requires independent support for every cup left in the rack.
    e = served("cup_0", 240)
    e.predicates["remaining_cups_retained"] = False
    e.predicates["rack_inventory_supported"] = True
    with pytest.raises(ValueError, match="remaining_cups_retained"):
        s.complete(A.SERVE_CUP, e)


@pytest.mark.parametrize("choice", ["serve_again", "exit"])
def test_refill_with_no_pending_order_waits_for_explicit_next_choice(choice):
    s = holding(requests=2)
    s.complete(A.SERVE_CUP, served("cup_0", 240))
    s.complete(A.SERVE_CUP, served("cup_1", 80))
    s.complete(A.RETURN_POT, RETURNED)
    assert s.pending_cups == 0 and s.next_action() == A.WAIT_REFILL
    s.acknowledge_refill(
        400,
        evidence(
            "staff_refill_confirmed",
            "pot_on_hotplate",
            "pot_stable_after_release",
            "both_grippers_empty",
            "previous_served_cups_preserved",
        ),
    )
    # Refilling alone must not cause a new grasp or consume another cup.
    assert s.next_action() == A.IDLE
    assert s.pending_cups == 0 and list(s.served_cups) == ["cup_0", "cup_1"]
    if choice == "serve_again":
        s.request_cups(1)
        assert s.next_action() == A.TAKE_POT
        s.complete(A.TAKE_POT, TAKEN)
        assert s.next_action() == A.SERVE_CUP
    else:
        s.request_exit()
        assert s.next_action() == A.EXIT
        s.complete(
            A.EXIT,
            evidence(
                "standing_body_pose",
                "arms_hanging",
                "both_grippers_empty",
                "pot_stable_after_release",
                "previous_served_cups_preserved",
            ),
        )
        assert s.state == "standing" and s.next_action() == A.IDLE
    assert s.remaining_pot_ml == 400
