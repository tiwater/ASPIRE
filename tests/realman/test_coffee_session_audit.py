"""The acceptance audit rejects broken business order despite matching cup counts."""

from copy import deepcopy
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "tools/realman/audit_coffee_session.py"
spec = importlib.util.spec_from_file_location("coffee_session_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    def event(action, before, after, remaining, pending, cup=None):
        return dict(
            action=action,
            **{"from": before, "to": after},
            remaining_pot_ml=remaining,
            pending_cups=pending,
            cup_id=cup,
        )

    events = [
        event("prepare", "standing", "ready", 400, 3),
        event("take_pot", "ready", "holding_pot", 400, 3),
        event("serve_cup", "holding_pot", "holding_pot", 240, 2, "cup_0"),
        event("serve_cup", "holding_pot", "holding_pot", 80, 1, "cup_1"),
        event("return_pot", "holding_pot", "waiting_refill", 80, 1),
        {"action": "external_refill", "remaining_pot_ml": 400},
        event("take_pot", "ready", "holding_pot", 400, 1),
        event("serve_cup", "holding_pot", "holding_pot", 240, 0, "cup_2"),
        event("return_pot", "holding_pot", "ready", 240, 0),
        event("exit", "ready", "standing", 240, 0),
    ]
    payload = {
        "initial_volume_m3": 400e-6,
        "pot_volume_m3": 240e-6,
        "vessel_volumes_m3": {"pot": 240e-6, **{f"cup_{i}": 160e-6 for i in range(3)}},
        "external_events": [{"added_m3": 320e-6}],
    }
    return events, payload


def test_three_cups_return_refill_and_exit_reconcile():
    events, payload = fixture()
    result = module.audit_business_sequence(events, payload, 200, 3)
    assert result["passed"]
    assert result["refills"] == 1
    assert result["served_cups"] == ["cup_0", "cup_1", "cup_2"]


def test_refill_while_holding_is_rejected_even_with_same_final_inventory():
    events, payload = fixture()
    events[4], events[5] = events[5], events[4]
    result = module.audit_business_sequence(events, payload, 200, 3)
    assert not result["passed"]
    assert "invalid from 'holding_pot'" in result["errors"][0]


def test_third_serving_without_refill_fails_full_cup_reserve():
    events, payload = fixture()
    third = deepcopy(events[7])
    third["remaining_pot_ml"] = -80
    result = module.audit_business_sequence(events[:4] + [third], payload, 200, 3)
    assert not result["passed"]
    assert "one-full-cup reserve" in result["errors"][0]


def test_refill_ledger_mismatch_and_missing_final_return_are_rejected():
    events, payload = fixture()
    payload["external_events"][0]["added_m3"] = 321e-6
    assert not module.audit_business_sequence(events, payload, 200, 3)["passed"]
    events, payload = fixture()
    del events[-2]
    assert not module.audit_business_sequence(events, payload, 200, 3)["passed"]


def test_exactly_one_full_cup_is_sufficient():
    events, payload = fixture()
    events = [events[0], events[1], events[2], events[-2], events[-1]]
    for index, event in enumerate(events):
        event["pending_cups"] = 1 if index < 2 else 0
        event["remaining_pot_ml"] = 200 if index < 2 else 40
    events[-2]["to"] = events[-1]["from"] = "waiting_refill"
    payload.update(initial_volume_m3=200e-6, pot_volume_m3=40e-6, external_events=[])
    payload["vessel_volumes_m3"] = {"pot": 40e-6, "cup_0": 160e-6, "cup_1": 0, "cup_2": 0}
    result = module.audit_business_sequence(events, payload, 200, 1)
    assert result["passed"]
