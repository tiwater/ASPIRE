import copy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / "tools/realman/audit_delivery_order.py"
spec = importlib.util.spec_from_file_location("delivery_order_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example():
    events = [{"cup_id": "cup_0", "time_s": 10.22}, {"cup_id": "cup_1", "time_s": 20.22}]
    def row(t, active, count):
        return {"time_s": t, "runtime_payload": {"active_cup": active},
                "staff_collection": {"events": copy.deepcopy(events[:count])}}
    trace = [row(0, "cup_0", 0), row(10.22, "cup_0", 1),
             row(10.22, "cup_1", 1), row(20.22, "cup_1", 2), row(21, "cup_1", 2)]
    return trace, events, {"cup_0": 10, "cup_1": 20}


def test_next_cup_can_start_at_same_clock_after_pickup():
    assert module.inspect_order(*example())["passed"]


@pytest.mark.parametrize("failure", ["early_next_cup", "late_pickup", "changed_receipt", "ledger_regression", "false_timestamp", "reused_cup", "missing_placement"])
def test_bad_order_or_false_completion_is_rejected(failure):
    trace, events, placements = example()
    if failure == "early_next_cup":
        trace[1]["runtime_payload"]["active_cup"] = "cup_1"
        trace[1]["staff_collection"]["events"] = []
    elif failure == "late_pickup": placements["cup_0"] = 9
    elif failure == "changed_receipt": trace[1]["staff_collection"]["events"][0]["time_s"] = 10.2
    elif failure == "ledger_regression": trace[2]["staff_collection"]["events"] = []
    elif failure == "false_timestamp": trace[1]["time_s"] = 10.21
    elif failure == "reused_cup": trace[-1]["runtime_payload"]["active_cup"] = "cup_0"
    elif failure == "missing_placement": placements.pop("cup_1")
    assert not module.inspect_order(trace, events, placements)["passed"]
