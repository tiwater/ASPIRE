import json
from pathlib import Path

import pytest

from cap_harness.realman.service_release import (
    RELEASE_MODEL_SHA256, REQUIRED_SUPPORT, release_retry_allowed,
)


def actual_failure():
    path = Path("assets/realman/coffee/service_acceptance_20261102/thirdcup_final_release_residual_contact.json")
    report = json.loads(path.read_text())
    predicates = {k: True for k in REQUIRED_SUPPORT}
    predicates.update({k: report[k] for k in ["pot_on_hotplate", "lid_outlet_aligned", "pot_upright", "left_gripper_released"]})
    predicates["pot_stable_after_release"] = False
    return predicates, report["residual_tool_handle_contacts"]


def allowed(predicates, contacts, **kwargs):
    return release_retry_allowed(predicates, contacts, model_sha256=kwargs.get("model_sha256", RELEASE_MODEL_SHA256), opening=kwargs.get("opening", .205))


def test_recorded_residual_contact_can_request_recovery_but_is_not_released():
    predicates, contacts = actual_failure()
    assert allowed(predicates, contacts)
    assert predicates["left_gripper_released"] is False
    predicates["pot_stable_after_release"] = True
    assert not allowed(predicates, contacts)


@pytest.mark.parametrize("name", REQUIRED_SUPPORT)
@pytest.mark.parametrize("value", [False, None])
def test_missing_support_or_preserved_objects_forbids_retry(name, value):
    predicates, contacts = actual_failure()
    predicates[name] = value
    assert not allowed(predicates, contacts)


@pytest.mark.parametrize("force", [.020001, float("nan"), float("inf"), -1])
def test_excessive_or_invalid_force_forbids_retry(force):
    predicates, contacts = actual_failure()
    contacts[0]["normal_force_n"] = force
    assert not allowed(predicates, contacts)


def test_other_contacts_missing_contact_and_changed_model_are_rejected():
    predicates, contacts = actual_failure()
    assert not allowed(predicates, [])
    assert not allowed(predicates, contacts, model_sha256="different-model")
    assert not allowed(predicates, contacts, opening=.22)
    contacts.append(dict(geom1="left_left_pad_contact", geom2="pot_glass_collision_1", normal_force_n=.002))
    assert not allowed(predicates, contacts)
