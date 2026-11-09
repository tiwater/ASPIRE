import json

import pytest

from tools.realman.audit_coffee_speed_policy import audit


def decision(clear, support, factor):
    return {
        "factor": factor,
        "pot_clear_of_machine": clear,
        "upward_platform_support_n": support,
        "independent_clearance_threshold_n": 0.02,
    }


def segment(stage, factor, evidence):
    return {
        "stage": stage,
        "factor": factor,
        "reference_peak_factor": factor,
        "planned_ticks": 10,
        "ramp_seconds": 0.5,
        "pot_transport_speed_decision": evidence,
    }


def write_result(path, segments):
    (path / "result.json").write_text(json.dumps({
        "experiment_success": True,
        "error": None,
        "parameters": {
            "arguments": {"free_speed_scale": 4, "pot_contact_speed_scale": 2},
            "speed_segments": segments,
        },
    }))


def test_speed_audit_accepts_clear_free_and_supported_contact_segments(tmp_path):
    write_result(tmp_path, [
        segment("pot_extract_020", 2, decision(False, 3.0, 2)),
        segment("pot_return_240", 4, decision(True, 0.0, 4)),
    ])

    report = audit(tmp_path)

    assert report["passed"] is True
    assert report["segments_checked"] == 2
    assert report["independently_clear_segments"] == 1
    assert report["contact_or_supported_segments"] == 1


def test_speed_audit_treats_support_above_threshold_as_contact(tmp_path):
    write_result(tmp_path, [
        segment("pot_extract_020", 2, decision(True, 0.020001, 2)),
        segment("pot_return_240", 4, decision(True, 0.0, 4)),
    ])

    report = audit(tmp_path)

    assert report["contact_or_supported_segments"] == 1


@pytest.mark.parametrize(
    "segments",
    [
        [segment("pot_extract_020", 4, decision(False, 1.0, 4)), segment("pot_return_240", 4, decision(True, 0, 4))],
        [segment("pot_extract_020", 2, None), segment("pot_return_240", 4, decision(True, 0, 4))],
        [segment("pot_extract_020", 2, decision(False, 1.0, 2))],
        [segment("pot_extract_020", 2, {**decision(False, 1.0, 2), "independent_clearance_threshold_n": 0.03}), segment("pot_return_240", 4, decision(True, 0, 4))],
        [{**segment("pot_extract_020", 2, decision(False, 1.0, 2)), "planned_ticks": 0}, segment("pot_return_240", 4, decision(True, 0, 4))],
    ],
)
def test_speed_audit_rejects_inconsistent_or_incomplete_segments(tmp_path, segments):
    write_result(tmp_path, segments)

    with pytest.raises(ValueError):
        audit(tmp_path)
