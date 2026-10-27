import pytest

from cap_harness.realman.task import STAGES, CoffeeProgress, Evidence, fill_target_ml


def test_no_ack_or_stale_evidence_can_advance():
    p = CoffeeProgress()
    assert not p.advance(
        {"controller_ok": Evidence(True, 10, "joint feedback")}, now_s=10, max_age_s=2
    )["advanced"]
    evidence = {k: Evidence(True, 1, "frame1") for k in STAGES[0][1]}
    assert not p.advance(evidence, now_s=10, max_age_s=2)["advanced"]
    evidence = {k: Evidence(True, 10, "calibration and scene observation") for k in STAGES[0][1]}
    assert p.advance(evidence, now_s=10, max_age_s=2)["advanced"]
    assert p.stage == "take_cup_right"
    assert not p.complete


def test_all_relations_required_and_stage_evidence_fresh():
    p = CoffeeProgress()
    for i, (_, required) in enumerate(STAGES):
        t = float(i + 1)
        evidence = {k: Evidence(True, t, f"observation-{i}") for k in required}
        last = required[-1]
        evidence[last] = Evidence(None, t, "occluded")
        assert not p.advance(evidence, now_s=t, max_age_s=1)["advanced"]
        evidence[last] = Evidence(True, t, f"second-view-{i}")
        assert p.advance(evidence, now_s=t, max_age_s=1)["advanced"]
    assert p.complete
    assert len(p.history) == len(STAGES)


def test_capacity_must_be_measured():
    assert fill_target_ml(200) == pytest.approx(160)
    for value in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            fill_target_ml(value)
