import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[2] / 'tools/realman/report_coffee_service_timing.py'
spec = importlib.util.spec_from_file_location('service_timing', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def groups(stages):
    time = 0
    result = []
    for stage, duration in stages:
        result.append(dict(stage=stage, start_s=time, end_s=time+duration))
        time += duration
    return result


PICKUP = [('cup_approach', 5), ('cup_close', 2), ('cup_extract_down', 9),
          ('cup_withdraw', 9), ('cup_hold', 3)]


def test_pot_first_order_does_not_create_a_false_cup_speedup():
    before = groups(PICKUP + [('pot_align', 10), ('pot_extract_010', 100), ('cup_raise', 3)])
    after = groups([('pot_align', 10), ('pot_extract_010', 100)] + PICKUP + [('cup_raise', 3)])
    b = module.cup_pickup_windows(before)[0]
    a = module.cup_pickup_windows(after)[0]
    assert b['duration_s'] == a['duration_s'] == 28
    assert module.windows(before, 'cup_approach', 'cup_raise')[0]['duration_s'] == 138


def test_three_repetitions_remain_separate_and_include_settling():
    stages = PICKUP + [('cup_hold_rack_settle', .5), ('cup_hold', 1), ('cup_raise', 3)]
    windows = module.cup_pickup_windows(groups(stages * 3))
    assert len(windows) == 3
    assert [w['duration_s'] for w in windows] == [29.5] * 3


@pytest.mark.parametrize('stages', [
    [], [('cup_raise', 3)], PICKUP[:-1],
    PICKUP[:-1] + [('pot_align', 5), ('cup_hold', 3)],
    PICKUP[:-1] + PICKUP,
])
def test_missing_hold_or_intervening_other_action_is_rejected(stages):
    with pytest.raises(ValueError):
        module.cup_pickup_windows(groups(stages))
