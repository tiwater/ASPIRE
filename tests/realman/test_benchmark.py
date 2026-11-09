from copy import deepcopy
import json
from pathlib import Path

import pytest

from cap_harness.realman.benchmark import check_fixtures, scene_for_seed, seeds_for_split


@pytest.fixture
def inputs():
    root = Path('configs/realman')
    return json.loads((root/'coffee_scene.json').read_text()), json.loads((root/'coffee_benchmark.json').read_text())


def test_seed_reproducibility_distinction_and_no_nominal_mutation(inputs):
    nominal, config = inputs
    original = deepcopy(nominal)
    one = scene_for_seed(nominal, config, split='development', seed=51)
    assert one == scene_for_seed(nominal, config, split='development', seed=51)
    other = scene_for_seed(nominal, config, split='development', seed=52)
    assert one['robot']['base_xy_m'] != other['robot']['base_xy_m']
    assert one['machine']['center_table_xy_m'] != other['machine']['center_table_xy_m']
    assert nominal == original
    assert one['pot']['mass_kg'] == .396
    assert one['task']['fill_fraction'] == .8
    assert one['cup'] == nominal['cup']
    assert one['machine']['pot_bay_x_m'] == nominal['machine']['pot_bay_x_m']


def test_partitions_and_reordered_json_do_not_leak_or_change_scene(inputs):
    nominal, config = inputs
    assert seeds_for_split(config, 'development') == list(range(51,66))
    with pytest.raises(ValueError, match='outside'):
        scene_for_seed(nominal, config, split='development', seed=1)
    first = scene_for_seed(nominal, config, split='development', seed=51)
    config['randomization'] = dict(reversed(list(config['randomization'].items())))
    assert first == scene_for_seed(nominal, config, split='development', seed=51)
    config['splits']['validation'] = [65,80]
    with pytest.raises(ValueError, match='overlapping'):
        seeds_for_split(config, 'development')


def test_invalid_randomization_is_rejected(inputs):
    nominal, config = inputs
    config['randomization']['dispenser_friction_scale'] = [-1,1]
    with pytest.raises(ValueError, match='range'):
        scene_for_seed(nominal, config, split='development', seed=51)


def test_runner_rejects_wrong_split_before_creating_output(tmp_path):
    pytest.importorskip('mujoco')
    with pytest.raises(ValueError, match='unique seeds'):
        check_fixtures('configs/realman/coffee_benchmark.json', tmp_path/'run', split='development', seeds=[1])
    assert not (tmp_path/'run').exists()
