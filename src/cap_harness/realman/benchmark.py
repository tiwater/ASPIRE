"""Seeded simulation fixtures, independent of live IO and of policy success scoring."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def seeds_for_split(config, split):
    """Validate disjoint inclusive ranges before selecting any trial seed."""
    splits = config['splits']
    if set(splits) != {'development', 'validation', 'heldout'}:
        raise ValueError('expected development/validation/heldout partitions')
    seen = set()
    groups = {}
    for name, bounds in splits.items():
        if (len(bounds) != 2 or any(type(x) is not int for x in bounds)
                or not 0 <= bounds[0] <= bounds[1] or bounds[1] - bounds[0] > 10000):
            raise ValueError('invalid seed range')
        values = list(range(bounds[0], bounds[1] + 1))
        if seen.intersection(values):
            raise ValueError('overlapping seed partitions')
        seen.update(values)
        groups[name] = values
    if split not in groups:
        raise ValueError('unknown split')
    return groups[split]


def scene_for_seed(nominal, config, *, split, seed):
    """Perturb complete fixtures; the pot moves with its machine and cups with their rack."""
    if type(seed) is not int or seed not in seeds_for_split(config, split):
        raise ValueError('seed is outside requested split')
    rng = np.random.default_rng(seed)
    sampled = {}
    # Stable ordering prevents JSON key reordering from changing the scene.
    for name, bounds in sorted(config['randomization'].items()):
        if (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1] or (name.endswith('_scale') and bounds[0] <= 0)):
            raise ValueError('invalid randomization range')
        sampled[name] = float(rng.uniform(*bounds))
    expected = {'table_height_delta_m', 'base_x_delta_m', 'base_y_delta_m',
                'base_yaw_delta_rad', 'machine_x_delta_m', 'machine_y_delta_m',
                'dispenser_x_delta_m', 'dispenser_y_delta_m',
                'dispenser_friction_scale', 'outlet_force_prior_scale'}
    if set(sampled) != expected:
        raise ValueError('unknown or missing randomization parameter')
    scene = deepcopy(nominal)
    scene['table']['top_height_m'] += sampled['table_height_delta_m']
    for index, axis in enumerate(['x', 'y']):
        scene['robot']['base_xy_m'][index] += sampled[f'base_{axis}_delta_m']
        for name in ['machine', 'dispenser']:
            scene[name]['center_table_xy_m'][index] += sampled[f'{name}_{axis}_delta_m']
    scene['robot']['yaw_rad'] += sampled['base_yaw_delta_rad']
    scene['dispenser']['finger_friction'] *= sampled['dispenser_friction_scale']
    scene['force_priors']['spring_full_pot_weight_fraction'] *= sampled['outlet_force_prior_scale']
    scene['benchmark_instance'] = {
        'suite': config['name'], 'split': split, 'seed': seed, 'sampled_parameters': sampled,
        'physical_backend': None, 'qualification': 'randomized_simulation_fixture_only',
    }
    return scene


def check_fixtures(config_path, output, *, split, seeds):
    """Build and settle every requested fixture, retaining failures without resampling.

    This is NOT a policy evaluation. It deliberately leaves task success unknown.
    """
    from PIL import Image

    from .model import build_model
    from .sim import CoffeeSimAdapter

    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    allowed = seeds_for_split(config, split)
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in allowed for s in seeds):
        raise ValueError('need unique seeds from the requested split')
    seconds = config['fixture_check']['settle_seconds']
    tolerance = config['fixture_check']['max_object_drift_m']
    if not math.isfinite(seconds) or not 0 < seconds <= 30:
        raise ValueError('invalid settling duration')
    if not math.isfinite(tolerance) or not 0 < tolerance <= .05:
        raise ValueError('invalid fixture tolerance')
    robot_path, scene_path = Path(config['robot_config']), Path(config['scene_config'])
    robot, nominal = json.loads(robot_path.read_text()), json.loads(scene_path.read_text())
    if robot.get('physical_motion_backend') is not None or config.get('physical_backend') is not None:
        raise ValueError('fixture runner requires simulation-only configuration')
    output.mkdir(parents=True, exist_ok=False)
    sources = {'benchmark': config_path, 'robot': robot_path, 'scene': scene_path,
               'fixture_code': Path(__file__), 'builder_code': Path(__file__).with_name('model.py'),
               'adapter_code': Path(__file__).with_name('sim.py')}
    hashes = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in sources.items()}
    for name, path in sources.items():
        (output / (name + path.suffix)).write_bytes(path.read_bytes())
    summary = {'suite': config['name'], 'split': split, 'seeds': seeds,
               'purpose': 'fixture_validation_not_policy_evaluation', 'source_sha256': hashes,
               'simulation_only': True, 'coffee_task_success': None, 'trials': []}
    for seed in seeds:
        folder = output / f'seed_{seed}'
        folder.mkdir()
        scene = scene_for_seed(nominal, config, split=split, seed=seed)
        variant = folder / 'scene_config.json'
        variant.write_text(json.dumps(scene, indent=2) + '\n')
        adapter = None
        result = {'seed': seed, 'fixture_ok': False, 'coffee_task_success': None}
        try:
            built = build_model(Path(robot['urdf']), variant, folder / 'model')
            adapter = CoffeeSimAdapter(built['scene'], width=320, height=240)
            objects = ['pot'] + [f'cup_{i}' for i in range(scene['dispenser']['stack_count'])]
            before = {n: adapter.data.body(n).xpos.copy() for n in objects}
            Image.fromarray(adapter.camera('overview').rgb).save(folder / 'initial.png')
            adapter._advance(seconds)
            drift = {n: float(np.linalg.norm(adapter.data.body(n).xpos - x)) for n, x in before.items()}
            finite = bool(np.isfinite(adapter.data.qpos).all() and np.isfinite(adapter.data.qvel).all())
            warnings = adapter.data.warning.number.tolist()
            Image.fromarray(adapter.camera('overview').rgb).save(folder / 'settled.png')
            result.update(object_drift_m=drift, warnings=warnings, finite=finite,
                          simulation_seconds=float(adapter.data.time),
                          fixture_ok=finite and not any(warnings) and max(drift.values()) < tolerance,
                          model_sha256=hashlib.sha256(Path(built['scene']).read_bytes()).hexdigest(),
                          whole_task_reachability_checked=False, grasp_checked=False)
        except Exception as exc:
            result['error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if adapter is not None:
                adapter.close()
        (folder / 'fixture_result.json').write_text(json.dumps(result, indent=2) + '\n')
        summary['trials'].append(result)
        summary['all_fixtures_ok'] = all(t['fixture_ok'] for t in summary['trials'])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
