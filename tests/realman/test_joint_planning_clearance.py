from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from cap_harness.realman.joint_planning import plan_joint_path


def adapter_with_gap(gap):
    mj = pytest.importorskip('mujoco')
    root = ET.Element('mujoco')
    ET.SubElement(root, 'compiler', angle='radian')
    world = ET.SubElement(root, 'worldbody')
    body = ET.SubElement(world, 'body', name='base_link')
    for i in range(7):
        body = ET.SubElement(body, 'body', name=f'link{i}')
        ET.SubElement(body, 'joint', name=f'j{i}', axis='0 0 1', range='-1 1')
        ET.SubElement(body, 'inertial', pos='0 0 0', mass='1', diaginertia='.01 .01 .01')
    ET.SubElement(body, 'geom', type='sphere', size='.01')
    pot = ET.SubElement(world, 'body', name='pot', pos=f'{.02 + gap} 0 0')
    ET.SubElement(pot, 'geom', type='sphere', size='.01')
    m = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    return SimpleNamespace(model=m, data=d, mj=mj, arms=['left'],
                           qadr={'left': np.arange(7)}, joint_ids={'left': np.arange(7)})


def test_positive_gap_rejected_without_mutating_live_state():
    a = adapter_with_gap(.003)
    original = a.data.qpos.copy()
    assert plan_joint_path(a, 'left', np.zeros(7))['found']
    result = plan_joint_path(a, 'left', np.zeros(7), obstacle_body='pot', clearance_m=.005)
    assert not result['found']
    assert result['reason'] == 'start_or_goal_collision'
    np.testing.assert_array_equal(a.data.qpos, original)


def test_sufficient_gap_is_accepted_and_reported():
    a = adapter_with_gap(.007)
    result = plan_joint_path(a, 'left', np.zeros(7), obstacle_body='pot', clearance_m=.005)
    assert result['found']
    assert result['minimum_obstacle_clearance_m'] == .005
    assert not result['continuous_collision_checked']


@pytest.mark.parametrize('clearance', [-.001, float('nan'), float('inf')])
def test_invalid_clearance(clearance):
    with pytest.raises(ValueError, match='clearance'):
        plan_joint_path(adapter_with_gap(.007), 'left', np.zeros(7), obstacle_body='pot', clearance_m=clearance)
