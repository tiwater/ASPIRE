import copy

import numpy as np
import pytest

from cap_harness.realman.rgbd_geometry import project_depth, zbuffer


def report():
    info = {'width': 4, 'height': 2, 'k': [10,0,0,0,10,0,0,0,1],
            'd': [0]*5, 'distortion_model': 'plumb_bob', 'header': {'frame_id': 'd'}}
    color = copy.deepcopy(info); color['header']['frame_id'] = 'c'
    return {'depth': {'camera_info': info}, 'color': {'camera_info': color},
            'color_from_depth': {'parent_frame': 'c', 'child_frame': 'd',
                                 'quaternion_xyzw': [0,0,0,1], 'translation_m': [.1,0,0]}}


def test_translation_direction_optical_z_and_holes():
    depth = np.full((2,4), np.nan); depth[0,0] = 1; depth[1,0] = 2
    xyz, uv = project_depth(depth, report())
    np.testing.assert_allclose(xyz, [[.1,0,1],[.1,.2,2]])
    np.testing.assert_array_equal(uv, [[1,0],[0,1]])
    bad = report(); bad['color_from_depth']['parent_frame'] = 'd'
    with pytest.raises(ValueError, match='direction'): project_depth(depth, bad)


def test_nearest_surface_wins_and_missing_pixels_stay_unknown():
    xyz = np.array([[0,0,2],[0,0,1],[1,0,3]])
    uv = np.array([[0,0],[0,0],[1,0]])
    result = zbuffer(xyz, uv, 2, 3)
    assert result[0,0,2] == 1
    assert result[0,1,2] == 3
    assert np.isnan(result[1]).all()


def test_no_valid_depth_and_unsupported_distortion():
    depth = np.zeros((2,4))
    xyz, uv = project_depth(depth, report())
    assert xyz.shape == (0,3) and uv.shape == (0,2)
    bad = report(); bad['depth']['camera_info']['distortion_model'] = 'equidistant'
    with pytest.raises(ValueError, match='distortion'): project_depth(depth, bad)
