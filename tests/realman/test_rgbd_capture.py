"""RGB-D timing, units and archives, with no ROS or robot connection."""

import json
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from cap_harness.realman.rgbd_capture import choose_pair, depth_metres, validate_frame, capture_rgbd


def frame(stamp=10.0, receipt=10.0, encoding="16UC1", big=False, data=None):
    s = NS(sec=int(stamp), nanosec=round((stamp % 1) * 1e9))
    msg = NS(header=NS(stamp=s, frame_id="depth_optical"), width=2, height=1,
             step=6, encoding=encoding, is_bigendian=int(big),
             data=data if data is not None else np.array([1000, 0, 999], dtype=">u2" if big else "<u2").tobytes())
    return {"message": msg, "received_monotonic_s": receipt,
            "camera_info": {"width": 2, "height": 1, "header": {"frame_id": "depth_optical", "stamp": {"sec": s.sec, "nanosec": s.nanosec}},
                            "k": [100, 0, 1, 0, 100, .5, 0, 0, 1]}}


@pytest.mark.parametrize("big", [False, True])
def test_depth_stride_endianness_and_zero(big):
    result = depth_metres(frame(big=big)["message"])
    assert result.shape == (1, 2)
    assert result[0, 0] == pytest.approx(1.0)
    assert np.isnan(result[0, 1])


def test_float_depth_units_and_nonfinite():
    f = frame(encoding="32FC1", data=np.array([1.25, float('inf')], dtype='<f4').tobytes())
    f['message'].step = 8
    result = depth_metres(f['message'])
    assert result[0, 0] == 1.25
    assert np.isnan(result[0, 1])


def test_corrupt_depth_rejected():
    f = frame(); f['message'].data = b'bad'
    with pytest.raises(ValueError, match='buffer'): depth_metres(f['message'])
    f = frame(); f['message'].encoding = '8UC1'
    with pytest.raises(ValueError, match='encoding'): depth_metres(f['message'])


def test_pair_uses_exposure_not_arrival_and_rejects_stale_or_old():
    color = frame(10.0, 10.1)
    wrong = frame(9.8, 10.1)
    good = frame(10.03, 10.09)
    pair = choose_pair([color], [wrong, good], after=10.01, now=10.2, ros_now_ns=10_200_000_000)
    assert pair[1] is good
    assert choose_pair([color], [wrong], after=10.01, now=10.2, ros_now_ns=10_200_000_000) is None
    assert choose_pair([color], [good], after=10.11, now=10.2, ros_now_ns=10_200_000_000) is None
    assert choose_pair([color], [good], after=9, now=11, ros_now_ns=11_000_000_000) is None


def test_camera_info_frame_and_time_checked():
    f = frame(); validate_frame(f)
    f['camera_info']['header']['frame_id'] = 'wrong'
    with pytest.raises(ValueError, match='frame'): validate_frame(f)
    f = frame(); f['camera_info']['header']['stamp']['sec'] = 8
    with pytest.raises(ValueError, match='timestamp'): validate_frame(f)


def test_archive_has_raw_units_intrinsics_transform_and_never_queries_robot(tmp_path):
    color, depth = frame(10, 10.1), frame(10.03, 10.12)
    color['message'].encoding = 'rgb8'
    color['message'].data = bytes([255, 0, 0, 0, 255, 0])
    color['message'].header.frame_id = 'color_optical'
    color['camera_info']['header']['frame_id'] = 'color_optical'
    state = {'state': {k: {'position': [0, 0]} for k in ['arm_left', 'arm_right', 'body', 'head']}}
    state['received_monotonic_s'] = 9.9
    transport = NS(artifact_root=tmp_path, next_state=lambda *a, **kw: state,
                   rgbd_pair=lambda *a, **kw: (color, depth, {'parent_frame': 'color_optical', 'child_frame': 'depth_optical',
                                                           'translation_m': [.02, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1]}))
    control = NS(lock=threading.Lock(), transport=transport)
    result = capture_rgbd(control, 'test_camera')
    assert not result['motion_commanded']
    assert not result['depth_registered_to_rgb']
    assert result['source_skew_s'] == pytest.approx(.03)
    assert result['depth_valid_fraction'] == .5
    saved = json.loads((next(tmp_path.iterdir()) / 'sample.json').read_text())
    assert saved['color_from_depth']['translation_m'] == [.02, 0, 0]
    assert not control.lock.locked()
    transport.rgbd_pair = lambda *a, **kw: (_ for _ in ()).throw(TimeoutError('missing frame'))
    with pytest.raises(TimeoutError): capture_rgbd(control, 'test_camera')
    assert not control.lock.locked()


def test_capture_rejects_concurrent_action_without_reading():
    lock = threading.Lock(); lock.acquire()
    with pytest.raises(RuntimeError, match='owns'):
        capture_rgbd(NS(lock=lock, transport=None), 'head_depth_cam')
    assert lock.locked()


def test_native_tool_is_readonly_and_camera_enum_bounded():
    pytest.importorskip('mcp')
    import asyncio
    from cap_harness.realman.live_server import build_server
    server = build_server(NS())
    tools = asyncio.run(server.list_tools())
    tool = next(t for t in tools if t.name == 'cap_capture_rgbd')
    assert len(tool.inputSchema['properties']['camera']['enum']) == 4
