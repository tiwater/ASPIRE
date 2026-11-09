"""Offline projection of archived depth into RGB; no robot or world coordinates."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def intrinsics(info):
    if info['distortion_model'] not in ('plumb_bob', 'rational_polynomial'):
        raise ValueError('unsupported distortion model')
    k = np.asarray(info['k'], dtype=float).reshape(3, 3)
    d = np.asarray(info['d'], dtype=float)
    if (not np.isfinite(k).all() or not np.isfinite(d).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0 or d.size not in (4, 5, 8, 12, 14)):
        raise ValueError('invalid intrinsics')
    return k, d


def project_depth(depth_m, report):
    """Return RGB-frame XYZ and nearest RGB pixel for each projected depth sample.

    Includes occluded samples: callers must z-buffer before using an RGB ROI.
    Uses optical Z rather than range; never fills missing or transparent surfaces.
    """
    di, ci = report['depth']['camera_info'], report['color']['camera_info']
    x = report['color_from_depth']
    if x['parent_frame'] != ci['header']['frame_id'] or x['child_frame'] != di['header']['frame_id']:
        raise ValueError('transform direction does not match camera frames')
    kd, dd = intrinsics(di)
    kc, dc = intrinsics(ci)
    depth_m = np.asarray(depth_m)
    if depth_m.shape != (di['height'], di['width']):
        raise ValueError('depth dimensions disagree')
    q, t = np.asarray(x['quaternion_xyzw']), np.asarray(x['translation_m'])
    if (q.shape != (4,) or t.shape != (3,) or not np.isfinite(q).all()
            or not np.isfinite(t).all() or abs(np.linalg.norm(q) - 1) > 1e-4):
        raise ValueError('invalid rigid transform')
    v, u = np.nonzero(np.isfinite(depth_m) & (depth_m > 0))
    if not len(u):
        return np.empty((0, 3)), np.empty((0, 2), dtype=int)
    rays = cv2.undistortPoints(np.stack([u, v], axis=1).astype(float)[:, None], kd, dd)[:, 0]
    xyz = np.column_stack([rays, np.ones(len(rays))]) * depth_m[v, u, None]
    xyz = xyz @ Rotation.from_quat(q).as_matrix().T + t
    xyz = xyz[xyz[:, 2] > 0]
    if not len(xyz):
        return xyz, np.empty((0, 2), dtype=int)
    uv = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), kc, dc)[0][:, 0]
    good = np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] < ci['width'] - .5) & (uv[:, 1] >= 0) & (uv[:, 1] < ci['height'] - .5)
    return xyz[good], np.rint(uv[good]).astype(int)


def zbuffer(xyz, uv, height, width):
    """Nearest optical Z wins at each pixel. NaN means unknown, not free space."""
    result = np.full((height, width, 3), np.nan, dtype=np.float32)
    if len(xyz):
        flat = uv[:, 1] * width + uv[:, 0]
        order = np.lexsort((xyz[:, 2], flat))
        _, first = np.unique(flat[order], return_index=True)
        keep = order[first]
        result[uv[keep, 1], uv[keep, 0]] = xyz[keep]
    return result
