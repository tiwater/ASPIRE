#!/usr/bin/env python3
"""Create a camera-frame scene cloud and depth coverage report from a native archive."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from cap_harness.realman.rgbd_geometry import project_depth, zbuffer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rois', type=Path, help='JSON name -> [x0,y0,x1,y1]; pixel bounds, not segmentation')
    args = parser.parse_args()
    report = json.loads((args.capture / 'sample.json').read_text())
    for name in ('color.png', 'depth.bin', 'depth_m.npy'):
        if hashlib.sha256((args.capture / name).read_bytes()).hexdigest() != report['sha256'][name]:
            raise ValueError('capture hash mismatch: ' + name)
    color = cv2.imread(str(args.capture / 'color.png'))
    ci = report['color']['camera_info']
    if color.shape[:2] != (ci['height'], ci['width']):
        raise ValueError('color dimensions disagree')
    xyz, uv = project_depth(np.load(args.capture / 'depth_m.npy', allow_pickle=False), report)
    cloud = zbuffer(xyz, uv, ci['height'], ci['width'])
    valid = np.isfinite(cloud[:, :, 2])
    rois = json.loads(args.rois.read_text()) if args.rois else {}
    stats = {}
    for name, bounds in rois.items():
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= ci['width'] and 0 <= y0 < y1 <= ci['height']):
            raise ValueError('invalid ROI bounds')
        points = cloud[y0:y1, x0:x1][valid[y0:y1, x0:x1]]
        stats[name] = {'pixel_bounds_xyxy': bounds, 'point_count': len(points),
                       'projected_pixel_fraction': len(points) / ((x1-x0)*(y1-y0)),
                       'xyz_percentiles_10_50_90_m': np.percentile(points, [10,50,90], axis=0).tolist() if len(points) else None,
                       'object_identity_verified_by_depth': False}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'rgb_frame_cloud.npz', xyz_m=cloud)
    # Raw nearest-pixel coverage, no interpolation or synthesized surface.
    shade = np.zeros(valid.shape, dtype=np.uint8)
    shade[valid] = np.clip((cloud[:, :, 2][valid]-.2)/1.8*255, 0, 255).astype(np.uint8)
    viz = cv2.applyColorMap(shade, cv2.COLORMAP_TURBO)
    viz[~valid] = 0
    cv2.imwrite(str(args.output / 'projected_depth.png'), viz)
    overlay = color.copy()
    overlay[valid] = (.45*color[valid] + .55*viz[valid]).astype(np.uint8)
    for name, (x0,y0,x1,y1) in rois.items():
        cv2.rectangle(overlay, (x0,y0), (x1,y1), (255,0,255), 2)
        cv2.putText(overlay, name, (x0, max(15,y0-4)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,255), 1)
    cv2.imwrite(str(args.output / 'coverage_overlay.png'), overlay)
    # Decimated visible scene only; no surface reconstruction or collision claim.
    mask = valid.copy(); mask[1::2] = False; mask[:,1::2] = False
    points, colors = cloud[mask], color[mask][:, ::-1]
    with (args.output / 'scene_camera.ply').open('w') as f:
        f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % len(points))
        for p, c in zip(points, colors):
            f.write('%.6f %.6f %.6f %d %d %d\n' % (*p, *c))
    summary = {'camera': report['camera'], 'frame': ci['header']['frame_id'],
               'sample_json_sha256': hashlib.sha256((args.capture / 'sample.json').read_bytes()).hexdigest(),
               'projected_pixel_fraction': float(valid.mean()), 'rois': stats,
               'motion_ready': False,
               'limitations': ['Sparse forward projection; holes are unknown, not free space.',
                               'ROI statistics can include background and transparent-object failures.',
                               'Camera optical coordinates only; not robot base, not a grasp target.',
                               'No surface mesh, collision clearance, calibrated depth scale or complete object dimensions claimed.']}
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
