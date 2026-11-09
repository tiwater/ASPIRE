"""Render audited fixed-slot coffee service with explicit staff pickup captions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--playback-speed", type=float, default=4)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/home/sawyer/.local/share/fonts/NotoCJKsc/NotoSansCJKsc-Regular.otf"),
    )
    args = parser.parse_args()
    if args.video.exists():
        raise FileExistsError("use a new video path")
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        raise ValueError("positive finite playback speed required")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    audit_path = args.trial / "audit.json"
    audit = json.loads(audit_path.read_text())
    if not audit["passed"] or audit.get("scope") != "continuous_simulation_service_with_assumed_staff_pickup":
        raise ValueError("complete continuous session audit must pass first")
    for filename, key in [("trace.json", "trace_sha256"), ("result.json", "result_sha256")]:
        if sha(args.trial / filename) != audit[key]:
            raise ValueError("audited evidence changed")
    model = args.trial.parent / "review/model/scene.xml"
    if sha(model) != audit["model_sha256"]:
        raise ValueError("model identity changed")
    trace = json.loads((args.trial / "trace.json").read_text())
    times = np.array([row["time_s"] for row in trace])
    m = mj.MjModel.from_xml_path(str(model))
    d = mj.MjData(m)
    m.vis.global_.offwidth = 1280
    m.vis.global_.offheight = 960
    overview = mj.MjvCamera()
    overview.lookat[:] = [0.5, 0, 0.9]
    overview.distance = 2.6
    overview.azimuth = 35
    overview.elevation = -15
    close = mj.MjvCamera()
    close.lookat[:] = [0.82, 0, 0.87]
    close.distance = 1.25
    close.azimuth = 45
    close.elevation = -30
    font = ImageFont.truetype(str(args.font), 25)
    small = ImageFont.truetype(str(args.font), 20)
    fps = 20
    samples = np.r_[np.arange(0, times[-1], args.playback_speed / fps), times[-1]]
    indices = np.maximum(0, np.searchsorted(times, samples, side="right") - 1)
    args.video.parent.mkdir(parents=True, exist_ok=True)
    with (
        mj.Renderer(m, width=1280, height=960) as full,
        mj.Renderer(m, width=480, height=360) as inset,
        imageio.get_writer(args.video, fps=fps, codec="libx264", quality=7) as writer,
    ):
        for frame, index in enumerate(indices):
            row = trace[int(index)]
            d.qpos[:] = row["qpos"]
            mj.mj_forward(m, d)
            full.update_scene(d, camera=overview)
            im = Image.fromarray(full.render())
            if not row["stage"].startswith(("initial", "standing", "prepare_", "exit_")):
                inset.update_scene(d, camera=close)
                im.paste(Image.fromarray(inset.render()), (785, 115))
            draw = ImageDraw.Draw(im)
            draw.rectangle([0, 0, 1279, 92], fill="black")
            draw.text(
                (15, 8),
                f"准备 → 咖啡服务 → 退出｜单次连续仿真｜播放 {args.playback_speed:g} 倍速",
                font=font,
                fill="white",
            )
            draw.text(
                (15, 48),
                f"{row['time_s']:.2f} / {times[-1]:.2f} s · {row['stage']}",
                font=small,
                fill="#ffe082",
            )
            collected = row.get("staff_collection", {}).get("events", [])
            draw.rectangle([0, 834, 1279, 875], fill="black")
            caption = f"固定交付点｜假设工作人员取走已放好的杯子：{len(collected)} 杯（外部事件）"
            if collected and row["time_s"] - collected[-1]["time_s"] < 5:
                caption += f" · 刚取走 {collected[-1]['cup_id']}"
            draw.text((15, 841), caption, font=small, fill="#80deea")
            payload = row.get("runtime_payload")
            if payload:
                volumes = payload.get(
                    "vessel_volumes_m3",
                    {"pot": payload["pot_volume_m3"], "cup_0": payload["cup_volume_m3"]},
                )
                text = "规定负载液量：" + "  ".join(
                    f"{name} {volume * 1e6:.1f} ml" for name, volume in volumes.items()
                )
                draw.rectangle([0, 878, 1279, 916], fill="black")
                draw.text((15, 884), text, font=small, fill="white")
            draw.rectangle([0, 920, 1279, 959], fill="black")
            draw.text(
                (15, 925),
                "几何真值与规定质量转移；身体范围为仿真候选，未认证真机或真实液流",
                font=small,
                fill="white",
            )
            writer.append_data(np.asarray(im))
            if frame % 500 == 0:
                print(f"rendered {frame}/{len(indices)}", flush=True)
    meta = {
        "video": str(args.video),
        "sha256": sha(args.video),
        "source_audit_sha256": sha(audit_path),
        "source_trace_sha256": audit["trace_sha256"],
        "simulation_seconds": float(times[-1]),
        "frames": len(indices),
        "fps": fps,
        "playback_speed": args.playback_speed,
        "uniform_simulation_sampling": True,
        "maximum_sample_age_s": float(np.max(samples - times[indices])),
        "continuous_run": True,
        "video_versioned": False,
        "font_sha256": sha(args.font),
        "camera_layout": "whole robot with synchronized table inset during coffee actions",
        "physical_motion_allowed": False,
        "fluid_simulated": False,
        "staff_pickup_assumed": True,
        "human_motion_simulated": False,
    }
    args.video.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
