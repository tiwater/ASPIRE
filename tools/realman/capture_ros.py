"""Run on the ROS2 host. Subscriptions only; no publishers, services or robot SDK.

Save raw Image bytes + exact metadata, CameraInfo, TF and receipt-timed states.
Unaligned color/depth remain separate; no unverified D2C transformation is applied.
"""

import argparse
import json
from pathlib import Path
import time


def main():
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from rosidl_runtime_py.convert import message_to_ordereddict
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=[
            "head_depth_cam",
            "chest_depth_cam",
            "left_arm_depth_cam",
            "right_arm_depth_cam",
            "chest_wide_cam",
        ],
    )
    args = parser.parse_args()
    if not 0 < args.timeout <= 120 or any(not c.replace("_", "").isalnum() for c in args.cameras):
        parser.error("invalid timeout or camera name")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = rclpy.create_node("aspire_realman_readonly_capture")
    start = time.monotonic()
    record = {
        "schema_version": 1,
        "started_unix_s": time.time(),
        "mode": "subscriptions_only",
        "states": [],
        "images": {},
        "camera_info": {},
        "tf": {},
        "errors": [],
    }
    subscriptions = []
    expected = []

    def state_cb(msg):
        try:
            item = {
                "received_unix_s": time.time(),
                "received_monotonic_s": time.monotonic(),
                "state": json.loads(msg.data),
            }
            record["states"].append(item)
            record["states"] = record["states"][-100:]
        except (ValueError, TypeError) as exc:
            record["errors"].append(str(exc))

    subscriptions.append(
        node.create_subscription(String, "/robot_slave/states", state_cb, qos_profile_sensor_data)
    )

    def info_cb(key, msg):
        record["camera_info"][key] = {
            "received_unix_s": time.time(),
            "message": message_to_ordereddict(msg),
        }

    def image_cb(key, msg):
        # First sample after subscription, never a cached file from a previous attempt.
        if key in record["images"]:
            return
        filename = key.replace("/", "_").strip("_") + ".bin"
        (out / filename).write_bytes(bytes(msg.data))
        record["images"][key] = {
            "file": filename,
            "received_unix_s": time.time(),
            "header": message_to_ordereddict(msg.header),
            "height": msg.height,
            "width": msg.width,
            "encoding": msg.encoding,
            "is_bigendian": msg.is_bigendian,
            "step": msg.step,
        }

    for cam in args.cameras:
        streams = [f"/{cam}"] if "wide" in cam else [f"/{cam}/color", f"/{cam}/depth"]
        for key in streams:
            expected.append(key)
            subscriptions.append(
                node.create_subscription(
                    Image,
                    key + "/image_raw",
                    lambda m, k=key: image_cb(k, m),
                    qos_profile_sensor_data,
                )
            )
            subscriptions.append(
                node.create_subscription(
                    CameraInfo,
                    key + "/camera_info",
                    lambda m, k=key: info_cb(k, m),
                    qos_profile_sensor_data,
                )
            )

    def tf_cb(msg):
        for t in msg.transforms:
            record["tf"][t.child_frame_id] = {
                "received_unix_s": time.time(),
                "message": message_to_ordereddict(t),
            }

    subscriptions.append(node.create_subscription(TFMessage, "/tf", tf_cb, qos_profile_sensor_data))
    static_qos = QoSProfile(
        depth=10,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )
    subscriptions.append(node.create_subscription(TFMessage, "/tf_static", tf_cb, static_qos))
    try:
        while time.monotonic() - start < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.1)
            if len(record["states"]) >= 2 and all(
                k in record["images"] and k in record["camera_info"] for k in expected
            ):
                break
        record["missing_images"] = [k for k in expected if k not in record["images"]]
        record["missing_camera_info"] = [k for k in expected if k not in record["camera_info"]]
        record["finished_unix_s"] = time.time()
        record["topics"] = node.get_topic_names_and_types()
        record["synchronized"] = False
        record["note"] = (
            "Diagnostic samples only. Check stamps, clock domains, freshness, D2C and extrinsics before assembling a CAP observation."
        )
        (out / "capture.json").write_text(json.dumps(record, indent=2))
        print(
            json.dumps(
                {
                    "output": str(out),
                    "state_samples": len(record["states"]),
                    "images": len(record["images"]),
                    "missing_images": record["missing_images"],
                }
            )
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
