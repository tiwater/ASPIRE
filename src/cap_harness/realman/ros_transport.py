"""Single ROS2 subscriber/client owner; no motor/camera-driver startup."""

from __future__ import annotations

from collections import deque
import copy
import json
from pathlib import Path
import threading
import time

from .control import ARMS, READ_COMMANDS, arm_name

CAMERAS = ("head_depth_cam", "chest_depth_cam", "left_arm_depth_cam", "right_arm_depth_cam")


def stamp_dict(header):
    return {
        "stamp": {"sec": header.stamp.sec, "nanosec": header.stamp.nanosec},
        "frame_id": header.frame_id,
    }


class RosTransport:
    def __init__(self, artifact_root, *, allow_motion=False, allow_stop=False):
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rm_robot_interfaces.srv import StringCmd
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformListener

        self.rclpy = rclpy
        self.request_type = StringCmd.Request
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.allow_motion, self.allow_stop = allow_motion, allow_stop
        self.condition = threading.Condition()
        self.audit_lock = threading.Lock()
        self.states = deque(maxlen=300)
        self.images = {}
        self.infos = {}
        self.rgbd_images = {}
        self.subscriptions = []
        rclpy.init()
        self.node = rclpy.create_node("aspire_realman_cap")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        group = ReentrantCallbackGroup()
        self.client = self.node.create_client(StringCmd, "/robot/command", callback_group=group)
        self.subscriptions.append(
            self.node.create_subscription(
                String, "/robot_slave/states", self._state, 10, callback_group=group
            )
        )
        for camera in CAMERAS:
            self.subscriptions.append(
                self.node.create_subscription(
                    Image,
                    f"/{camera}/color/image_raw",
                    lambda msg, name=camera: self._image(name, msg),
                    qos_profile_sensor_data,
                    callback_group=group,
                )
            )
            self.subscriptions.append(
                self.node.create_subscription(
                    Image, f"/{camera}/depth/image_raw",
                    lambda msg, name=camera: self._depth(name, msg),
                    qos_profile_sensor_data, callback_group=group,
                )
            )
            self.subscriptions.append(
                self.node.create_subscription(
                    CameraInfo, f"/{camera}/depth/camera_info",
                    lambda msg, name=camera: self._info(name + "/depth", msg),
                    qos_profile_sensor_data, callback_group=group,
                )
            )
            self.subscriptions.append(
                self.node.create_subscription(
                    CameraInfo,
                    f"/{camera}/color/camera_info",
                    lambda msg, name=camera: self._info(name, msg),
                    qos_profile_sensor_data,
                    callback_group=group,
                )
            )
        self.executor = MultiThreadedExecutor(num_threads=4)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def audit(self, event):
        record = {"unix_s": time.time(), "monotonic_s": time.monotonic(), **event}
        with self.audit_lock, (self.artifact_root / "events.jsonl").open("a") as out:
            out.write(json.dumps(record, allow_nan=False) + "\n")

    def _state(self, msg):
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict):
                return
            record = {
                "received_unix_s": time.time(),
                "received_monotonic_s": time.monotonic(),
                "source_stamp_available": False,
                "state": value,
            }
            with self.condition:
                self.states.append(record)
                self.condition.notify_all()
        except ValueError:
            self.audit({"event": "invalid_ros_state"})

    def _image(self, name, msg):
        with self.condition:
            self.images[name] = {
                "message": msg,
                "received_monotonic_s": time.monotonic(),
                "received_unix_s": time.time(),
            }
            self.rgbd_images.setdefault(name, deque(maxlen=8)).append(self.images[name])
            self.condition.notify_all()

    def _depth(self, name, msg):
        with self.condition:
            self.rgbd_images.setdefault(name + "/depth", deque(maxlen=8)).append({
                "message": msg, "received_monotonic_s": time.monotonic(),
                "received_unix_s": time.time(),
            })
            self.condition.notify_all()

    def rgbd_pair(self, camera, *, after, timeout=3):
        """Pair existing driver streams; never changes camera settings or motion policy."""
        from .rgbd_capture import choose_pair, validate_frame
        if camera not in CAMERAS:
            raise ValueError("unknown camera")
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                pair = choose_pair(self.rgbd_images.get(camera, []),
                                   self.rgbd_images.get(camera + "/depth", []),
                                   after=after, now=time.monotonic(),
                                   ros_now_ns=self.node.get_clock().now().nanoseconds)
                if pair and camera in self.infos and camera + "/depth" in self.infos:
                    color, depth = [dict(f) for f in pair]
                    color["camera_info"] = copy.deepcopy(self.infos[camera])
                    depth["camera_info"] = copy.deepcopy(self.infos[camera + "/depth"])
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("fresh RGB-D pair/CameraInfo unavailable within 50ms skew")
                self.condition.wait(remaining)
        validate_frame(color)
        validate_frame(depth)
        from rclpy.time import Time
        from rclpy.duration import Duration
        target = color["message"].header.frame_id
        source = depth["message"].header.frame_id
        tf = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0.2))
        t, q = tf.transform.translation, tf.transform.rotation
        registration = {"parent_frame": target, "child_frame": source,
                        "translation_m": [t.x, t.y, t.z],
                        "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                        "header": stamp_dict(tf.header),
                        "source": "driver TF latest rigid intra-camera transform; not handeye"}
        return color, depth, registration

    def _info(self, name, msg):
        with self.condition:
            self.infos[name] = {
                "header": stamp_dict(msg.header),
                "height": msg.height,
                "width": msg.width,
                "distortion_model": msg.distortion_model,
                "k": list(msg.k),
                "d": list(msg.d),
                "r": list(msg.r),
                "p": list(msg.p),
            }
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            if not self.states:
                raise RuntimeError("no ROS state received")
            result = copy.deepcopy(self.states[-1])
        if not 0 <= time.monotonic() - result["received_monotonic_s"] <= 0.5:
            raise RuntimeError("ROS state stale")
        return result

    def next_state(self, after, timeout):
        with self.condition:
            if not self.condition.wait_for(
                lambda: bool(self.states) and self.states[-1]["received_monotonic_s"] > after,
                timeout,
            ):
                raise TimeoutError("fresh ROS state timeout")
        return self.snapshot()

    def frame(self, camera, *, after=0, timeout=2):
        if camera not in CAMERAS:
            raise ValueError("unknown camera")
        with self.condition:
            ready = lambda: (
                camera in self.images
                and camera in self.infos
                and self.images[camera]["received_monotonic_s"] > after
            )
            if not self.condition.wait_for(ready, timeout):
                raise TimeoutError("fresh image or CameraInfo timeout")
            frame = dict(self.images[camera])
            frame["camera_info"] = copy.deepcopy(self.infos[camera])
        msg = frame["message"]
        source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        frame["source_age_s"] = (self.node.get_clock().now().nanoseconds - source_ns) / 1e9
        if (
            not 0 <= time.monotonic() - frame["received_monotonic_s"] <= 0.5
            or not -0.05 <= frame["source_age_s"] <= 0.5
        ):
            raise ValueError("image source or receipt timestamp stale/inconsistent")
        info = frame["camera_info"]
        if (info["width"], info["height"], info["header"]["frame_id"]) != (
            msg.width,
            msg.height,
            msg.header.frame_id,
        ):
            raise ValueError("image/CameraInfo dimensions or frame disagree")
        return frame

    def query(self, arm, payload, *, timeout_s=2.0):
        arm_name(arm)
        command = payload.get("command")
        allowed = set(READ_COMMANDS)
        if self.allow_motion:
            allowed.add("movej")
        if self.allow_stop:
            allowed.add("set_arm_stop")
        if command not in allowed:
            raise ValueError("command not allowed by transport policy")
        if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 2:
            raise ValueError("invalid query deadline")
        started = time.monotonic()
        if not self.client.wait_for_service(timeout_sec=min(1, timeout_s)):
            raise RuntimeError("/robot/command unavailable")
        request = self.request_type()
        request.data = json.dumps({"device": ARMS[arm], "payload": payload}, allow_nan=False)
        self._query_audit(command, {"event": "query_send", "arm": arm, "payload": payload})
        future = self.client.call_async(request)
        complete = threading.Event()
        future.add_done_callback(lambda _: complete.set())
        remaining = max(0, timeout_s - (time.monotonic() - started))
        if not complete.wait(remaining):
            future.cancel()
            self._query_audit(command, {"event": "query_timeout", "arm": arm, "command": command})
            raise TimeoutError(f"{command} response timeout; never replay automatically")
        reply = future.result()
        response = json.loads(reply.data)
        if not isinstance(response, dict):
            raise ValueError("controller response is not an object")
        self._query_audit(
            command,
            {
                "event": "query_reply",
                "arm": arm,
                "command": command,
                "elapsed_s": time.monotonic() - started,
                "response": response,
            },
        )
        return response

    def _query_audit(self, command, event):
        try:
            self.audit(event)
        except OSError:
            # Disk failure must prevent a new move, never suppress an emergency stop.
            if command != "set_arm_stop":
                raise

    def state_window(self, start, end):
        with self.condition:
            return copy.deepcopy(
                [v for v in self.states if start <= v["received_monotonic_s"] <= end]
            )

    def close(self):
        self.executor.shutdown(timeout_sec=3)
        self.node.destroy_node()
        self.rclpy.shutdown()
        self.thread.join(timeout=3)


def frame_png(frame):
    import io

    import numpy as np
    from PIL import Image

    msg = frame["message"]
    if msg.encoding not in ("rgb8", "bgr8") or msg.step < msg.width * 3:
        raise ValueError("unsupported RGB encoding/stride")
    array = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    array = array[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "bgr8":
        array = array[:, :, ::-1]
    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()
