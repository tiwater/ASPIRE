"""Read fixed, documented RealMan query commands via ROS2; never send motion.

Run on the robot with ROS2 and rm_robot_interfaces sourced. Preserve raw units,
responses and request/receipt times; these are diagnostic snapshots, not
synchronized image/pose pairs or verified tool transforms.
"""

import argparse
import json
from pathlib import Path
import time

READ_COMMANDS = (
    "get_current_arm_state",
    "get_current_tool_frame",
    "get_current_work_frame",
)


def main():
    import rclpy
    from rm_robot_interfaces.srv import StringCmd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    # Reserve a new file before touching the service. Never overwrite evidence.
    with args.output.open("x") as output:
        rclpy.init()
        node = rclpy.create_node("aspire_realman_controller_readback")
        client = node.create_client(StringCmd, "/robot/command")
        report = {
            "mode": "fixed_read_queries_only",
            "records": [],
            "synchronized_with_images": False,
            "handeye_qualified": False,
        }
        try:
            if not client.wait_for_service(timeout_sec=5):
                raise RuntimeError("/robot/command unavailable")
            for arm, device in (("left", 0), ("right", 1)):
                for command in READ_COMMANDS:
                    request = StringCmd.Request()
                    request.data = json.dumps({"device": device, "payload": {"command": command}})
                    record = {"arm": arm, "query": command, "requested_unix_s": time.time()}
                    future = client.call_async(request)
                    rclpy.spin_until_future_complete(node, future, timeout_sec=5)
                    record["received_unix_s"] = time.time()
                    if not future.done():
                        future.cancel()
                        record["error"] = "timeout; no retry"
                    else:
                        try:
                            response = future.result()
                            if response is None:
                                raise RuntimeError("empty service result")
                            record["response"] = json.loads(response.data)
                        except (ValueError, RuntimeError) as exc:
                            record["error"] = str(exc)
                    report["records"].append(record)
        finally:
            json.dump(report, output, indent=2)
            node.destroy_node()
            rclpy.shutdown()
    print(json.dumps({"output": str(args.output), "read_queries": len(report["records"])}))


if __name__ == "__main__":
    main()
