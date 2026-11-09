"""Exercise deployed MCP with an explicit read-only tool allowlist; save evidence."""

from __future__ import annotations

import argparse
import asyncio
import base64
from datetime import timedelta
import json
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]
READ_ONLY = {
    "cap_get_capabilities",
    "cap_get_robot_state",
    "cap_observe",
    "cap_preview_joint_move",
    "cap_capture_handeye_sample",
    "cap_probe_feedback",
}


async def check(output, capture, concurrent=False):
    output.mkdir(parents=True, exist_ok=False)
    params = StdioServerParameters(
        command="bash", args=[str(ROOT / "scripts/realman/start_remote_mcp.sh")]
    )
    with (output / "server_stderr.log").open("w") as errlog:
        async with (
            stdio_client(params, errlog=errlog) as (reader, writer),
            ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=30)) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            (output / "tools.json").write_text(tools.model_dump_json(indent=2))
            print("Connected tools:", ", ".join(t.name for t in tools.tools), flush=True)

            async def call(name, args):
                if name not in READ_ONLY:
                    raise ValueError("read-only integration check refuses action tools")
                result = await session.call_tool(name, args)
                if result.isError:
                    raise RuntimeError(str(result.content))
                text = [x.text for x in result.content if x.type == "text"]
                data = result.structuredContent
                if data is None and len(text) == 1:
                    try:
                        data = json.loads(text[0])
                    except ValueError:
                        data = {"text": text}
                (output / (name + ".json")).write_text(
                    json.dumps(data or {"text": text}, indent=2) + "\n"
                )
                for index, block in enumerate(result.content):
                    if block.type == "image":
                        (output / f"{name}_{index}.png").write_bytes(base64.b64decode(block.data))
                return data

            caps = await call("cap_get_capabilities", {})
            assert caps["startup_motion"] is False
            assert caps["motion_blockers"]["left"] and caps["motion_blockers"]["right"]
            # Stop-only commissioning deployments still prohibit motion.
            if concurrent:
                async with (
                    stdio_client(params, errlog=errlog) as (reader2, writer2),
                    ClientSession(
                        reader2, writer2, read_timeout_seconds=timedelta(seconds=30)
                    ) as second,
                ):
                    await second.initialize()
                    second_tools = await second.list_tools()
                    assert {t.name for t in second_tools.tools} == {t.name for t in tools.tools}
                    second_caps = await second.call_tool("cap_get_capabilities", {})
                    assert not second_caps.isError
                    assert (await call("cap_get_capabilities", {}))["startup_motion"] is False
                # Closing an observer must not terminate the shared robot owner.
                assert (await call("cap_get_capabilities", {}))["startup_motion"] is False
                (output / "concurrent_connections.json").write_text(
                    json.dumps(
                        {
                            "two_simultaneous_clients": True,
                            "tool_count_each": len(tools.tools),
                            "first_usable_after_second_disconnect": True,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print("Concurrent clients and observer disconnect checks passed.", flush=True)
            state = await call("cap_get_robot_state", {})
            q = state["controller"]["left"]["joint_deg"]
            print("Live left controller degrees:", q, flush=True)
            await call("cap_observe", {"camera": "left_arm_depth_cam"})
            preview = await call(
                "cap_preview_joint_move", {"arm": "left", "target_deg": q, "expected_start_deg": q}
            )
            assert preview["motion_blockers"]
            if capture:
                sample = await call("cap_capture_handeye_sample", {"arm": "left"})
                print(
                    "Capture:",
                    json.dumps(
                        {
                            k: sample.get(k)
                            for k in [
                                "path",
                                "timing_stationarity_passed",
                                "bracket_span_s",
                                "error",
                            ]
                        }
                    ),
                    flush=True,
                )
            print("Read-only MCP checks completed; no motion/stop tool invoked.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-left", action="store_true")
    parser.add_argument("--check-concurrent", action="store_true")
    args = parser.parse_args()
    asyncio.run(check(args.output, args.capture_left, args.check_concurrent))
