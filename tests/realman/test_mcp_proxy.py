"""Transport failures must not cause an action to be replayed."""

import asyncio
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("mcp")
from mcp import types

from cap_harness.realman.mcp_proxy import proxy_server


def test_new_request_reconnects_but_failed_action_is_not_replayed():
    sessions, calls = [], []

    class Upstream:
        async def list_tools(self):
            return types.ListToolsResult(
                tools=[types.Tool(name="action", inputSchema={"type": "object"})]
            )

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            if len(calls) == 1:
                raise RuntimeError("lost reply; unknown outcome")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="fresh session")], isError=False
            )

    @asynccontextmanager
    async def connect():
        connection = Upstream()
        sessions.append(connection)
        yield connection

    server = proxy_server(connect)

    async def exercise():
        request = types.CallToolRequest(
            method="tools/call", params=types.CallToolRequestParams(name="action", arguments={})
        )
        first = await server.request_handlers[types.CallToolRequest](request)
        assert first.root.isError
        assert len(calls) == 1
        count = len(sessions)
        second = await server.request_handlers[types.CallToolRequest](request)
        assert not second.root.isError
        assert second.root.content[0].text == "fresh session"
        assert len(sessions) > count
        assert len(calls) == 2

    asyncio.run(exercise())
