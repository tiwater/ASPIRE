"""Per-connection STDIO proxy to one robot-local MCP owner; opens no ROS clients."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

BACKEND_URL = "http://127.0.0.1:8232/mcp"


@asynccontextmanager
async def connect_upstream():
    # Each request gets a fresh session, so a supervised backend restart does not
    # strand an already connected STDIO client. Never retry a failed tool call:
    # a lost motion response has an unknown outcome.
    async with (
        streamable_http_client(BACKEND_URL) as (reader, writer, _),
        ClientSession(reader, writer, read_timeout_seconds=timedelta(seconds=40)) as upstream,
    ):
        await upstream.initialize()
        yield upstream


def proxy_server(connect=connect_upstream):
    server = Server("ASPIRE RealMan connection")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        async with connect() as upstream:
            result = await upstream.list_tools()
        return result.tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
        # Preserve structured output, images and isError, not just a success string.
        async with connect() as upstream:
            return await upstream.call_tool(name, arguments)

    return server


async def run():
    server = proxy_server()
    async with stdio_server() as (stdin, stdout):
        await server.run(stdin, stdout, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(run())
