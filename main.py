from __future__ import annotations

import argparse
import asyncio
import logging
from contextlib import AsyncExitStack

from dotenv import load_dotenv

from chat import (
    MAX_TOOL_CALLS,
    MCP_REAL_CONFIG,
    MCP_SIM_CONFIG,
    MODEL,
    ChatLoop,
    GeminiClient,
    MCPManager,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat Interface - Gemini + MCP")
    parser.add_argument(
        "-s",
        "--sim",
        action="store_true",
        help="Use the simulator MCP server instead of the real one",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("chat")

    config_path = MCP_SIM_CONFIG if args.sim else MCP_REAL_CONFIG
    mode = "SIMULATION" if args.sim else "REAL"

    log.info("Chat Interface starting up [%s]", mode)
    log.info("Model: %s", MODEL)
    log.debug("Config path: %s", config_path)

    client = GeminiClient().get_client()
    log.info("Google API key loaded")

    mcp = MCPManager()
    mcp_config = mcp.load_config(config_path)
    mcp_servers = mcp_config.get("mcpServers", {})

    if not mcp_servers:
        log.warning("No MCP servers configured in %s", config_path.name)
    else:
        log.info("Loading %d server(s) from %s", len(mcp_servers), config_path.name)

    async with AsyncExitStack() as stack:
        await mcp.connect_servers(mcp_config, stack)

        connected = len(mcp.sessions)
        failed = len(mcp_servers) - connected
        if mcp_servers:
            log.info("MCP: %d connected, %d failed", connected, failed)

        tool_count = await mcp.list_tools()

        log.info("%d MCP tool(s) available to %s", tool_count, MODEL)
        log.info("Automatic function calling enabled (max %d calls/turn)", MAX_TOOL_CALLS)

        loop = ChatLoop(client, mcp.session_list, tool_count)
        await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
