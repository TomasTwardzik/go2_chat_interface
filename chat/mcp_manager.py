from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextlib import AsyncExitStack
    from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

log = logging.getLogger("chat.mcp")


class MCPManager:
    """Manages MCP server connections, lifecycle, and tool discovery."""

    def __init__(self) -> None:
        self._sessions: dict[str, ClientSession] = {}

    @property
    def sessions(self) -> dict[str, ClientSession]:
        return self._sessions

    @property
    def session_list(self) -> list[ClientSession]:
        return list(self._sessions.values())

    @staticmethod
    def load_config(config_path: Path) -> dict[str, Any]:
        """Load MCP server configuration from the given JSON file."""
        log.debug("Loading MCP config from %s", config_path)
        if not config_path.exists():
            log.warning("MCP config file not found: %s", config_path)
            return {}
        with open(config_path) as f:
            data = json.load(f)
        log.debug("MCP config loaded: %d server(s)", len(data.get("mcpServers", {})))
        return data

    async def connect_servers(
        self,
        config: dict[str, Any],
        exit_stack: AsyncExitStack,
    ) -> None:
        """Connect to all MCP servers defined in config. Populates sessions."""
        mcp_servers = config.get("mcpServers", {})
        log.debug("Connecting to %d MCP server(s)", len(mcp_servers))

        for name, server_cfg in mcp_servers.items():
            command = server_cfg["command"]
            args = server_cfg.get("args", [])
            cwd = server_cfg.get("cwd")
            log.info("[%s] Launching: %s %s", name, command, " ".join(args))
            if cwd:
                log.info("[%s] Working dir: %s", name, cwd)

            params = StdioServerParameters(
                command=command,
                args=args,
                env={**os.environ, **server_cfg.get("env", {})},
                cwd=cwd,
            )
            try:
                log.debug("[%s] Opening stdio transport", name)
                read_stream, write_stream = await exit_stack.enter_async_context(
                    stdio_client(params)
                )
                log.debug("[%s] Creating ClientSession", name)
                session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                log.debug("[%s] Initializing session", name)
                result = await session.initialize()
                server_info = result.serverInfo if result else None
                if server_info:
                    version = f" v{server_info.version}" if server_info.version else ""
                    log.info("[%s] Connected: %s%s", name, server_info.name, version)
                else:
                    log.info("[%s] Connected (no server info)", name)

                capabilities = session.get_server_capabilities()
                log.debug("[%s] Capabilities: %s", name, capabilities)
                if capabilities and capabilities.tools:
                    log.info("[%s] Capabilities: tools supported", name)
                self._sessions[name] = session
            except Exception as e:
                log.error("[%s] FAILED to connect: %s", name, e)

        log.debug("Connected to %d/%d server(s)", len(self._sessions), len(mcp_servers))

    async def list_tools(self) -> int:
        """Discover and cache tools from all connected sessions. Returns total count.

        The google-genai SDK calls session.list_tools() on every generate_content()
        call. By caching the result here we avoid repeated ListToolsRequest RPCs.
        """
        total = 0
        for server_name, session in self._sessions.items():
            log.debug("[%s] Listing tools", server_name)
            result = await session.list_tools()
            count = len(result.tools)
            total += count
            log.info("[%s] Discovered %d tool(s)", server_name, count)
            for tool in result.tools:
                desc = f" - {tool.description}" if tool.description else ""
                log.info("  - %s%s", tool.name, desc)

            cached = result
            log.debug("[%s] Caching list_tools result (%d tools)", server_name, count)

            async def _cached_list_tools(_cached: Any = cached) -> Any:
                return _cached

            session.list_tools = _cached_list_tools  # type: ignore[assignment]
        log.debug("Total tools discovered: %d", total)
        return total
