"""Configuration factory and lifecycle wrapper for tool runtimes."""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .builtins import create_tool_registry
from .interface import ApprovalCallback, ToolRegistry
from .mcp import MCPConnections


@dataclass(slots=True)
class ToolRuntime:
    registry: ToolRegistry
    mcp_connections: MCPConnections
    mcp_servers: list[dict[str, Any]]
    max_tool_rounds: int = 4

    async def __aenter__(self):
        await self.mcp_connections.__aenter__()
        try:
            await self.mcp_connections.connect_from_config(self.mcp_servers)
        except Exception:
            await self.mcp_connections.__aexit__(*sys.exc_info())
            raise
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return await self.mcp_connections.__aexit__(exc_type, exc_value, traceback)


class ToolFactory:
    @staticmethod
    def load_config(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Tools configuration not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("Tools configuration must contain a JSON object")
        return config

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        rag=None,
        approval_callback: ApprovalCallback | None = None,
    ) -> ToolRuntime:
        config = cls.load_config(path)
        web_config = config.get("web_search", {})
        mcp_config = config.get("mcp", {})
        runtime_config = config.get("orchestrator", {})
        enabled = config.get("enabled_tools")

        if not isinstance(web_config, dict) or not isinstance(mcp_config, dict):
            raise ValueError("web_search and mcp configuration must be JSON objects")
        if not isinstance(runtime_config, dict):
            raise ValueError("orchestrator configuration must be a JSON object")
        if enabled is not None and (not isinstance(enabled, list) or not all(isinstance(item, str) for item in enabled)):
            raise ValueError("enabled_tools must be an array of names")
        servers = mcp_config.get("servers", [])
        if not isinstance(servers, list) or not all(isinstance(server, dict) for server in servers):
            raise ValueError("mcp.servers must be an array of objects")

        registry = create_tool_registry(
            rag=rag,
            web_search=web_config,
            enabled_tools=set(enabled) if enabled is not None else None,
            approval_callback=approval_callback,
        )
        manager = MCPConnections(
            registry,
            timeout_seconds=float(mcp_config.get("timeout_seconds", 20.0)),
        )
        return ToolRuntime(
            registry=registry,
            mcp_connections=manager,
            mcp_servers=servers,
            max_tool_rounds=int(runtime_config.get("max_tool_rounds", 4)),
        )