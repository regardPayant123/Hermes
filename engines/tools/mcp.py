"""Lifecycle-managed MCP client connections exposed through ToolRegistry."""

import json
import os
import re
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from .interface import ToolRegistry, ToolSpec


class MCPConnections:
    """Connect configured MCP servers and register their tools by namespace.

    Connections live for the duration of this manager's async context. Server
    processes and remote endpoints are only contacted when explicitly enabled.
    """

    def __init__(self, registry: ToolRegistry, *, timeout_seconds: float = 20.0):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.registry = registry
        self.timeout_seconds = timeout_seconds
        self._stack: AsyncExitStack | None = None
        self._sessions: list[Any] = []

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._stack is not None:
            return await self._stack.__aexit__(exc_type, exc_value, traceback)
        return False

    async def connect_from_config(self, servers: list[dict[str, Any]]) -> None:
        if self._stack is None:
            raise RuntimeError("Use MCPConnections inside an 'async with' block")
        for server in servers:
            if not server.get("enabled", False):
                continue
            name = server.get("name")
            transport = server.get("transport")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Each MCP server requires a non-empty name")
            if transport == "stdio":
                session = await self._connect_stdio(server)
            elif transport in {"streamable_http", "sse"}:
                session = await self._connect_http(server, transport)
            else:
                raise ValueError(f"Unsupported MCP transport: {transport}")
            self._sessions.append(session)
            await self._register_tools(
                name,
                session,
                require_approval=bool(server.get("require_approval", True)),
            )

    async def _connect_stdio(self, config: dict[str, Any]):
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError("Install MCP support with: pip install -r requirements-tools.txt") from exc

        command = config.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("stdio MCP servers require a command")
        arguments = config.get("args", [])
        if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
            raise ValueError("MCP stdio args must be an array of strings")
        environment = config.get("env", {})
        if not isinstance(environment, dict):
            raise ValueError("MCP stdio env must be an object")
        base_environment = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "TMPDIR", "SYSTEMROOT")
            if key in os.environ
        }
        environment = {
            **base_environment,
            **{key: _expand_environment(value) for key, value in environment.items()},
        }

        parameters = StdioServerParameters(
            command=command,
            args=arguments,
            env=environment or None,
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(parameters))
        session = await self._stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
            )
        )
        await session.initialize()
        return session

    async def _connect_http(self, config: dict[str, Any], transport: str):
        try:
            from mcp import ClientSession
            if transport == "sse":
                from mcp.client.sse import sse_client as client_transport
            else:
                from mcp.client.streamable_http import streamablehttp_client as client_transport
        except ImportError as exc:
            raise RuntimeError("Install MCP support with: pip install -r requirements-tools.txt") from exc

        url = config.get("url")
        if not isinstance(url, str) or not url.startswith(("https://", "http://")):
            raise ValueError("MCP HTTP servers require an http(s) URL")
        headers = config.get("headers", {})
        if not isinstance(headers, dict):
            raise ValueError("MCP HTTP headers must be an object")
        headers = {key: _expand_environment(value) for key, value in headers.items()}
        if not all(isinstance(value, str) for value in headers.values()):
            raise ValueError("MCP HTTP header values must be strings")
        transport_context = client_transport(url, headers=headers)
        streams = await self._stack.enter_async_context(transport_context)
        read_stream, write_stream = streams[:2]
        session = await self._stack.enter_async_context(
            ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=self.timeout_seconds),
            )
        )
        await session.initialize()
        return session

    async def _register_tools(self, server_name: str, session, *, require_approval: bool) -> None:
        response = await session.list_tools()
        namespace = re.sub(r"[^a-zA-Z0-9_.]", "_", server_name)
        for remote_tool in response.tools:
            tool_name = re.sub(r"[^a-zA-Z0-9_.]", "_", remote_tool.name)
            qualified_name = f"{namespace}.{tool_name}"
            schema = remote_tool.inputSchema or {"type": "object", "properties": {}}
            annotations = getattr(remote_tool, "annotations", None)
            read_only = bool(getattr(annotations, "readOnlyHint", False))

            async def invoke(arguments, *, connected_session=session, remote_name=remote_tool.name):
                result = await connected_session.call_tool(remote_name, arguments)
                if getattr(result, "isError", False):
                    raise RuntimeError(_mcp_result_content(result))
                structured = getattr(result, "structuredContent", None)
                if structured is not None:
                    return structured
                return {"content": _mcp_result_content(result)}

            self.registry.register(ToolSpec(
                name=qualified_name,
                description=remote_tool.description or f"MCP tool {remote_tool.name} from {server_name}",
                input_schema=schema,
                handler=invoke,
                timeout_seconds=self.timeout_seconds,
                requires_confirmation=require_approval and not read_only,
            ))


def _expand_environment(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    expanded = os.path.expandvars(value)
    if re.search(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", expanded):
        raise ValueError("MCP credentials must be provided through defined environment variables")
    return expanded


def _mcp_result_content(result) -> str:
    pieces = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            pieces.append(text)
        elif hasattr(item, "model_dump"):
            pieces.append(json.dumps(item.model_dump(), ensure_ascii=False, default=str))
    return "\n".join(pieces)