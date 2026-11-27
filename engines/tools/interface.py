"""Tool definitions and guarded execution registry."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

if TYPE_CHECKING:
    from engines.brain.llm.interface import ToolCall


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]
ApprovalCallback = Callable[["ToolSpec", dict[str, Any]], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    timeout_seconds: float = 15.0
    max_output_chars: int = 12000
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").replace(".", "").isalnum():
            raise ValueError("tool name must contain only letters, numbers, underscores, or dots")
        if not self.description.strip():
            raise ValueError("tool description cannot be empty")
        if self.timeout_seconds <= 0 or self.max_output_chars <= 0:
            raise ValueError("tool timeout and output limit must be positive")
        if self.input_schema.get("type") != "object":
            raise ValueError("tool input_schema must have type 'object'")

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolRegistry:
    def __init__(self, approval_callback: ApprovalCallback | None = None):
        self._tools: dict[str, ToolSpec] = {}
        self.approval_callback = approval_callback

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ChatMessage:
        from engines.brain.llm.interface import ChatMessage

        tool = self._tools.get(call.name)
        if tool is None:
            return self._tool_message(call, {"error": f"Tool not allowed: {call.name}"})
        try:
            _validate_arguments(tool.input_schema, call.arguments)
            if tool.requires_confirmation:
                if self.approval_callback is None:
                    raise PermissionError("This tool requires user approval; no approval handler is configured")
                approved = await self.approval_callback(tool, call.arguments)
                if not approved:
                    raise PermissionError("User approval was not granted")
            result = await asyncio.wait_for(
                tool.handler(call.arguments),
                timeout=tool.timeout_seconds,
            )
            content = _serialize_result(result, tool.max_output_chars)
        except Exception as exc:
            content = _serialize_result({"error": str(exc)}, tool.max_output_chars)
        return ChatMessage(
            role="tool",
            name=call.name,
            tool_call_id=call.id,
            content=content,
        )

    @staticmethod
    def _tool_message(call: ToolCall, result: dict[str, str]) -> ChatMessage:
        from engines.brain.llm.interface import ChatMessage

        return ChatMessage(
            role="tool",
            name=call.name,
            tool_call_id=call.id,
            content=json.dumps(result, ensure_ascii=False),
        )


def _validate_arguments(schema: dict[str, Any], value: Any, path: str = "arguments") -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    if schema.get("additionalProperties") is False:
        unknown = set(value) - set(properties)
        if unknown:
            raise ValueError(f"Unknown arguments: {', '.join(sorted(unknown))}")
    for key, item in value.items():
        if key in properties:
            _validate_value(properties[key], item, f"{path}.{key}")


def _validate_value(schema: dict[str, Any], value: Any, path: str) -> None:
    expected = schema.get("type")
    validators = {
        "string": lambda item: isinstance(item, str),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "array": lambda item: isinstance(item, list),
        "object": lambda item: isinstance(item, dict),
    }
    if expected in validators and not validators[expected](value):
        raise ValueError(f"{path} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if expected == "string":
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path} is too short")
        if len(value) > schema.get("maxLength", float("inf")):
            raise ValueError(f"{path} is too long")
    if expected in {"integer", "number"}:
        if value < schema.get("minimum", float("-inf")):
            raise ValueError(f"{path} is below its minimum")
        if value > schema.get("maximum", float("inf")):
            raise ValueError(f"{path} exceeds its maximum")
    if expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(schema["items"], item, f"{path}[{index}]")
    if expected == "object":
        _validate_arguments(schema, value, path)


def _serialize_result(result: Any, max_chars: int) -> str:
    try:
        serialized = json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        serialized = json.dumps(str(result), ensure_ascii=False)
    if len(serialized) > max_chars:
        preview_size = max(0, max_chars - 40)
        serialized = json.dumps(
            {"truncated": True, "preview": serialized[:preview_size]},
            ensure_ascii=False,
        )
    return serialized