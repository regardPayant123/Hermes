"""Safe built-in tools for Hermes."""

import ast
import math
from datetime import datetime, timezone
from typing import Any

from .interface import ApprovalCallback, ToolRegistry, ToolSpec
from .web_search import BraveSearchProvider, SearchProvider, TavilySearchProvider, create_web_search_tool


def create_tool_registry(
    *,
    rag=None,
    web_search: dict[str, Any] | None = None,
    search_provider: SearchProvider | None = None,
    enabled_tools: set[str] | None = None,
    approval_callback: ApprovalCallback | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(approval_callback=approval_callback)
    enabled = enabled_tools

    async def current_time(_arguments: dict[str, Any]):
        return {"utc": datetime.now(timezone.utc).isoformat()}

    if enabled is None or "current_time" in enabled:
        registry.register(ToolSpec(
            name="current_time",
            description="Get the current UTC time.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=current_time,
        ))
    if enabled is None or "calculator" in enabled:
        registry.register(_calculator_tool())

    if rag is not None and (enabled is None or "rag_search" in enabled):
        async def rag_search(arguments: dict[str, Any]):
            chunks = await rag.retrieve(arguments["query"], limit=arguments.get("limit", 5))
            return {
                "results": [
                    {
                        "content": chunk.content,
                        "source": chunk.source,
                        "score": chunk.score,
                    }
                    for chunk in chunks
                ]
            }

        registry.register(ToolSpec(
            name="rag_search",
            description="Search Hermes' local indexed documents and return relevant passages with sources.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 2, "maxLength": 1000},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=rag_search,
            max_output_chars=10000,
        ))

    search_config = web_search or {}
    if search_config.get("enabled", True) and (enabled is None or "web_search" in enabled):
        if search_provider is None:
            provider_name = str(search_config.get("provider", "tavily")).lower()
            if provider_name == "tavily":
                search_provider = TavilySearchProvider(timeout_seconds=float(search_config.get("timeout_seconds", 12)))
            elif provider_name == "brave":
                search_provider = BraveSearchProvider(timeout_seconds=float(search_config.get("timeout_seconds", 12)))
            else:
                raise ValueError(f"Unknown web search provider: {provider_name}")
        registry.register(create_web_search_tool(search_provider))
    return registry


def _calculator_tool() -> ToolSpec:
    async def calculate(arguments: dict[str, Any]):
        expression = arguments["expression"]
        if len(expression) > 256:
            raise ValueError("expression is too long")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 64:
            raise ValueError("expression is too complex")
        result = _evaluate(tree.body)
        if not math.isfinite(result):
            raise ValueError("result is not finite")
        return {"result": result}

    return ToolSpec(
        name="calculator",
        description="Evaluate a basic arithmetic expression. No names, functions, or code execution are allowed.",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string", "minLength": 1, "maxLength": 256}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=calculate,
        max_output_chars=1000,
    )


def _evaluate(node: ast.AST) -> float:
    binary = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.FloorDiv: lambda left, right: left // right,
        ast.Mod: lambda left, right: left % right,
        ast.Pow: lambda left, right: left ** right,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in binary:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("exponent magnitude cannot exceed 12")
        result = binary[type(node.op)](left, right)
        if abs(result) > 1e100:
            raise ValueError("result magnitude is too large")
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("only numeric literals and basic arithmetic operators are allowed")