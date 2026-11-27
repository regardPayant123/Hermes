"""Common interface for local and remote language models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import Any, Iterable, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if not self.content.strip() and not self.tool_calls and self.role != "tool":
            raise ValueError("message content cannot be empty")

    def to_provider_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": self.role, "content": self.content or None}
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        return message


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: str = "stop"
    tool_calls: tuple[ToolCall, ...] = ()


class LLMProvider(ABC):
    """Common interface for llama.cpp, transformers, and API-backed LLMs."""

    @abstractmethod
    async def generate(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 20000,
        stop: tuple[str, ...] = (),
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Generate a response for a conversation."""
        raise NotImplementedError

    def validate_generation(self, temperature: float, max_tokens: int) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")


def parse_tool_calls(raw_calls: list[dict[str, Any]]) -> tuple[ToolCall, ...]:
    calls = []
    for raw_call in raw_calls:
        function = raw_call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a JSON object")
        calls.append(ToolCall(
            id=str(raw_call.get("id", "")),
            name=str(function.get("name", "")),
            arguments=arguments,
        ))
    return tuple(calls)
