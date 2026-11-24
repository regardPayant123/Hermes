"""Common interface for local and remote language models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("message content cannot be empty")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: str = "stop"


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
    ) -> LLMResponse:
        """Generate a response for a conversation."""
        raise NotImplementedError

    def validate_generation(self, temperature: float, max_tokens: int) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
