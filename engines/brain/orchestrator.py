"""Orchestrates LLM selection, conversation state, and fallback behavior."""

from collections import deque
from typing import Iterable

from .llm.interface import ChatMessage, LLMProvider, LLMResponse


class LLMOrchestrator:
    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: Iterable[LLMProvider] = (),
        *,
        system_prompt: str | None = None,
        history_limit: int = 20,
    ):
        if history_limit <= 0:
            raise ValueError("history_limit must be greater than zero")
        self.primary = primary
        self.fallbacks = tuple(fallbacks)
        self.system_prompt = system_prompt
        self.history: deque[ChatMessage] = deque(maxlen=history_limit)

    def clear_history(self) -> None:
        self.history.clear()

    def set_primary(self, provider: LLMProvider) -> None:
        self.primary = provider

    def _messages(self, user_message: str) -> list[ChatMessage]:
        messages = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.extend(self.history)
        messages.append(ChatMessage(role="user", content=user_message))
        return messages

    async def run(
        self,
        user_message: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        if not user_message.strip():
            raise ValueError("user_message cannot be empty")

        messages = self._messages(user_message)
        failures = []

        for provider in (self.primary, *self.fallbacks):
            try:
                response = await provider.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.history.append(ChatMessage(role="user", content=user_message))
                self.history.append(ChatMessage(role="assistant", content=response.text))
                return response
            except Exception as exc:
                failures.append(f"{provider.__class__.__name__}: {exc}")

        raise RuntimeError("All LLM providers failed: " + " | ".join(failures))
