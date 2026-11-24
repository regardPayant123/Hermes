"""Llama family provider using llama.cpp bindings."""

from pathlib import Path
from typing import Iterable

from .interface import ChatMessage, LLMProvider, LLMResponse


LLAMA_TIERS = {
    "light": "Llama-3.2-1B-Instruct",
    "middle": "Llama-3.1-8B-Instruct",
    "heavy": "Llama-3.3-70B-Instruct",
}


class LlamaLLM(LLMProvider):
    def __init__(
        self,
        model_path: str,
        tier: str = "light",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        chat_format: str | None = None,
    ):
        if tier not in LLAMA_TIERS:
            raise ValueError("Llama tier must be light, middle, or heavy")
        self.model_path = Path(model_path)
        self.tier = tier
        self.model_name = LLAMA_TIERS[tier]
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.chat_format = chat_format
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install llama.cpp with: pip install llama-cpp-python") from exc
        if not str(self.model_path) or not self.model_path.is_file():
            raise FileNotFoundError(f"Llama GGUF weights not found: {self.model_path}")
        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            chat_format=self.chat_format,
        )
        return self._model

    async def generate(
        self,
        messages: Iterable[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stop: tuple[str, ...] = (),
    ) -> LLMResponse:
        self.validate_generation(temperature, max_tokens)
        formatted = [{"role": message.role, "content": message.content} for message in messages]
        model = self.load()
        result = model.create_chat_completion(
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=list(stop) or None,
        )
        choice = result["choices"][0]
        text = choice["message"]["content"]
        return LLMResponse(
            text=text,
            model=self.model_name,
            finish_reason=choice.get("finish_reason", "stop"),
        )
