"""Qwen family provider using local GGUF weights through llama.cpp."""

from pathlib import Path
from typing import Iterable

from .interface import ChatMessage, LLMProvider, LLMResponse


QWEN_TIERS = {
    "light": "Qwen2.5-1.5B-Instruct",
    "middle": "Qwen2.5-7B-Instruct",
    "heavy": "Qwen2.5-72B-Instruct",
}


class QwenLLM(LLMProvider):
    def __init__(
        self,
        model_path: str,
        tier: str = "light",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
    ):
        if tier not in QWEN_TIERS:
            raise ValueError("Qwen tier must be light, middle, or heavy")
        self.model_path = Path(model_path)
        self.tier = tier
        self.model_name = QWEN_TIERS[tier]
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("Install llama.cpp with: pip install llama-cpp-python") from exc
        if not str(self.model_path) or not self.model_path.is_file():
            raise FileNotFoundError(f"Qwen GGUF weights not found: {self.model_path}")
        self._model = Llama(
            model_path=str(self.model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            chat_format="chatml",
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
