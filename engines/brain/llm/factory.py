"""Factory for creating configured LLM providers."""

from typing import Any


def create_llm(config: dict[str, Any]):
    provider = str(config.get("provider", "")).lower()
    options = dict(config.get("options", {}))

    if provider == "llama":
        from .llama import LlamaLLM
        if "model_path" not in options:
            raise ValueError("Llama LLM requires options.model_path")
        return LlamaLLM(**options)
    if provider == "qwen":
        from .qwen import QwenLLM
        if "model_path" not in options:
            raise ValueError("Qwen LLM requires options.model_path")
        return QwenLLM(**options)
    raise ValueError(f"Unknown LLM provider: {provider}")
