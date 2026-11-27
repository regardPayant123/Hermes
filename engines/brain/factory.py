"""Factory and JSON configuration loader for the brain engine."""

import json
from pathlib import Path
from typing import Any


class BrainFactory:
    @staticmethod
    def load_config(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Brain configuration not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("Brain configuration must contain a JSON object")
        return config

    @staticmethod
    def create_llm(config: dict[str, Any]):
        from .llm.factory import create_llm
        return create_llm(config)

    @classmethod
    def from_json(cls, path: str | Path, *, tools=None):
        from .orchestrator import LLMOrchestrator

        config = cls.load_config(path)
        primary_config = config.get("primary", {})
        fallback_configs = config.get("fallbacks", [])
        orchestrator_config = config.get("orchestrator", {})

        if not isinstance(primary_config, dict):
            raise ValueError("primary must contain a JSON object")
        if not isinstance(fallback_configs, list):
            raise ValueError("fallbacks must contain a JSON array")

        primary = cls.create_llm(primary_config)
        fallbacks = [cls.create_llm(section) for section in fallback_configs]
        return LLMOrchestrator(
            primary,
            fallbacks,
            system_prompt=orchestrator_config.get("system_prompt"),
            history_limit=orchestrator_config.get("history_limit", 20),
            tools=tools,
            max_tool_rounds=orchestrator_config.get("max_tool_rounds", 4),
        )
