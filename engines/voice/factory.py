"""Factories and JSON configuration loader for voice providers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class VoiceProviderFactory:
    """Create the configured STT, TTS, and VAD provider implementations."""

    @staticmethod
    def load_config(path: str | Path) -> dict[str, Any]:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Voice configuration not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("Voice configuration must contain a JSON object")
        return config

    @staticmethod
    def create_stt(config: dict[str, Any]):
        provider = str(config.get("provider", "")).lower()
        options = dict(config.get("options", {}))

        if provider == "whisper":
            from .stt.whisper import WhisperSTT
            if "model_path" not in options:
                raise ValueError("Whisper STT requires options.model_path")
            return WhisperSTT(**options)
        if provider == "vosk":
            from .stt.vosk import VoskSTT
            if "model_path" not in options:
                raise ValueError("Vosk STT requires options.model_path")
            return VoskSTT(**options)
        if provider == "google":
            from .stt.google import GoogleSTT
            return GoogleSTT(**options)
        raise ValueError(f"Unknown STT provider: {provider}")

    @staticmethod
    def create_tts(config: dict[str, Any]):
        provider = str(config.get("provider", "")).lower()
        options = dict(config.get("options", {}))

        if provider == "piper":
            from .tts.piper import PiperTTS
            if "model_path" not in options:
                raise ValueError("Piper TTS requires options.model_path")
            return PiperTTS(**options)
        if provider == "elevenlabs":
            from .tts.elevenlabs import ElevenLabsTTS
            return ElevenLabsTTS(**options)
        if provider == "edge_tts":
            from .tts.edge_tts import EdgeTTS
            return EdgeTTS(**options)
        raise ValueError(f"Unknown TTS provider: {provider}")

    @staticmethod
    def create_vad(config: dict[str, Any]):
        provider = str(config.get("provider", "")).lower()
        options = dict(config.get("options", {}))

        if provider == "silero":
            from .vad.silero import SileroVAD
            if "model_path" not in options:
                raise ValueError("Silero VAD requires options.model_path")
            return SileroVAD(**options)
        if provider == "webrtc":
            from .vad.webrtc import WebRTCVAD
            return WebRTCVAD(**options)
        raise ValueError(f"Unknown VAD provider: {provider}")

    @classmethod
    def from_json(cls, path: str | Path):
        config = cls.load_config(path)
        stt_config = config.get("stt", {})
        tts_config = config.get("tts", {})
        vad_config = config.get("vad", {})
        if not all(isinstance(section, dict) for section in (stt_config, tts_config, vad_config)):
            raise ValueError("stt, tts, and vad must contain JSON objects")
        return {
            "stt": cls.create_stt(stt_config),
            "tts": cls.create_tts(tts_config),
            "vad": cls.create_vad(vad_config),
        }
