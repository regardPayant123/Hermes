"""Piper implementation of the TTS provider interface."""

import wave
from io import BytesIO
from pathlib import Path

from .interface import SynthesizedSpeech, TTSProvider


class PiperTTS(TTSProvider):
    def __init__(self, model_path: str, config_path: str | None = None, sample_rate: int = 22050):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path) if config_path else self.model_path.with_suffix(".json")
        self.sample_rate = sample_rate

    async def speak(self, text: str, *, language: str | None = None, voice: str | None = None, speed: float = 1.0, format: str = "wav") -> SynthesizedSpeech:
        self.validate_request(text, speed)
        if format.lower() != "wav":
            raise ValueError("Piper currently supports only WAV output")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Piper config not found: {self.config_path}")

        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise RuntimeError("Install Piper with: pip install piper-tts") from exc

        piper_voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path))
        buffer = BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            chunks = piper_voice.synthesize(text)
            wav.writeframes(b"".join(bytes(chunk.audio_int16_bytes) for chunk in chunks))
        return SynthesizedSpeech(buffer.getvalue(), format="wav", sample_rate=self.sample_rate)
