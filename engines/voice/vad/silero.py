"""Silero implementation of the VAD provider interface."""

from pathlib import Path

import torch

from .interface import AudioInput, VADProvider, VoiceActivity


class SileroVAD(VADProvider):
    def __init__(self, model_path: str, sample_rate: int = 16000, threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD weights not found: {self.model_path}")
        self._model = torch.jit.load(str(self.model_path), map_location="cpu")
        self._model.eval()
        return self._model

    async def detect(self, audio: AudioInput, *, sample_rate: int = 16000) -> VoiceActivity:
        self.validate_request(audio, sample_rate)
        model = self.load()
        if sample_rate not in {8000, 16000}:
            raise ValueError("Silero supports 8000 or 16000 Hz audio")

        waveform = torch.frombuffer(bytes(audio), dtype=torch.int16).float() / 32768.0
        model.reset_states()
        confidence = 0.0
        chunk_size = 512 if sample_rate == 16000 else 256
        for offset in range(0, waveform.numel(), chunk_size):
            chunk = waveform[offset:offset + chunk_size]
            if chunk.numel() < chunk_size:
                break
            confidence = max(confidence, float(model(chunk, sample_rate).item()))
        is_speech = confidence >= self.threshold
        duration_ms = round(waveform.numel() * 1000 / sample_rate)
        return VoiceActivity(
            is_speech=is_speech,
            confidence=confidence,
            start_ms=0 if is_speech else None,
            end_ms=duration_ms if is_speech else None,
        )
