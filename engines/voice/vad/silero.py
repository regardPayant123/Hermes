"""Silero implementation of the VAD provider interface."""

from pathlib import Path
import asyncio

from .interface import AudioInput, VADProvider, VoiceActivity


class SileroVAD(VADProvider):
    def __init__(self, model_path: str, sample_rate: int = 16000, threshold: float = 0.5):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model = None
        self._torch = None
        self._stream_remainder = None
        self._stream_sample_rate = None

    def load(self):
        if self._model is not None:
            return self._model
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch with: pip install torch") from exc
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Silero VAD weights not found: {self.model_path}")
        self._model = torch.jit.load(str(self.model_path), map_location="cpu")
        self._model.eval()
        self._torch = torch
        return self._model

    def unload(self) -> None:
        self._model = None
        self._torch = None
        self._stream_remainder = None
        self._stream_sample_rate = None

    def reset_stream(self) -> None:
        model = self.load()
        model.reset_states()
        self._stream_remainder = None
        self._stream_sample_rate = None

    async def detect_stream(self, audio: AudioInput, *, sample_rate: int = 16000) -> VoiceActivity:
        self.validate_request(audio, sample_rate)
        if sample_rate not in {8000, 16000}:
            raise ValueError("Silero supports 8000 or 16000 Hz audio")
        return await asyncio.to_thread(self._detect_stream_sync, bytes(audio), sample_rate)

    def _detect_stream_sync(self, audio: bytes, sample_rate: int) -> VoiceActivity:
        import numpy as np

        model = self.load()
        torch = self._torch
        if self._stream_sample_rate != sample_rate:
            model.reset_states()
            self._stream_remainder = np.empty(0, dtype=np.int16)
            self._stream_sample_rate = sample_rate

        samples = np.frombuffer(audio, dtype=np.int16)
        if self._stream_remainder.size:
            samples = np.concatenate((self._stream_remainder, samples))
        chunk_size = 512 if sample_rate == 16000 else 256
        complete_samples = (samples.size // chunk_size) * chunk_size
        self._stream_remainder = samples[complete_samples:].copy()
        confidence = 0.0
        for offset in range(0, complete_samples, chunk_size):
            chunk = torch.from_numpy(samples[offset:offset + chunk_size].copy()).float() / 32768.0
            confidence = max(confidence, float(model(chunk, sample_rate).item()))
        duration_ms = round(samples.size * 1000 / sample_rate)
        is_speech = confidence >= self.threshold
        return VoiceActivity(
            is_speech=is_speech,
            confidence=confidence,
            start_ms=0 if is_speech else None,
            end_ms=duration_ms if is_speech else None,
        )

    async def detect(self, audio: AudioInput, *, sample_rate: int = 16000) -> VoiceActivity:
        self.validate_request(audio, sample_rate)
        model = self.load()
        torch = self._torch
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
