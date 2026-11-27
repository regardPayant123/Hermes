"""openWakeWord provider using local ONNX wake-word and feature models."""

import asyncio
import time
from pathlib import Path

from .interface import WakeWordProvider, WakeWordResult


class OpenWakeWordProvider(WakeWordProvider):
    def __init__(
        self,
        model_path: str,
        melspectrogram_path: str,
        embedding_path: str,
        keyword: str = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        frame_samples: int = 1280,
        cooldown_seconds: float = 1.5,
        vad_threshold: float = 0.0,
        ncpu: int = 1,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if sample_rate != 16000:
            raise ValueError("openWakeWord models require a 16000 Hz sample rate")
        if frame_samples <= 0 or frame_samples % 1280 != 0:
            raise ValueError("frame_samples must be a positive multiple of 1280")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        if not 0.0 <= vad_threshold <= 1.0:
            raise ValueError("vad_threshold must be between 0 and 1")
        if ncpu <= 0:
            raise ValueError("ncpu must be greater than zero")

        self.model_path = Path(model_path)
        self.melspectrogram_path = Path(melspectrogram_path)
        self.embedding_path = Path(embedding_path)
        self.keyword = keyword
        self.threshold = threshold
        self.frame_samples = frame_samples
        self.cooldown_seconds = cooldown_seconds
        self.vad_threshold = vad_threshold
        self.ncpu = ncpu
        self._model = None
        self._pending = None
        self._last_detection = float("-inf")

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def frame_length(self) -> int:
        return self.frame_samples

    def load(self):
        if self._model is not None:
            return self._model
        for path in (self.model_path, self.melspectrogram_path, self.embedding_path):
            if not path.is_file():
                raise FileNotFoundError(f"openWakeWord model file not found: {path}")
        try:
            from openwakeword.model import Model
        except ImportError as exc:
            raise RuntimeError("Install openWakeWord with: pip install openwakeword") from exc

        self._model = Model(
            wakeword_models=[str(self.model_path)],
            inference_framework="onnx",
            melspec_model_path=str(self.melspectrogram_path),
            embedding_model_path=str(self.embedding_path),
            vad_threshold=self.vad_threshold,
            ncpu=self.ncpu,
        )
        return self._model

    def unload(self) -> None:
        self._model = None
        self._pending = None
        self._last_detection = float("-inf")

    async def detect(self, audio: bytes, *, sample_rate: int = 16000) -> WakeWordResult:
        if not isinstance(audio, bytes):
            raise TypeError("audio must be bytes containing signed 16-bit PCM samples")
        if not audio:
            raise ValueError("audio cannot be empty")
        if sample_rate != self.sample_rate:
            raise ValueError("openWakeWord input must be sampled at 16000 Hz")
        if len(audio) % 2:
            raise ValueError("PCM int16 audio must contain an even number of bytes")

        import numpy as np

        model = self.load()
        samples = np.frombuffer(audio, dtype=np.int16)
        if self._pending is not None and self._pending.size:
            samples = np.concatenate((self._pending, samples))
        complete_samples = (samples.size // self.frame_samples) * self.frame_samples
        self._pending = samples[complete_samples:].copy()
        if complete_samples == 0:
            return WakeWordResult(False, self.keyword, 0.0)

        triggered = False
        confidence = 0.0
        for offset in range(0, complete_samples, self.frame_samples):
            frame = np.ascontiguousarray(samples[offset:offset + self.frame_samples])
            scores = await asyncio.to_thread(model.predict, frame)
            confidence = max((float(score) for score in scores.values()), default=0.0)
            now = time.monotonic()
            if confidence >= self.threshold and now - self._last_detection >= self.cooldown_seconds:
                triggered = True
                self._last_detection = now

        return WakeWordResult(triggered, self.keyword, confidence)