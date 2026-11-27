"""Contracts for streaming wake-word detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WakeWordResult:
    detected: bool
    keyword: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.keyword.strip():
            raise ValueError("keyword cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


class WakeWordProvider(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def frame_length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def detect(self, audio: bytes, *, sample_rate: int = 16000) -> WakeWordResult:
        """Consume mono signed 16-bit PCM and report a wake-word activation."""
        raise NotImplementedError