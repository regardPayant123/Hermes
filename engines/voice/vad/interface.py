# engines/voice/vad/interface.py

from abc import ABC, abstractmethod
from dataclasses import dataclass


AudioInput = bytes | bytearray | memoryview


@dataclass(frozen=True, slots=True)
class VoiceActivity:
    """Result of voice activity detection for an audio chunk."""

    is_speech: bool
    confidence: float
    start_ms: int | None = None
    end_ms: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.start_ms is not None and self.start_ms < 0:
            raise ValueError("start_ms cannot be negative")
        if self.end_ms is not None and self.end_ms < 0:
            raise ValueError("end_ms cannot be negative")
        if (
            self.start_ms is not None
            and self.end_ms is not None
            and self.end_ms < self.start_ms
        ):
            raise ValueError("end_ms cannot be earlier than start_ms")


class VADProvider(ABC):
    """Common interface for voice activity detection providers."""

    @abstractmethod
    async def detect(
        self,
        audio: AudioInput,
        *,
        sample_rate: int = 16000,
    ) -> VoiceActivity:
        """Analyze an audio chunk and report whether it contains speech.

        Args:
            audio: Raw encoded or PCM audio bytes.
            sample_rate: Audio sample rate in Hertz.

        Raises:
            TypeError: If the audio input type is unsupported.
            ValueError: If the audio or sample rate is invalid.
            RuntimeError: If the provider fails to analyze the audio.
        """
        raise NotImplementedError

    def validate_request(self, audio: AudioInput, sample_rate: int) -> None:
        """Validate options shared by every VAD provider."""
        if not isinstance(audio, (bytes, bytearray, memoryview)):
            raise TypeError("audio must be bytes, bytearray, or memoryview")
        if not audio:
            raise ValueError("audio cannot be empty")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")