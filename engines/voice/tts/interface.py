# engines/voice/tts/interface.py

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SynthesizedSpeech:
    """Audio produced by a text-to-speech provider."""

    audio: bytes
    format: str = "wav"
    sample_rate: int | None = None

    def __post_init__(self) -> None:
        if not self.audio:
            raise ValueError("audio cannot be empty")
        if not self.format or not self.format.strip():
            raise ValueError("audio format cannot be empty")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")


class TTSProvider(ABC):
    """Common interface for text-to-speech providers."""

    @abstractmethod
    async def speak(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        speed: float = 1.0,
        format: str = "wav",
    ) -> SynthesizedSpeech:
        """Synthesize text into playable audio.

        Args:
            text: Text to convert to speech.
            language: Optional provider language code, for example ``"fr"``.
            voice: Optional provider-specific voice identifier.
            speed: Relative speech speed. Providers use 1.0 as normal speed.
            format: Requested output container or encoding, for example
                ``"wav"`` or ``"mp3"``.

        Raises:
            ValueError: If the text or synthesis options are invalid.
            RuntimeError: If the provider fails to synthesize the audio.
        """
        raise NotImplementedError

    def validate_request(self, text: str, speed: float) -> None:
        """Validate options shared by every TTS provider."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text cannot be empty")
        if not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")