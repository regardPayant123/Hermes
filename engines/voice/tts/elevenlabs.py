"""ElevenLabs implementation of the TTS provider interface."""

import os

from .interface import SynthesizedSpeech, TTSProvider


class ElevenLabsTTS(TTSProvider):
    def __init__(self, api_key: str | None = None, voice: str = "21m00Tcm4TlvDq8ikWAM", model: str = "eleven_multilingual_v2"):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.voice = voice
        self.model = model

    async def speak(self, text: str, *, language: str | None = None, voice: str | None = None, speed: float = 1.0, format: str = "wav") -> SynthesizedSpeech:
        self.validate_request(text, speed)
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key is missing")
        try:
            from elevenlabs import ElevenLabs
        except ImportError as exc:
            raise RuntimeError("Install ElevenLabs with: pip install elevenlabs") from exc

        client = ElevenLabs(api_key=self.api_key)
        audio_format = "pcm_16000" if format.lower() == "wav" else format.lower()
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice or self.voice,
            model_id=self.model,
            output_format=audio_format,
        )
        data = b"".join(audio)
        if not data:
            raise RuntimeError("ElevenLabs returned empty audio")
        return SynthesizedSpeech(data, format=format, sample_rate=16000 if format.lower() == "wav" else None)
