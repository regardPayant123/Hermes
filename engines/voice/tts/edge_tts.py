"""Microsoft Edge TTS implementation of the TTS provider interface."""

from .interface import SynthesizedSpeech, TTSProvider


class EdgeTTS(TTSProvider):
    def __init__(self, voice: str = "fr-FR-HenriNeural"):
        self.voice = voice

    async def speak(self, text: str, *, language: str | None = None, voice: str | None = None, speed: float = 1.0, format: str = "wav") -> SynthesizedSpeech:
        self.validate_request(text, speed)
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("Install Edge TTS with: pip install edge-tts") from exc

        if format.lower() not in {"wav", "mp3"}:
            raise ValueError("Edge TTS supports WAV or MP3 output")
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice or self.voice,
            rate=f"{int((speed - 1.0) * 100):+d}%",
        )
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        data = b"".join(chunks)
        if not data:
            raise RuntimeError("Edge TTS returned empty audio")
        return SynthesizedSpeech(data, format=format.lower())
