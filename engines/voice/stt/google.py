"""Google Cloud Speech-to-Text provider implementation."""

from pathlib import Path

from .interface import STTProvider, Transcript, AudioInput


class GoogleSTT(STTProvider):
    def __init__(self, credentials_path: str | None = None, language: str = "fr-FR", encoding: str = "LINEAR16", sample_rate_hertz: int = 16000):
        self.credentials_path = Path(credentials_path).expanduser() if credentials_path else None
        self.language = language
        self.encoding = encoding
        self.sample_rate_hertz = sample_rate_hertz

    async def transcribe(self, audio: AudioInput, *, language: str | None = None, task="transcribe") -> Transcript:
        if task != "transcribe":
            raise ValueError("Google Speech-to-Text translation must be configured separately")
        if isinstance(audio, (str, Path)):
            content = Path(audio).read_bytes()
        elif isinstance(audio, (bytes, bytearray, memoryview)):
            content = bytes(audio)
        else:
            raise TypeError("Unsupported audio input")

        try:
            from google.cloud import speech
        except ImportError as exc:
            raise RuntimeError("Install Google Speech with: pip install google-cloud-speech") from exc

        if self.credentials_path:
            if not self.credentials_path.is_file():
                raise FileNotFoundError(f"Google credentials not found: {self.credentials_path}")
            import os
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(self.credentials_path)

        client = speech.SpeechClient()
        request = speech.RecognizeRequest(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding[self.encoding],
                sample_rate_hertz=self.sample_rate_hertz,
                language_code=language or self.language,
            ),
            audio=speech.RecognitionAudio(content=content),
        )
        response = client.recognize(request=request)
        alternatives = [result.alternatives[0] for result in response.results if result.alternatives]
        if not alternatives:
            raise RuntimeError("Google Speech-to-Text returned no transcript")
        text = " ".join(alternative.transcript.strip() for alternative in alternatives if alternative.transcript.strip())
        confidence = sum(alternative.confidence for alternative in alternatives if alternative.confidence) / max(1, sum(bool(alternative.confidence) for alternative in alternatives))
        return Transcript(text=text, language=language or self.language, confidence=confidence or None)
