"""Application-level model lifecycle for Hermes communication and specialists."""

from typing import Any

from .model_manager import ModelManager


class HermesModelRuntime:
    def __init__(
        self,
        *,
        brain,
        stt,
        tts,
        vad,
        wake_word=None,
        specialists: dict[str, Any] | None = None,
    ):
        self.brain = brain
        self.stt = stt
        self.tts = tts
        self.models = ModelManager()
        self.models.register_communication_models(
            brain=brain.primary,
            wake_word=wake_word,
            stt=stt,
            tts=tts,
            vad=vad,
        )
        for name, specialist in (specialists or {}).items():
            self.models.register_specialist(name, specialist)

    @classmethod
    def from_configs(
        cls,
        *,
        brain_config: str = "configs/brain.json",
        voice_config: str = "configs/voice.json",
        vision_config: str = "configs/vision.json",
        wake_word=None,
        tools=None,
    ):
        from engines.brain.factory import BrainFactory
        from engines.voice.factory import VoiceProviderFactory
        from engines.vision import VisionFactory

        brain = BrainFactory.from_json(brain_config, tools=tools)
        voice = VoiceProviderFactory.from_json(voice_config)
        vision = VisionFactory.from_json(vision_config)
        specialists = {name: provider for name, provider in vision.items() if name != "stream"}
        runtime = cls(
            brain=brain,
            stt=voice["stt"],
            tts=voice["tts"],
            vad=voice["vad"],
            wake_word=wake_word if wake_word is not None else voice.get("wake_word"),
            specialists=specialists,
        )
        runtime.vision_stream = vision.get("stream")
        runtime.audio_input = voice["audio_input"]
        return runtime

    async def create_live_voice_pipeline(self):
        from engines.voice.audio.command_pipeline import LiveVoiceCommandPipeline

        wake_word = await self.models.get("wake_word")
        return LiveVoiceCommandPipeline(
            audio_source=self.audio_input,
            wake_word=wake_word,
            vad=self.vad,
            stt=self.stt,
            brain=self.brain,
            sample_rate=wake_word.sample_rate,
        )

    async def run_live_voice_once(self):
        pipeline = await self.create_live_voice_pipeline()
        return await pipeline.listen_once()

    async def run_live_voice_forever(self, on_command=None) -> None:
        pipeline = await self.create_live_voice_pipeline()
        await pipeline.run_forever(on_command=on_command)

    async def startup(self) -> None:
        await self.models.startup()

    async def get_specialist(self, name: str):
        if name in {"brain", "wake_word", "stt", "tts", "vad"}:
            raise ValueError(f"{name} is a resident communication model, not a specialist")
        return await self.models.get(name)

    async def shutdown(self) -> None:
        await self.models.shutdown()

    async def __aenter__(self):
        await self.startup()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.shutdown()
        return False