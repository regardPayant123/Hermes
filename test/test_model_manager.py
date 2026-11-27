import asyncio

from backend.model_manager import ModelManager, ModelState


class FakeModel:
    def __init__(self):
        self.load_count = 0
        self.unload_count = 0

    def load(self):
        self.load_count += 1

    def unload(self):
        self.unload_count += 1


def test_startup_loads_only_resident_models_and_get_loads_specialists():
    async def run():
        manager = ModelManager()
        brain = FakeModel()
        wake_word = FakeModel()
        stt = FakeModel()
        tts = FakeModel()
        vad = FakeModel()
        vision = FakeModel()

        manager.register_communication_models(
            brain=brain,
            wake_word=wake_word,
            stt=stt,
            tts=tts,
            vad=vad,
        )
        manager.register_specialist("vision", vision)

        await manager.startup()
        assert [model.load_count for model in (brain, wake_word, stt, tts, vad)] == [1, 1, 1, 1, 1]
        assert vision.load_count == 0
        assert manager.state("vision") is ModelState.REGISTERED

        assert await manager.get("vision") is vision
        assert await manager.get("vision") is vision
        assert vision.load_count == 1
        assert manager.state("vision") is ModelState.READY

        await manager.shutdown()
        assert [model.unload_count for model in (brain, wake_word, stt, tts, vad, vision)] == [1, 1, 1, 1, 1, 1]
        assert manager.state("brain") is ModelState.UNLOADED

    asyncio.run(run())
