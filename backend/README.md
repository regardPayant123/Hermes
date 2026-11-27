# Model Runtime

`HermesModelRuntime` owns model lifetimes for the application. It eagerly loads the brain primary, wake-word provider, STT provider, TTS provider, and VAD at startup. Vision, OCR, VLM, and other specialists remain lazy and are loaded with `get_specialist(name)`. LLM fallback providers remain lazy through the brain orchestrator.

```python
from backend.runtime import HermesModelRuntime

runtime = HermesModelRuntime(
    brain=brain,
    wake_word=wake_word_provider,
    stt=voice_providers["stt"],
    tts=voice_providers["tts"],
    vad=voice_providers["vad"],
    specialists={
        **vision_providers,
    },
)

async with runtime:
    response = await brain.run("Bonjour")
```

Providers with synchronous `load()` methods are initialized off the asyncio event loop. Providers can expose `unload()` to release their weights on application shutdown. Piper now caches its voice between `speak()` calls; Whisper and the local LLM providers expose unload hooks, and Silero imports PyTorch only when loaded.

The repository does not yet include a wake-word provider implementation. Until one is added, pass the selected provider from the voice application when constructing the runtime; do not silently substitute STT for wake-word detection.
