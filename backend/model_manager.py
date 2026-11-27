"""Explicit eager/lazy lifecycle management for model providers."""

import asyncio
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class ModelState(str, Enum):
    REGISTERED = "registered"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    UNLOADED = "unloaded"


@dataclass(slots=True)
class _ModelEntry:
    model: Any
    load_on_startup: bool
    state: ModelState = ModelState.REGISTERED
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_error: Exception | None = None


class ModelManager:
    """Keep communication models resident and load specialists on demand.

    Register ``brain``, ``wake_word``, ``stt``, ``tts``, and ``vad`` with
    ``load_on_startup=True``. Register other providers as lazy; resolve them
    through ``get`` when a feature actually needs them.
    """

    def __init__(self):
        self._models: dict[str, _ModelEntry] = {}
        self._started = False

    def register_communication_models(
        self,
        *,
        brain: Any,
        stt: Any,
        tts: Any,
        vad: Any,
        wake_word: Any | None = None,
    ) -> None:
        communication_models = {
            "brain": brain,
            "wake_word": wake_word,
            "stt": stt,
            "tts": tts,
            "vad": vad,
        }
        missing = [name for name, model in communication_models.items() if model is None and name != "wake_word"]
        if missing:
            raise ValueError(f"Missing required communication models: {', '.join(missing)}")
        for name, model in communication_models.items():
            if model is not None:
                self.register(name, model, load_on_startup=True)

    def register_specialist(self, name: str, model: Any) -> None:
        self.register(name, model, load_on_startup=False)

    def register(self, name: str, model: Any, *, load_on_startup: bool = False) -> None:
        if not name or not name.strip():
            raise ValueError("model name cannot be empty")
        if name in self._models:
            raise ValueError(f"Model already registered: {name}")
        self._models[name] = _ModelEntry(model=model, load_on_startup=load_on_startup)

    async def startup(self) -> None:
        loaded = []
        try:
            for name, entry in self._models.items():
                if entry.load_on_startup:
                    await self._load(name, entry)
                    loaded.append(name)
            self._started = True
        except Exception:
            for name in reversed(loaded):
                await self.unload(name)
            raise

    async def get(self, name: str) -> Any:
        try:
            entry = self._models[name]
        except KeyError as exc:
            raise KeyError(f"Model is not registered: {name}") from exc
        if entry.state is not ModelState.READY:
            await self._load(name, entry)
        return entry.model

    async def unload(self, name: str) -> None:
        try:
            entry = self._models[name]
        except KeyError as exc:
            raise KeyError(f"Model is not registered: {name}") from exc
        async with entry.lock:
            if entry.state is not ModelState.READY:
                return
            unloader = getattr(entry.model, "unload", None)
            if unloader is not None:
                await _invoke(unloader)
            entry.state = ModelState.UNLOADED

    async def shutdown(self) -> None:
        for name in reversed(tuple(self._models)):
            await self.unload(name)
        self._started = False

    def state(self, name: str) -> ModelState:
        try:
            return self._models[name].state
        except KeyError as exc:
            raise KeyError(f"Model is not registered: {name}") from exc

    async def _load(self, name: str, entry: _ModelEntry) -> None:
        async with entry.lock:
            if entry.state is ModelState.READY:
                return
            entry.state = ModelState.LOADING
            entry.last_error = None
            loader = getattr(entry.model, "load", None)
            try:
                if loader is not None:
                    await _invoke(loader)
                entry.state = ModelState.READY
            except Exception as exc:
                entry.state = ModelState.FAILED
                entry.last_error = exc
                raise RuntimeError(f"Failed to load model '{name}': {exc}") from exc


async def _invoke(method):
    if inspect.iscoroutinefunction(method):
        return await method()
    result = await asyncio.to_thread(method)
    if inspect.isawaitable(result):
        return await result
    return result
