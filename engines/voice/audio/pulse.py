"""Raw mono PCM capture from PulseAudio-compatible servers using parec."""

import asyncio
from asyncio.subprocess import Process


class PulseAudioStream:
    def __init__(
        self,
        *,
        device: str | None = None,
        command: str = "parec",
        sample_rate: int = 16000,
        channels: int = 1,
        latency_ms: int = 80,
    ):
        if sample_rate <= 0 or channels != 1 or latency_ms <= 0:
            raise ValueError("PulseAudio capture requires positive rate/latency and mono audio")
        self.device = device
        self.command = command
        self.sample_rate = sample_rate
        self.channels = channels
        self.latency_ms = latency_ms
        self._process: Process | None = None

    async def __aenter__(self):
        command = [
            self.command,
            "--raw",
            "--format=s16le",
            f"--rate={self.sample_rate}",
            f"--channels={self.channels}",
            f"--latency-msec={self.latency_ms}",
        ]
        if self.device:
            command.append(f"--device={self.device}")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("parec is unavailable; install pulseaudio-utils and check the PulseAudio/PipeWire server") from exc
        return self

    async def read_frame(self, frame_samples: int) -> bytes:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("PulseAudioStream must be used inside 'async with'")
        if frame_samples <= 0:
            raise ValueError("frame_samples must be greater than zero")
        try:
            return await self._process.stdout.readexactly(frame_samples * 2)
        except asyncio.IncompleteReadError as exc:
            raise RuntimeError("PulseAudio stream ended before a complete PCM frame was read") from exc

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            self._process = None
        return False