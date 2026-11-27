"""Run Hermes live voice commands through PulseAudio."""

import argparse
import asyncio

from .runtime import HermesModelRuntime


def _arguments():
    parser = argparse.ArgumentParser(description="Listen for Hey Jarvis over PulseAudio")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Handle one voice command and exit; default is continuous listening",
    )
    parser.add_argument(
        "--device",
        help="PulseAudio/PipeWire source name; defaults to audio_input.options.device",
    )
    return parser.parse_args()


async def _run(once: bool, device: str | None) -> None:
    runtime = HermesModelRuntime.from_configs()
    if device:
        runtime.audio_input.device = device

    async with runtime:
        if once:
            result = await runtime.run_live_voice_once()
            print(f"Wake word: {result.wake_word}")
            print(f"You: {result.transcript.text}")
            print(f"Hermes: {result.response.text}")
            return

        async def print_command(result):
            print(f"Wake word: {result.wake_word}")
            print(f"You: {result.transcript.text}")
            print(f"Hermes: {result.response.text}\n")

        print("Listening for Hey Jarvis. Press Ctrl+C to stop.")
        await runtime.run_live_voice_forever(on_command=print_command)


def main() -> None:
    arguments = _arguments()
    try:
        asyncio.run(_run(arguments.once, arguments.device))
    except KeyboardInterrupt:
        print("Voice listener stopped.")


if __name__ == "__main__":
    main()
