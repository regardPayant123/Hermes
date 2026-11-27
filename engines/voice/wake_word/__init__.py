"""Wake-word detection providers."""

from .interface import WakeWordProvider, WakeWordResult
from .openwakeword import OpenWakeWordProvider

__all__ = ["OpenWakeWordProvider", "WakeWordProvider", "WakeWordResult"]