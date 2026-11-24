"""Short-term memory interface."""

from abc import ABC, abstractmethod
from collections import deque

from ..interface import MemoryRecord


class ShortTermMemory(ABC):
    @abstractmethod
    async def add(self, record: MemoryRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError
