"""Long-term memory interface."""

from abc import ABC, abstractmethod

from ..interface import MemoryRecord


class LongTermMemory(ABC):
    @abstractmethod
    async def add(self, record: MemoryRecord) -> str:
        raise NotImplementedError

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, record_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def clear(self) -> None:
        raise NotImplementedError
