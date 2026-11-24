"""In-memory short-term conversation buffer."""

from collections import deque
from uuid import uuid4

from ..interface import MemoryRecord
from .interface import ShortTermMemory


class BufferShortTermMemory(ShortTermMemory):
    def __init__(self, max_records: int = 50):
        if max_records <= 0:
            raise ValueError("max_records must be greater than zero")
        self._records: deque[MemoryRecord] = deque(maxlen=max_records)

    async def add(self, record: MemoryRecord) -> str:
        record_id = str(uuid4())
        self._records.append(record)
        return record_id

    async def get_recent(self, limit: int = 10) -> list[MemoryRecord]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        return list(self._records)[-limit:]

    async def clear(self) -> None:
        self._records.clear()
