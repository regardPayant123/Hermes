"""Shared contracts for memory systems."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("memory content cannot be empty")


class MemoryStore(Protocol):
    async def add(self, record: MemoryRecord) -> str: ...
    async def search(self, query: str, limit: int = 5) -> list[MemoryRecord]: ...
    async def clear(self) -> None: ...
