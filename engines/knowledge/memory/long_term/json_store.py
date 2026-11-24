"""Portable JSON-backed long-term memory store."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..interface import MemoryRecord
from .interface import LongTermMemory


class JsonLongTermMemory(LongTermMemory):
    def __init__(self, storage_path: str = "data/memory/long_term.json"):
        self.storage_path = Path(storage_path)
        self._lock = asyncio.Lock()

    def _load(self) -> dict[str, dict]:
        if not self.storage_path.is_file():
            return {}
        with self.storage_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Long-term memory JSON must contain an object")
        return data

    def _save(self, records: dict[str, dict]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("w", encoding="utf-8") as file:
            json.dump(records, file, ensure_ascii=False, indent=2)

    async def add(self, record: MemoryRecord) -> str:
        record_id = str(uuid4())
        async with self._lock:
            records = self._load()
            records[record_id] = {
                "content": record.content,
                "metadata": record.metadata,
                "timestamp": record.timestamp.isoformat(),
            }
            self._save(records)
        return record_id

    async def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        terms = [term.lower() for term in query.lower().split() if term]
        async with self._lock:
            records = self._load()

        ranked = []
        for item in records.values():
            content = str(item.get("content", ""))
            score = sum(term in content.lower() for term in terms)
            if terms and score == 0:
                continue
            timestamp = item.get("timestamp")
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc)
            except ValueError:
                parsed_timestamp = datetime.now(timezone.utc)
            ranked.append((score, parsed_timestamp, MemoryRecord(
                content=content,
                metadata=dict(item.get("metadata", {})),
                timestamp=parsed_timestamp,
            )))

        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [entry[2] for entry in ranked[:limit]]

    async def delete(self, record_id: str) -> bool:
        async with self._lock:
            records = self._load()
            deleted = records.pop(record_id, None) is not None
            if deleted:
                self._save(records)
        return deleted

    async def clear(self) -> None:
        async with self._lock:
            self._save({})
