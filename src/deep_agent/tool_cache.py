"""Caching utilities for tool and subagent invocations."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional


@dataclass
class CacheEntry:
    """Cache entry with value and expiry timestamp."""

    value: Any
    expires_at: float


class ToolCache:
    """Simple in-memory cache with TTL and namespaced keys."""

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._store: Dict[str, Dict[str, CacheEntry]] = {}

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def build_namespace(self, tool_name: str, agent_name: str) -> str:
        return f"{tool_name}:{agent_name}"

    def build_cache_key(self, tool_inputs: Mapping[str, Any]) -> str:
        payload = json.dumps(tool_inputs, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        entry = self._store.get(namespace, {}).get(key)
        if not entry:
            return None
        if entry.expires_at <= self._clock():
            self._store[namespace].pop(key, None)
            if not self._store[namespace]:
                self._store.pop(namespace, None)
            return None
        return entry.value

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        ttl = self._ttl_seconds if ttl_seconds is None else ttl_seconds
        expires_at = self._clock() + ttl
        self._store.setdefault(namespace, {})[key] = CacheEntry(
            value=value, expires_at=expires_at
        )
