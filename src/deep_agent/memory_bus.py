"""Shared memory bus with namespace-aware access control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol

MemoryNamespace = Literal["workflow", "agent", "org"]


class MemoryBackend(Protocol):
    """Backend interface for the shared memory bus."""

    def get(self, namespace: MemoryNamespace, key: str) -> Any | None:
        """Return a stored value or None if missing."""

    def set(self, namespace: MemoryNamespace, key: str, value: Any) -> None:
        """Persist a value in the backend."""

    def delete(self, namespace: MemoryNamespace, key: str) -> None:
        """Remove a key from the backend."""

    def list_keys(self, namespace: MemoryNamespace, prefix: Optional[str] = None) -> List[str]:
        """List keys in a namespace, optionally filtered by prefix."""

    def clear(self, namespace: MemoryNamespace) -> None:
        """Remove all keys for a namespace."""


class InMemoryMemoryBackend:
    """In-memory implementation of the memory backend."""

    def __init__(self) -> None:
        self._store: Dict[MemoryNamespace, Dict[str, Any]] = {
            "workflow": {},
            "agent": {},
            "org": {},
        }

    def get(self, namespace: MemoryNamespace, key: str) -> Any | None:
        return self._store[namespace].get(key)

    def set(self, namespace: MemoryNamespace, key: str, value: Any) -> None:
        self._store[namespace][key] = value

    def delete(self, namespace: MemoryNamespace, key: str) -> None:
        self._store[namespace].pop(key, None)

    def list_keys(self, namespace: MemoryNamespace, prefix: Optional[str] = None) -> List[str]:
        if prefix is None:
            return sorted(self._store[namespace].keys())
        return sorted(key for key in self._store[namespace] if key.startswith(prefix))

    def clear(self, namespace: MemoryNamespace) -> None:
        self._store[namespace].clear()


@dataclass(frozen=True)
class AccessContext:
    """Access context for memory bus operations."""

    actor: str
    allowed_namespaces: frozenset[MemoryNamespace] = field(default_factory=frozenset)

    @classmethod
    def for_workflow(cls, actor: str) -> "AccessContext":
        return cls(actor=actor, allowed_namespaces=frozenset({"workflow"}))

    @classmethod
    def for_agent(cls, actor: str) -> "AccessContext":
        return cls(actor=actor, allowed_namespaces=frozenset({"agent"}))

    @classmethod
    def for_org(cls, actor: str) -> "AccessContext":
        return cls(actor=actor, allowed_namespaces=frozenset({"workflow", "agent", "org"}))

    def allows(self, namespace: MemoryNamespace) -> bool:
        return namespace in self.allowed_namespaces


class MemoryBus:
    """Namespace-aware shared memory bus with access checks."""

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        self._backend = backend or InMemoryMemoryBackend()

    def get(
        self,
        namespace: MemoryNamespace,
        key: str,
        *,
        access_context: AccessContext,
    ) -> Any | None:
        self._require_access(namespace, access_context)
        return self._backend.get(namespace, key)

    def set(
        self,
        namespace: MemoryNamespace,
        key: str,
        value: Any,
        *,
        access_context: AccessContext,
    ) -> None:
        self._require_access(namespace, access_context)
        self._backend.set(namespace, key, value)

    def delete(
        self,
        namespace: MemoryNamespace,
        key: str,
        *,
        access_context: AccessContext,
    ) -> None:
        self._require_access(namespace, access_context)
        self._backend.delete(namespace, key)

    def list_keys(
        self,
        namespace: MemoryNamespace,
        *,
        access_context: AccessContext,
        prefix: Optional[str] = None,
    ) -> List[str]:
        self._require_access(namespace, access_context)
        return self._backend.list_keys(namespace, prefix=prefix)

    def clear(self, namespace: MemoryNamespace, *, access_context: AccessContext) -> None:
        self._require_access(namespace, access_context)
        self._backend.clear(namespace)

    @staticmethod
    def _require_access(namespace: MemoryNamespace, access_context: AccessContext) -> None:
        if not access_context.allows(namespace):
            raise PermissionError(
                f"Access denied for namespace '{namespace}' by actor '{access_context.actor}'."
            )
