"""Shared memory bus for Deep Agent workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Literal, Optional, Protocol

MemoryNamespace = Literal["workflow", "agent", "org"]


class MemoryBackend(Protocol):
    """Backend interface for memory persistence."""

    def get(self, namespace: MemoryNamespace, key: str) -> Any:
        """Return the value stored for a namespace/key."""

    def set(self, namespace: MemoryNamespace, key: str, value: Any) -> None:
        """Store a value for a namespace/key."""

    def delete(self, namespace: MemoryNamespace, key: str) -> None:
        """Delete a value for a namespace/key."""

    def list(self, namespace: MemoryNamespace) -> Dict[str, Any]:
        """List all values in a namespace."""


class InMemoryMemoryBackend:
    """In-memory backend for memory persistence."""

    def __init__(self) -> None:
        self._store: Dict[MemoryNamespace, Dict[str, Any]] = {
            "workflow": {},
            "agent": {},
            "org": {},
        }

    def get(self, namespace: MemoryNamespace, key: str) -> Any:
        return self._store[namespace].get(key)

    def set(self, namespace: MemoryNamespace, key: str, value: Any) -> None:
        self._store[namespace][key] = value

    def delete(self, namespace: MemoryNamespace, key: str) -> None:
        self._store[namespace].pop(key, None)

    def list(self, namespace: MemoryNamespace) -> Dict[str, Any]:
        return dict(self._store[namespace])


class MemoryAccessError(PermissionError):
    """Raised when a caller lacks access to a namespace."""


@dataclass(frozen=True)
class AccessContext:
    """Access context for memory operations."""

    actor: str
    allowed_namespaces: FrozenSet[MemoryNamespace]

    @classmethod
    def for_workflow(cls, actor: str) -> "AccessContext":
        return cls(actor=actor, allowed_namespaces=frozenset({"workflow"}))

    @classmethod
    def for_agent(cls, actor: str) -> "AccessContext":
        return cls(actor=actor, allowed_namespaces=frozenset({"workflow", "agent"}))

    @classmethod
    def for_org(cls, actor: str) -> "AccessContext":
        return cls(actor=actor, allowed_namespaces=frozenset({"workflow", "agent", "org"}))

    def allows(self, namespace: MemoryNamespace) -> bool:
        return namespace in self.allowed_namespaces


class MemoryBus:
    """Shared memory bus with namespace isolation and access checks."""

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        self._backend = backend or InMemoryMemoryBackend()

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    def _check_access(self, namespace: MemoryNamespace, access_context: AccessContext) -> None:
        if not access_context.allows(namespace):
            raise MemoryAccessError(
                f"Actor '{access_context.actor}' cannot access '{namespace}' namespace."
            )

    def get(
        self, namespace: MemoryNamespace, key: str, access_context: AccessContext
    ) -> Any:
        self._check_access(namespace, access_context)
        return self._backend.get(namespace, key)

    def set(
        self,
        namespace: MemoryNamespace,
        key: str,
        value: Any,
        access_context: AccessContext,
    ) -> None:
        self._check_access(namespace, access_context)
        self._backend.set(namespace, key, value)

    def delete(self, namespace: MemoryNamespace, key: str, access_context: AccessContext) -> None:
        self._check_access(namespace, access_context)
        self._backend.delete(namespace, key)

    def list(
        self, namespace: MemoryNamespace, access_context: AccessContext
    ) -> Dict[str, Any]:
        self._check_access(namespace, access_context)
        return self._backend.list(namespace)

    def merge(
        self,
        namespace: MemoryNamespace,
        values: Dict[str, Any],
        access_context: AccessContext,
    ) -> None:
        self._check_access(namespace, access_context)
        for key, value in values.items():
            self._backend.set(namespace, key, value)

    def allowed_namespaces(self, access_context: AccessContext) -> Iterable[MemoryNamespace]:
        return access_context.allowed_namespaces
