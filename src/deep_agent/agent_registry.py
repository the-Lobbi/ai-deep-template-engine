"""Agent registry for dynamic subagent discovery and instantiation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class TaskRequirements:
    """Describe what a task needs from subagents."""

    task: str
    capabilities: Sequence[str] = field(default_factory=list)
    allow_team: bool = True
    priority: str = "normal"


@dataclass(frozen=True)
class SubagentSpec:
    """Describe a subagent and how to instantiate it."""

    name: str
    description: str
    capabilities: Sequence[str]
    supported_tasks: Sequence[str]
    factory: Callable[[Dict[str, Any]], Any]

    def matches(self, requirements: TaskRequirements) -> bool:
        """Return True if this subagent can satisfy the requirements."""
        if requirements.task in self.supported_tasks:
            return True
        required = set(requirements.capabilities)
        if not required:
            return True
        return required.issubset(set(self.capabilities))

    def match_score(self, requirements: TaskRequirements) -> int:
        """Score how well this subagent matches the requirements."""
        score = 0
        if requirements.task in self.supported_tasks:
            score += 2
        score += len(set(requirements.capabilities).intersection(self.capabilities))
        return score


@dataclass(frozen=True)
class SubagentInvocation:
    """Plan for invoking a subagent with preserved context."""

    name: str
    reason: str
    context: Dict[str, Any]


class AgentRegistry:
    """Registry for subagents with discovery and instantiation helpers."""

    def __init__(self) -> None:
        self._registry: Dict[str, SubagentSpec] = {}

    def register(self, spec: SubagentSpec) -> None:
        """Register a subagent specification."""
        self._registry[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Remove a subagent from the registry."""
        self._registry.pop(name, None)

    def get(self, name: str) -> Optional[SubagentSpec]:
        """Fetch a subagent specification by name."""
        return self._registry.get(name)

    def list_names(self) -> List[str]:
        """List registered subagent names."""
        return sorted(self._registry.keys())

    def discover(self, requirements: TaskRequirements) -> List[SubagentSpec]:
        """Find subagents that match the requirements."""
        return [spec for spec in self._registry.values() if spec.matches(requirements)]

    def select_for_task(self, requirements: TaskRequirements) -> List[SubagentSpec]:
        """Select a subagent or team based on the requirements."""
        matches = self.discover(requirements)
        if not matches:
            return []
        scored = [(spec, spec.match_score(requirements)) for spec in matches]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        if requirements.allow_team:
            return [spec for spec, score in ranked if score > 0]
        return [ranked[0][0]]

    def instantiate(self, name: str, context: Dict[str, Any]) -> Any:
        """Instantiate a subagent by name."""
        spec = self._registry.get(name)
        if spec is None:
            raise KeyError(f"Subagent '{name}' is not registered")
        return spec.factory(context)

    def preserve_context(self, base_context: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Return a new context preserving existing data with a history trail."""
        context = {**base_context}
        history = list(context.get("context_history", []))
        # Avoid allowing callers to override or corrupt the history field directly.
        # We strip out any 'context_history' key from updates before recording it in
        # the history trail and applying it to the new context.
        filtered_updates = {k: v for k, v in updates.items() if k != "context_history"}
        history.append(filtered_updates)
        context.update(filtered_updates)
        context["context_history"] = history
        return context

    def plan_invocations(
        self, requirements: TaskRequirements, context: Dict[str, Any], reason: str
    ) -> List[SubagentInvocation]:
        """Create invocation plans for matching subagents."""
        selected = self.select_for_task(requirements)
        preserved = self.preserve_context(context, {"requirements": requirements})
        return [
            SubagentInvocation(name=spec.name, reason=reason, context=preserved)
            for spec in selected
        ]

    def plan_for_node(
        self, node_name: str, requirements: TaskRequirements, context: Dict[str, Any]
    ) -> List[SubagentInvocation]:
        """Hook for approaching a node in the workflow."""
        return self.plan_invocations(
            requirements,
            context,
            reason=f"Approaching node '{node_name}'",
        )

    def plan_for_edge(
        self, source: str, destination: str, requirements: TaskRequirements, context: Dict[str, Any]
    ) -> List[SubagentInvocation]:
        """Hook for approaching an edge in the workflow."""
        return self.plan_invocations(
            requirements,
            context,
            reason=f"Transitioning from '{source}' to '{destination}'",
        )


def default_subagent_factory(name: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create a simple factory for placeholder subagents."""

    def _factory(context: Dict[str, Any]) -> Dict[str, Any]:
        return {"subagent": name, "context": context}

    return _factory
