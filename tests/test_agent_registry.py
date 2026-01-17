"""Comprehensive tests for the agent registry module."""

import pytest
from typing import Any, Dict

from deep_agent.agent_registry import (
    AgentRegistry,
    SubagentSpec,
    TaskRequirements,
    SubagentInvocation,
    default_subagent_factory,
)


# Test fixtures and helpers
@pytest.fixture
def empty_registry():
    """Create an empty agent registry."""
    return AgentRegistry()


@pytest.fixture
def sample_factory():
    """Create a simple factory function for testing."""
    def _factory(context: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "created", "context": context}
    return _factory


@pytest.fixture
def terraform_spec(sample_factory):
    """Create a Terraform subagent spec."""
    return SubagentSpec(
        name="terraform-agent",
        description="Terraform infrastructure agent",
        capabilities=["terraform", "infrastructure", "aws"],
        supported_tasks=["terraform_plan", "terraform_apply"],
        factory=sample_factory,
    )


@pytest.fixture
def docker_spec(sample_factory):
    """Create a Docker subagent spec."""
    return SubagentSpec(
        name="docker-agent",
        description="Docker containerization agent",
        capabilities=["docker", "containerization", "kubernetes"],
        supported_tasks=["containerize", "build_image"],
        factory=sample_factory,
    )


@pytest.fixture
def generalist_spec(sample_factory):
    """Create a generalist subagent spec."""
    return SubagentSpec(
        name="generalist-agent",
        description="General purpose agent",
        capabilities=["general", "scripting", "automation"],
        supported_tasks=["general_task"],
        factory=sample_factory,
    )


@pytest.fixture
def populated_registry(terraform_spec, docker_spec, generalist_spec):
    """Create a registry with multiple agents."""
    registry = AgentRegistry()
    registry.register(terraform_spec)
    registry.register(docker_spec)
    registry.register(generalist_spec)
    return registry


# Tests for TaskRequirements
class TestTaskRequirements:
    """Tests for TaskRequirements dataclass."""

    def test_task_requirements_creation(self):
        """Test creating TaskRequirements with minimal parameters."""
        req = TaskRequirements(task="test_task")
        assert req.task == "test_task"
        assert req.capabilities == []
        assert req.allow_team is True
        assert req.priority == "normal"

    def test_task_requirements_with_capabilities(self):
        """Test creating TaskRequirements with capabilities."""
        req = TaskRequirements(
            task="terraform_plan",
            capabilities=["terraform", "aws"],
            allow_team=False,
            priority="high",
        )
        assert req.task == "terraform_plan"
        assert req.capabilities == ["terraform", "aws"]
        assert req.allow_team is False
        assert req.priority == "high"

    def test_task_requirements_immutable(self):
        """Test that TaskRequirements is immutable."""
        req = TaskRequirements(task="test")
        with pytest.raises(AttributeError):
            req.task = "new_task"


# Tests for SubagentSpec
class TestSubagentSpec:
    """Tests for SubagentSpec functionality."""

    def test_subagent_spec_creation(self, terraform_spec):
        """Test creating a SubagentSpec."""
        assert terraform_spec.name == "terraform-agent"
        assert terraform_spec.description == "Terraform infrastructure agent"
        assert "terraform" in terraform_spec.capabilities
        assert "terraform_plan" in terraform_spec.supported_tasks

    def test_subagent_spec_immutable(self, terraform_spec):
        """Test that SubagentSpec is immutable."""
        with pytest.raises(AttributeError):
            terraform_spec.name = "new-name"

    def test_matches_by_task(self, terraform_spec):
        """Test matching by supported task."""
        req = TaskRequirements(task="terraform_plan")
        assert terraform_spec.matches(req) is True

    def test_matches_by_task_not_supported(self, terraform_spec):
        """Test not matching when task is not supported."""
        req = TaskRequirements(task="containerize", capabilities=["docker"])
        assert terraform_spec.matches(req) is False

    def test_matches_by_capabilities(self, terraform_spec):
        """Test matching by capabilities."""
        req = TaskRequirements(task="unknown_task", capabilities=["terraform", "aws"])
        assert terraform_spec.matches(req) is True

    def test_matches_by_partial_capabilities(self, terraform_spec):
        """Test matching with subset of capabilities."""
        req = TaskRequirements(task="unknown_task", capabilities=["terraform"])
        assert terraform_spec.matches(req) is True

    def test_no_match_capabilities_not_subset(self, terraform_spec):
        """Test not matching when required capabilities are not a subset."""
        req = TaskRequirements(task="unknown_task", capabilities=["docker", "kubernetes"])
        assert terraform_spec.matches(req) is False

    def test_matches_with_empty_capabilities(self, terraform_spec):
        """Test matching with no capability requirements."""
        req = TaskRequirements(task="unknown_task", capabilities=[])
        assert terraform_spec.matches(req) is True

    def test_match_score_by_task(self, terraform_spec):
        """Test scoring for matching task."""
        req = TaskRequirements(task="terraform_plan")
        score = terraform_spec.match_score(req)
        assert score == 2  # Task match gives +2

    def test_match_score_by_capabilities(self, terraform_spec):
        """Test scoring for capability matches."""
        req = TaskRequirements(task="unknown_task", capabilities=["terraform", "aws"])
        score = terraform_spec.match_score(req)
        assert score == 2  # Two matching capabilities

    def test_match_score_combined(self, terraform_spec):
        """Test scoring with both task and capability matches."""
        req = TaskRequirements(task="terraform_plan", capabilities=["terraform", "aws"])
        score = terraform_spec.match_score(req)
        assert score == 4  # Task match (2) + two capabilities (2)

    def test_match_score_zero(self, terraform_spec):
        """Test scoring with no matches."""
        req = TaskRequirements(task="unknown_task", capabilities=["docker"])
        score = terraform_spec.match_score(req)
        assert score == 0

    def test_match_score_partial_capabilities(self, terraform_spec):
        """Test scoring with only some capability matches."""
        req = TaskRequirements(
            task="unknown_task", 
            capabilities=["terraform", "docker", "aws"]
        )
        score = terraform_spec.match_score(req)
        assert score == 2  # Only terraform and aws match


# Tests for AgentRegistry basic operations
class TestAgentRegistryBasics:
    """Tests for basic AgentRegistry operations."""

    def test_empty_registry_creation(self, empty_registry):
        """Test creating an empty registry."""
        assert empty_registry.list_names() == []

    def test_register_agent(self, empty_registry, terraform_spec):
        """Test registering a single agent."""
        empty_registry.register(terraform_spec)
        assert "terraform-agent" in empty_registry.list_names()
        assert empty_registry.get("terraform-agent") == terraform_spec

    def test_register_multiple_agents(self, empty_registry, terraform_spec, docker_spec):
        """Test registering multiple agents."""
        empty_registry.register(terraform_spec)
        empty_registry.register(docker_spec)
        names = empty_registry.list_names()
        assert len(names) == 2
        assert "terraform-agent" in names
        assert "docker-agent" in names

    def test_register_duplicate_overwrites(self, empty_registry, terraform_spec, sample_factory):
        """Test that registering duplicate name overwrites previous."""
        empty_registry.register(terraform_spec)
        
        new_spec = SubagentSpec(
            name="terraform-agent",
            description="Updated description",
            capabilities=["terraform"],
            supported_tasks=["new_task"],
            factory=sample_factory,
        )
        empty_registry.register(new_spec)
        
        retrieved = empty_registry.get("terraform-agent")
        assert retrieved.description == "Updated description"
        assert len(empty_registry.list_names()) == 1

    def test_unregister_agent(self, populated_registry):
        """Test unregistering an agent."""
        assert "terraform-agent" in populated_registry.list_names()
        populated_registry.unregister("terraform-agent")
        assert "terraform-agent" not in populated_registry.list_names()
        assert populated_registry.get("terraform-agent") is None

    def test_unregister_nonexistent_agent(self, populated_registry):
        """Test unregistering a non-existent agent doesn't raise error."""
        initial_count = len(populated_registry.list_names())
        populated_registry.unregister("nonexistent-agent")
        assert len(populated_registry.list_names()) == initial_count

    def test_get_nonexistent_agent(self, empty_registry):
        """Test getting a non-existent agent returns None."""
        assert empty_registry.get("nonexistent") is None

    def test_list_names_sorted(self, populated_registry):
        """Test that list_names returns sorted names."""
        names = populated_registry.list_names()
        assert names == sorted(names)


# Tests for AgentRegistry discovery and selection
class TestAgentRegistryDiscovery:
    """Tests for agent discovery and selection."""

    def test_discover_empty_registry(self, empty_registry):
        """Test discovery in empty registry returns empty list."""
        req = TaskRequirements(task="any_task")
        matches = empty_registry.discover(req)
        assert matches == []

    def test_discover_by_task(self, populated_registry):
        """Test discovering agents by task.
        
        Note: When requirements have no capabilities specified, all agents match
        due to the empty capabilities check returning True. We verify that the
        terraform-agent is in the matches since it explicitly supports the task.
        """
        req = TaskRequirements(task="terraform_plan")
        matches = populated_registry.discover(req)
        # All agents match because capabilities is empty, but terraform-agent should be included
        assert len(matches) >= 1
        names = [m.name for m in matches]
        assert "terraform-agent" in names

    def test_discover_by_task_only(self, populated_registry):
        """Test discovering agents by specific task match only.
        
        By including a capability requirement that only one agent has,
        we can test that task matching works correctly.
        """
        req = TaskRequirements(task="terraform_plan", capabilities=["terraform"])
        matches = populated_registry.discover(req)
        assert len(matches) == 1
        assert matches[0].name == "terraform-agent"

    def test_discover_by_capabilities(self, populated_registry):
        """Test discovering agents by capabilities."""
        req = TaskRequirements(task="unknown_task", capabilities=["docker"])
        matches = populated_registry.discover(req)
        assert len(matches) == 1
        assert matches[0].name == "docker-agent"

    def test_discover_multiple_matches(self, populated_registry, sample_factory):
        """Test discovering when multiple agents match."""
        # Add another agent with terraform capability
        spec = SubagentSpec(
            name="terraform-specialist",
            description="Another terraform agent",
            capabilities=["terraform", "gcp"],
            supported_tasks=["terraform_validate"],
            factory=sample_factory,
        )
        populated_registry.register(spec)
        
        req = TaskRequirements(task="unknown_task", capabilities=["terraform"])
        matches = populated_registry.discover(req)
        assert len(matches) == 2
        names = [m.name for m in matches]
        assert "terraform-agent" in names
        assert "terraform-specialist" in names

    def test_discover_no_matches(self, populated_registry):
        """Test discovering when no agents match."""
        req = TaskRequirements(task="unknown_task", capabilities=["nonexistent"])
        matches = populated_registry.discover(req)
        assert matches == []

    def test_select_for_task_single_match(self, populated_registry):
        """Test selecting agent when only one matches."""
        req = TaskRequirements(task="terraform_plan", allow_team=False)
        selected = populated_registry.select_for_task(req)
        assert len(selected) == 1
        assert selected[0].name == "terraform-agent"

    def test_select_for_task_no_matches(self, populated_registry):
        """Test selecting when no agents match."""
        req = TaskRequirements(task="unknown_task", capabilities=["nonexistent"])
        selected = populated_registry.select_for_task(req)
        assert selected == []

    def test_select_for_task_best_match(self, populated_registry, sample_factory):
        """Test that highest scoring agent is selected first."""
        # Add agent with exact task match
        spec = SubagentSpec(
            name="exact-match",
            description="Exact match agent",
            capabilities=["terraform"],
            supported_tasks=["terraform_plan"],
            factory=sample_factory,
        )
        populated_registry.register(spec)
        
        req = TaskRequirements(task="terraform_plan", allow_team=False)
        selected = populated_registry.select_for_task(req)
        assert len(selected) == 1
        # Both have task match, but we want to verify the first one is selected
        assert selected[0].name in ["terraform-agent", "exact-match"]

    def test_select_for_task_team_mode(self, populated_registry, sample_factory):
        """Test selecting multiple agents in team mode."""
        # Add another terraform-capable agent
        spec = SubagentSpec(
            name="terraform-helper",
            description="Helper agent",
            capabilities=["terraform", "infrastructure"],
            supported_tasks=["terraform_validate"],
            factory=sample_factory,
        )
        populated_registry.register(spec)
        
        req = TaskRequirements(
            task="unknown_task", 
            capabilities=["terraform"],
            allow_team=True
        )
        selected = populated_registry.select_for_task(req)
        assert len(selected) >= 2
        names = [s.name for s in selected]
        assert "terraform-agent" in names
        assert "terraform-helper" in names

    def test_select_for_task_single_mode(self, populated_registry, sample_factory):
        """Test selecting only one agent when allow_team is False."""
        # Add multiple matching agents
        spec = SubagentSpec(
            name="terraform-helper",
            description="Helper agent",
            capabilities=["terraform"],
            supported_tasks=["terraform_validate"],
            factory=sample_factory,
        )
        populated_registry.register(spec)
        
        req = TaskRequirements(
            task="unknown_task",
            capabilities=["terraform"],
            allow_team=False
        )
        selected = populated_registry.select_for_task(req)
        assert len(selected) == 1

    def test_select_for_task_ranking(self, populated_registry, sample_factory):
        """Test that agents are ranked by score."""
        # Add agents with different match scores
        high_score_spec = SubagentSpec(
            name="high-score",
            description="High score agent",
            capabilities=["terraform", "aws", "infrastructure"],
            supported_tasks=["terraform_plan"],
            factory=sample_factory,
        )
        low_score_spec = SubagentSpec(
            name="low-score",
            description="Low score agent",
            capabilities=["terraform"],
            supported_tasks=[],
            factory=sample_factory,
        )
        populated_registry.register(high_score_spec)
        populated_registry.register(low_score_spec)
        
        req = TaskRequirements(
            task="terraform_plan",
            capabilities=["terraform", "aws"],
            allow_team=True
        )
        selected = populated_registry.select_for_task(req)
        
        # High score agent should be first
        assert len(selected) >= 2
        # Verify the first one has higher or equal score
        scores = [s.match_score(req) for s in selected]
        assert scores == sorted(scores, reverse=True)


# Tests for AgentRegistry instantiation
class TestAgentRegistryInstantiation:
    """Tests for agent instantiation."""

    def test_instantiate_agent(self, populated_registry):
        """Test instantiating a registered agent."""
        context = {"key": "value"}
        instance = populated_registry.instantiate("terraform-agent", context)
        assert instance["status"] == "created"
        assert instance["context"] == context

    def test_instantiate_with_context(self, populated_registry):
        """Test that context is passed to factory."""
        context = {"user": "test", "project": "demo"}
        instance = populated_registry.instantiate("docker-agent", context)
        assert instance["context"]["user"] == "test"
        assert instance["context"]["project"] == "demo"

    def test_instantiate_nonexistent_agent(self, populated_registry):
        """Test instantiating non-existent agent raises KeyError."""
        with pytest.raises(KeyError, match="is not registered"):
            populated_registry.instantiate("nonexistent-agent", {})

    def test_instantiate_with_default_factory(self, empty_registry):
        """Test instantiation with default factory."""
        factory = default_subagent_factory("test-agent")
        spec = SubagentSpec(
            name="test-agent",
            description="Test agent",
            capabilities=["test"],
            supported_tasks=["test_task"],
            factory=factory,
        )
        empty_registry.register(spec)
        
        context = {"test_key": "test_value"}
        instance = empty_registry.instantiate("test-agent", context)
        assert instance["subagent"] == "test-agent"
        assert instance["context"] == context


# Tests for context preservation
class TestContextPreservation:
    """Tests for context preservation functionality."""

    def test_preserve_context_basic(self, empty_registry):
        """Test basic context preservation."""
        base = {"key1": "value1"}
        updates = {"key2": "value2"}
        
        result = empty_registry.preserve_context(base, updates)
        
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert "context_history" in result

    def test_preserve_context_history_tracking(self, empty_registry):
        """Test that context history is tracked."""
        base = {"initial": "data"}
        updates = {"new": "info"}
        
        result = empty_registry.preserve_context(base, updates)
        
        assert "context_history" in result
        assert len(result["context_history"]) == 1
        assert result["context_history"][0] == updates

    def test_preserve_context_multiple_updates(self, empty_registry):
        """Test multiple context preservation calls."""
        context = {"start": "value"}
        
        context = empty_registry.preserve_context(context, {"update1": "data1"})
        context = empty_registry.preserve_context(context, {"update2": "data2"})
        context = empty_registry.preserve_context(context, {"update3": "data3"})
        
        assert len(context["context_history"]) == 3
        assert context["update1"] == "data1"
        assert context["update2"] == "data2"
        assert context["update3"] == "data3"

    def test_preserve_context_overwrites(self, empty_registry):
        """Test that updates overwrite existing keys."""
        base = {"key": "old_value"}
        updates = {"key": "new_value"}
        
        result = empty_registry.preserve_context(base, updates)
        
        assert result["key"] == "new_value"

    def test_preserve_context_immutability(self, empty_registry):
        """Test that original context is not modified."""
        base = {"key1": "value1"}
        updates = {"key2": "value2"}
        
        result = empty_registry.preserve_context(base, updates)
        
        # Original should not be modified
        assert "key2" not in base
        assert "context_history" not in base

    def test_preserve_context_existing_history(self, empty_registry):
        """Test preserving context when history already exists."""
        base = {
            "data": "value",
            "context_history": [{"old": "update"}]
        }
        updates = {"new": "update"}
        
        result = empty_registry.preserve_context(base, updates)
        
        assert len(result["context_history"]) == 2
        assert result["context_history"][0] == {"old": "update"}
        assert result["context_history"][1] == {"new": "update"}


# Tests for planning methods
class TestPlanningMethods:
    """Tests for invocation planning methods."""

    def test_plan_invocations_single_match(self, populated_registry):
        """Test planning invocations with single matching agent."""
        req = TaskRequirements(task="terraform_plan", allow_team=False)
        context = {"project": "test"}
        reason = "Testing invocation"
        
        plans = populated_registry.plan_invocations(req, context, reason)
        
        assert len(plans) == 1
        assert isinstance(plans[0], SubagentInvocation)
        assert plans[0].name == "terraform-agent"
        assert plans[0].reason == reason
        assert "project" in plans[0].context

    def test_plan_invocations_team(self, populated_registry, sample_factory):
        """Test planning invocations with team mode."""
        # Add multiple matching agents
        spec = SubagentSpec(
            name="terraform-helper",
            description="Helper",
            capabilities=["terraform"],
            supported_tasks=["terraform_validate"],
            factory=sample_factory,
        )
        populated_registry.register(spec)
        
        req = TaskRequirements(
            task="unknown_task",
            capabilities=["terraform"],
            allow_team=True
        )
        context = {"data": "test"}
        
        plans = populated_registry.plan_invocations(req, context, "test")
        
        assert len(plans) >= 2

    def test_plan_invocations_context_preserved(self, populated_registry):
        """Test that context is preserved in invocation plans."""
        req = TaskRequirements(task="terraform_plan")
        context = {"original": "data"}
        
        plans = populated_registry.plan_invocations(req, context, "test")
        
        assert plans[0].context["original"] == "data"
        assert "requirements" in plans[0].context
        assert "context_history" in plans[0].context

    def test_plan_invocations_no_matches(self, populated_registry):
        """Test planning when no agents match."""
        req = TaskRequirements(task="nonexistent_task", capabilities=["fake"])
        context = {}
        
        plans = populated_registry.plan_invocations(req, context, "test")
        
        assert plans == []

    def test_plan_for_node(self, populated_registry):
        """Test planning for a workflow node."""
        req = TaskRequirements(task="terraform_plan")
        context = {"node_data": "value"}
        
        plans = populated_registry.plan_for_node("analyze", req, context)
        
        assert len(plans) >= 1
        assert plans[0].reason == "Approaching node 'analyze'"
        assert "node_data" in plans[0].context

    def test_plan_for_edge(self, populated_registry):
        """Test planning for a workflow edge."""
        req = TaskRequirements(task="containerize")
        context = {"edge_data": "value"}
        
        plans = populated_registry.plan_for_edge("source", "destination", req, context)
        
        assert len(plans) >= 1
        assert plans[0].reason == "Transitioning from 'source' to 'destination'"
        assert "edge_data" in plans[0].context


# Tests for default factory
class TestDefaultFactory:
    """Tests for default_subagent_factory."""

    def test_default_factory_creation(self):
        """Test creating a factory with default_subagent_factory."""
        factory = default_subagent_factory("test-agent")
        assert callable(factory)

    def test_default_factory_returns_dict(self):
        """Test that default factory returns correct structure."""
        factory = default_subagent_factory("test-agent")
        context = {"key": "value"}
        
        result = factory(context)
        
        assert isinstance(result, dict)
        assert result["subagent"] == "test-agent"
        assert result["context"] == context

    def test_default_factory_different_names(self):
        """Test that different factories have different names."""
        factory1 = default_subagent_factory("agent1")
        factory2 = default_subagent_factory("agent2")
        
        result1 = factory1({})
        result2 = factory2({})
        
        assert result1["subagent"] == "agent1"
        assert result2["subagent"] == "agent2"

    def test_default_factory_preserves_context(self):
        """Test that default factory preserves all context."""
        factory = default_subagent_factory("agent")
        context = {
            "key1": "value1",
            "key2": "value2",
            "nested": {"data": "here"}
        }
        
        result = factory(context)
        
        assert result["context"] == context
        assert result["context"]["nested"]["data"] == "here"


# Edge case tests
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_task_string(self, terraform_spec):
        """Test matching with empty task string."""
        req = TaskRequirements(task="", capabilities=["terraform"])
        assert terraform_spec.matches(req) is True

    def test_empty_capabilities_list(self, terraform_spec):
        """Test matching with empty capabilities."""
        req = TaskRequirements(task="unknown", capabilities=[])
        assert terraform_spec.matches(req) is True

    def test_large_capability_list(self, empty_registry, sample_factory):
        """Test agent with large capability list."""
        caps = [f"capability_{i}" for i in range(100)]
        spec = SubagentSpec(
            name="large-agent",
            description="Agent with many capabilities",
            capabilities=caps,
            supported_tasks=["test"],
            factory=sample_factory,
        )
        empty_registry.register(spec)
        
        req = TaskRequirements(task="test")
        matches = empty_registry.discover(req)
        assert len(matches) == 1

    def test_unicode_in_names(self, empty_registry, sample_factory):
        """Test handling unicode characters in names."""
        spec = SubagentSpec(
            name="agent-αβγ",
            description="Unicode test agent",
            capabilities=["test"],
            supported_tasks=["test"],
            factory=sample_factory,
        )
        empty_registry.register(spec)
        
        assert "agent-αβγ" in empty_registry.list_names()
        assert empty_registry.get("agent-αβγ") is not None

    def test_special_characters_in_context(self, empty_registry):
        """Test context preservation with special characters."""
        base = {"key": "value with\nnewlines\tand\ttabs"}
        updates = {"special": "chars: @#$%^&*()"}
        
        result = empty_registry.preserve_context(base, updates)
        
        assert result["key"] == "value with\nnewlines\tand\ttabs"
        assert result["special"] == "chars: @#$%^&*()"

    def test_nested_context_structures(self, empty_registry):
        """Test context preservation with nested structures."""
        base = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }
        updates = {"new_nested": {"data": "here"}}
        
        result = empty_registry.preserve_context(base, updates)
        
        assert result["level1"]["level2"]["level3"] == "deep_value"
        assert result["new_nested"]["data"] == "here"

    def test_zero_score_filtering(self, populated_registry):
        """Test that zero-score agents are filtered in team mode."""
        req = TaskRequirements(
            task="nonexistent",
            capabilities=["terraform"],
            allow_team=True
        )
        
        selected = populated_registry.select_for_task(req)
        
        # All selected agents should have score > 0
        for spec in selected:
            assert spec.match_score(req) > 0

    def test_multiple_task_matches(self, empty_registry, sample_factory):
        """Test agent with multiple supported tasks."""
        spec = SubagentSpec(
            name="multi-task",
            description="Multi-task agent",
            capabilities=["task1", "task2"],
            supported_tasks=["task_a", "task_b", "task_c"],
            factory=sample_factory,
        )
        empty_registry.register(spec)
        
        for task in ["task_a", "task_b", "task_c"]:
            req = TaskRequirements(task=task)
            assert spec.matches(req) is True
            assert spec.match_score(req) == 2
