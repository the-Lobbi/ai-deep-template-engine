"""Tests for Deep Agent Harness Automation System."""

import pytest
from httpx import AsyncClient
from pytest_httpx import HTTPXMock

from deep_agent import HarnessDeepAgent, AgentConfig


@pytest.fixture
def agent_config():
    """Create test agent configuration."""
    return AgentConfig(
        harness_account_id="test_account",
        harness_api_url="https://test.harness.io/gateway",
        harness_api_token="test_token",
        org_identifier="test_org",
        project_identifier="test_project"
    )


@pytest.fixture
async def agent(agent_config):
    """Create test agent instance."""
    agent = HarnessDeepAgent(agent_config)
    yield agent
    await agent.client.aclose()


@pytest.mark.asyncio
async def test_agent_initialization(agent_config):
    """Test agent initialization with config."""
    agent = HarnessDeepAgent(agent_config)

    assert agent.config.harness_account_id == "test_account"
    assert agent.config.org_identifier == "test_org"
    assert len(agent.registry.list_names()) == 3

    await agent.client.aclose()


@pytest.mark.asyncio
async def test_health_check_success(agent, httpx_mock: HTTPXMock):
    """Test successful health check."""
    httpx_mock.add_response(
        url="https://test.harness.io/gateway/ng/api/user/currentUser",
        json={"data": {"uuid": "test_user"}}
    )

    result = await agent.health_check()

    assert result["status"] == "healthy"
    assert result["harness_api"] == "connected"


@pytest.mark.asyncio
async def test_health_check_failure(agent, httpx_mock: HTTPXMock):
    """Test health check failure."""
    httpx_mock.add_response(
        url="https://test.harness.io/gateway/ng/api/user/currentUser",
        status_code=401
    )

    result = await agent.health_check()

    assert result["status"] == "unhealthy"
    assert "error" in result


@pytest.mark.asyncio
async def test_create_repository(agent, httpx_mock: HTTPXMock):
    """Test repository creation."""
    httpx_mock.add_response(
        url="https://test.harness.io/gateway/code/api/v1/repos/test_org/test_project",
        method="POST",
        json={
            "path": "test_org/test_project/test_repo",
            "identifier": "test_repo"
        }
    )

    result = await agent.create_repository(
        repo_name="test_repo",
        project_identifier="test_project",
        description="Test repository"
    )

    assert result["path"] == "test_org/test_project/test_repo"
    assert result["identifier"] == "test_repo"


@pytest.mark.asyncio
async def test_create_pipeline(agent, httpx_mock: HTTPXMock):
    """Test pipeline creation."""
    httpx_mock.add_response(
        url="https://test.harness.io/gateway/pipeline/api/pipelines/v2",
        method="POST",
        json={
            "identifier": "test_pipeline",
            "status": "SUCCESS"
        }
    )

    result = await agent.create_pipeline(
        pipeline_name="test_pipeline",
        project_identifier="test_project",
        pipeline_yaml="pipeline:\n  name: test"
    )

    assert result["identifier"] == "test_pipeline"
    assert result["status"] == "SUCCESS"


@pytest.mark.asyncio
async def test_get_repositories(agent, httpx_mock: HTTPXMock):
    """Test repository listing."""
    httpx_mock.add_response(
        url="https://test.harness.io/gateway/code/api/v1/repos/test_org/test_project?page=1&limit=50",
        json=[
            {"identifier": "repo1", "path": "test_org/test_project/repo1"},
            {"identifier": "repo2", "path": "test_org/test_project/repo2"}
        ]
    )

    result = await agent.get_repositories(project_identifier="test_project")

    assert len(result) == 2
    assert result[0]["identifier"] == "repo1"
    assert result[1]["identifier"] == "repo2"


@pytest.mark.asyncio
async def test_delegate_to_subagent(agent):
    """Test subagent delegation."""
    result = await agent.delegate_to_subagent(
        subagent="iac-golden-architect",
        task="terraform_plan",
        context={"working_dir": "/test"}
    )

    assert result["subagent"] == "iac-golden-architect"
    assert result["task"] == "terraform_plan"
    assert result["status"] == "delegated"
    assert result["context"]["working_dir"] == "/test"
    assert result["instance"]["subagent"] == "iac-golden-architect"


@pytest.mark.asyncio
async def test_delegate_to_invalid_subagent(agent):
    """Test delegation to invalid subagent."""
    with pytest.raises(ValueError, match="is not enabled"):
        await agent.delegate_to_subagent(
            subagent="invalid-agent",
            task="test",
            context={}
        )


@pytest.mark.asyncio
async def test_context_manager(agent_config, httpx_mock: HTTPXMock):
    """Test agent as async context manager."""
    httpx_mock.add_response(
        url="https://test.harness.io/gateway/ng/api/user/currentUser",
        json={"data": {"uuid": "test_user"}}
    )

    async with HarnessDeepAgent(agent_config) as agent:
        result = await agent.health_check()
        assert result["status"] == "healthy"

    # Client should be closed after context exit
    assert agent.client.is_closed


@pytest.mark.asyncio
async def test_agent_with_custom_subagents(agent_config):
    """Test agent with custom subagent list."""
    agent_config.enabled_subagents = ["iac-golden-architect"]
    agent = HarnessDeepAgent(agent_config)

    # Should work with enabled subagent
    result = await agent.delegate_to_subagent(
        subagent="iac-golden-architect",
        task="test",
        context={}
    )
    assert result["status"] == "delegated"

    # Should fail with disabled subagent
    with pytest.raises(ValueError):
        await agent.delegate_to_subagent(
            subagent="container-workflow",
            task="test",
            context={}
        )

    await agent.client.aclose()


@pytest.mark.asyncio
async def test_agent_with_empty_subagents(agent_config):
    """Test agent with explicitly empty subagent list disables all subagents."""
    agent_config.enabled_subagents = []
    agent = HarnessDeepAgent(agent_config)

    # No subagents should be registered
    assert len(agent.registry.list_names()) == 0

    # Should fail to delegate to any subagent
    with pytest.raises(ValueError, match="is not enabled"):
        await agent.delegate_to_subagent(
            subagent="iac-golden-architect",
            task="test",
            context={}
        )

    await agent.client.aclose()


@pytest.mark.asyncio
async def test_agent_with_none_subagents(agent_config):
    """Test agent with None uses default subagents."""
    agent_config.enabled_subagents = None
    agent = HarnessDeepAgent(agent_config)

    # Default subagents should be registered
    assert len(agent.registry.list_names()) == 3
    assert "iac-golden-architect" in agent.registry.list_names()
    assert "container-workflow" in agent.registry.list_names()
    assert "team-accelerator" in agent.registry.list_names()

    await agent.client.aclose()


def test_plan_hooks(agent_config):
    """Ensure planning hooks return invocations for nodes and edges."""
    agent = HarnessDeepAgent(agent_config)
    node_plan = agent.plan_subagents_for_node(
        node_name="analyze",
        task="terraform_plan",
        context={"request_id": "123"},
        capabilities=["terraform"],
    )
    edge_plan = agent.plan_subagents_for_edge(
        source="analyze",
        destination="iac_architect",
        task="containerize",
        context={"request_id": "456"},
        capabilities=["docker"],
    )

    assert node_plan
    assert edge_plan
    assert node_plan[0].context["request_id"] == "123"
    assert edge_plan[0].context["request_id"] == "456"
