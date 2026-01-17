"""Test script for DevOps multi-agent API server.

This module provides tests for the FastAPI server endpoints.

Usage:
    pytest test_server.py
    pytest test_server.py -v
    pytest test_server.py -k test_health_check
"""

import asyncio
import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from .server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "llm" in data
    assert "pinecone" in data
    assert "version" in data


def test_list_agents(client):
    """Test list agents endpoint."""
    response = client.get("/agents")
    assert response.status_code == 200

    data = response.json()
    assert "agents" in data
    assert "descriptions" in data
    assert "workflow_routing" in data

    # Check that expected agents are present
    expected_agents = [
        "harness_expert",
        "scaffold_agent",
        "codegen_agent",
        "kubernetes_agent",
        "monitoring_agent",
        "incident_agent",
        "database_agent",
        "testing_agent",
        "deployment_agent",
        "template_manager",
    ]

    for agent in expected_agents:
        assert agent in data["agents"]


def test_invoke_agent_missing_task(client):
    """Test agent invocation with missing task."""
    response = client.post("/agent/invoke", json={})
    assert response.status_code == 422  # Validation error


def test_invoke_agent_direct(client):
    """Test direct agent invocation."""
    request_data = {
        "task": "Create a basic CI/CD pipeline",
        "agent_name": "harness_expert",
        "context": {"service": "my-service"},
    }

    response = client.post("/agent/invoke", json=request_data)

    # Note: This may fail if ANTHROPIC_API_KEY is not set
    # In that case, we expect a 500 error
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "result" in data
        assert "agent_used" in data
        assert "execution_time" in data
        assert "thread_id" in data
        assert data["agent_used"] == "harness_expert"
    else:
        # Expected to fail without API key
        assert response.status_code in [500, 503]


def test_invoke_agent_workflow(client):
    """Test workflow-based agent invocation."""
    request_data = {
        "task": "Deploy my application to Kubernetes",
        "context": {"environment": "staging"},
    }

    response = client.post("/agent/invoke", json=request_data)

    # Note: This may fail if ANTHROPIC_API_KEY is not set
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "result" in data
        assert "agent_used" in data
        assert "execution_time" in data
        assert "thread_id" in data
    else:
        # Expected to fail without API key
        assert response.status_code in [500, 503]


def test_invoke_agent_with_stream_flag(client):
    """Test that invoke endpoint rejects stream=True."""
    request_data = {
        "task": "Create a pipeline",
        "agent_name": "harness_expert",
        "stream": True,
    }

    response = client.post("/agent/invoke", json=request_data)
    assert response.status_code == 400

    data = response.json()
    assert "streaming" in data["detail"].lower()


def test_workflow_status_not_found(client):
    """Test workflow status for non-existent thread."""
    response = client.get("/workflow/status/non-existent-thread-id")
    assert response.status_code == 404


def test_approve_workflow_not_found(client):
    """Test approval for non-existent thread."""
    response = client.post(
        "/workflow/approve/non-existent-thread-id",
        json={"approved": True},
    )
    assert response.status_code == 404


def test_task_type_inference():
    """Test task type inference helper."""
    from .server import _infer_task_type

    # Infrastructure tasks
    assert _infer_task_type("Deploy to Kubernetes") == "infrastructure"
    assert _infer_task_type("Create a CI/CD pipeline") == "infrastructure"
    assert _infer_task_type("Scaffold a new project") == "infrastructure"

    # Development tasks
    assert _infer_task_type("Generate API client") == "development"
    assert _infer_task_type("Create database migration") == "development"
    assert _infer_task_type("Generate tests") == "development"

    # Operations tasks
    assert _infer_task_type("Monitor application metrics") == "operations"
    assert _infer_task_type("Troubleshoot incident") == "operations"
    assert _infer_task_type("Check logs") == "operations"

    # Default
    assert _infer_task_type("Some random task") == "infrastructure"


def test_request_validation():
    """Test request model validation."""
    from .server import AgentRequest

    # Valid request
    request = AgentRequest(task="Test task")
    assert request.task == "Test task"
    assert request.agent_name is None
    assert request.context == {}
    assert request.stream is False
    assert request.thread_id is None

    # Request with all fields
    request = AgentRequest(
        task="Test task",
        agent_name="harness_expert",
        context={"key": "value"},
        stream=True,
        thread_id="test-thread-id",
    )
    assert request.task == "Test task"
    assert request.agent_name == "harness_expert"
    assert request.context == {"key": "value"}
    assert request.stream is True
    assert request.thread_id == "test-thread-id"


def test_response_validation():
    """Test response model validation."""
    from .server import AgentResponse

    # Valid response
    response = AgentResponse(
        status="success",
        result={"message": "Task completed"},
        agent_used="harness_expert",
        execution_time=1.23,
        thread_id="test-thread-id",
    )
    assert response.status == "success"
    assert response.result == {"message": "Task completed"}
    assert response.agent_used == "harness_expert"
    assert response.execution_time == 1.23
    assert response.thread_id == "test-thread-id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
