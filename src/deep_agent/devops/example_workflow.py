"""Example usage of the DevOps multi-agent workflow.

This script demonstrates how to create and execute the DevOps workflow
for various task types.
"""

import asyncio
import logging
from typing import Dict, Any

from langchain_core.messages import HumanMessage

from .workflow import create_devops_workflow
from .state import DevOpsAgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_initial_state(
    task_type: str,
    context: Dict[str, Any],
    user_message: str,
) -> DevOpsAgentState:
    """Create initial state for workflow execution.

    Args:
        task_type: Type of DevOps task
        context: Additional task context
        user_message: User's initial message

    Returns:
        Initial agent state
    """
    return {
        "messages": [HumanMessage(content=user_message)],
        "task_type": task_type,
        "context": context,
        "current_phase": "init",
        "supervisor_path": [],
        "routing_trace": [],
        "next_action": "",
        "subagent_results": {},
        "active_agents": [],
        "infrastructure_state": {},
        "metrics": {},
        "logs": [],
        "alerts": [],
        "retrieved_docs": [],
        "search_queries": [],
        "web_search_results": [],
        "api_responses": {},
        "memory_bus": None,
        "reflection_outcomes": {},
        "remediation_tasks": [],
    }


async def run_deployment_workflow():
    """Example: Run a deployment workflow."""
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Kubernetes Deployment")
    logger.info("=" * 80)

    workflow = create_devops_workflow()

    initial_state = create_initial_state(
        task_type="deploy",
        context={
            "action": "deploy",
            "namespace": "production",
            "service_name": "api-server",
            "image": "api-server:v1.2.3",
        },
        user_message="Deploy api-server v1.2.3 to production",
    )

    config = {"configurable": {"thread_id": "deploy-001"}}

    logger.info("Executing deployment workflow...")
    result = await workflow.ainvoke(initial_state, config)

    logger.info("\nWorkflow completed!")
    logger.info(f"Supervisor path: {result['supervisor_path']}")
    logger.info(f"Final phase: {result['current_phase']}")
    logger.info(f"Results: {list(result['subagent_results'].keys())}")

    return result


async def run_incident_workflow():
    """Example: Run an incident response workflow."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Incident Response")
    logger.info("=" * 80)

    workflow = create_devops_workflow()

    initial_state = create_initial_state(
        task_type="incident_response",
        context={
            "incident_id": "INC-12345",
            "severity": "high",
            "service": "payment-service",
            "symptom": "high_latency",
        },
        user_message="Investigate high latency in payment-service",
    )

    config = {"configurable": {"thread_id": "incident-001"}}

    logger.info("Executing incident response workflow...")
    result = await workflow.ainvoke(initial_state, config)

    logger.info("\nWorkflow completed!")
    logger.info(f"Supervisor path: {result['supervisor_path']}")
    logger.info(f"Final phase: {result['current_phase']}")
    logger.info(f"Results: {list(result['subagent_results'].keys())}")

    return result


async def run_pipeline_workflow():
    """Example: Run a CI/CD pipeline creation workflow."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Pipeline Creation")
    logger.info("=" * 80)

    workflow = create_devops_workflow()

    initial_state = create_initial_state(
        task_type="pipeline",
        context={
            "pipeline_name": "microservice-cicd",
            "repository": "https://github.com/org/microservice",
            "environments": ["dev", "staging", "prod"],
        },
        user_message="Create CI/CD pipeline for microservice",
    )

    config = {"configurable": {"thread_id": "pipeline-001"}}

    logger.info("Executing pipeline creation workflow...")
    result = await workflow.ainvoke(initial_state, config)

    logger.info("\nWorkflow completed!")
    logger.info(f"Supervisor path: {result['supervisor_path']}")
    logger.info(f"Final phase: {result['current_phase']}")
    logger.info(f"Results: {list(result['subagent_results'].keys())}")

    return result


async def run_codegen_workflow():
    """Example: Run a code generation workflow."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: API Client Generation")
    logger.info("=" * 80)

    workflow = create_devops_workflow()

    initial_state = create_initial_state(
        task_type="codegen",
        context={
            "generation_type": "API client",
            "spec_url": "https://api.example.com/openapi.json",
            "language": "python",
        },
        user_message="Generate Python API client from OpenAPI spec",
    )

    config = {"configurable": {"thread_id": "codegen-001"}}

    logger.info("Executing code generation workflow...")
    result = await workflow.ainvoke(initial_state, config)

    logger.info("\nWorkflow completed!")
    logger.info(f"Supervisor path: {result['supervisor_path']}")
    logger.info(f"Final phase: {result['current_phase']}")
    logger.info(f"Results: {list(result['subagent_results'].keys())}")

    return result


async def run_database_workflow():
    """Example: Run a database migration workflow."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Database Migration")
    logger.info("=" * 80)

    workflow = create_devops_workflow()

    initial_state = create_initial_state(
        task_type="database",
        context={
            "action": "migration",
            "database": "users_db",
            "migration_type": "add_column",
            "table": "users",
            "column": "email_verified",
        },
        user_message="Create migration to add email_verified column to users table",
    )

    config = {"configurable": {"thread_id": "database-001"}}

    logger.info("Executing database workflow...")
    result = await workflow.ainvoke(initial_state, config)

    logger.info("\nWorkflow completed!")
    logger.info(f"Supervisor path: {result['supervisor_path']}")
    logger.info(f"Final phase: {result['current_phase']}")
    logger.info(f"Results: {list(result['subagent_results'].keys())}")

    return result


async def run_monitoring_workflow():
    """Example: Run a monitoring analysis workflow."""
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 6: Monitoring Analysis")
    logger.info("=" * 80)

    workflow = create_devops_workflow()

    initial_state = create_initial_state(
        task_type="monitor",
        context={
            "action": "metrics_analysis",
            "service": "api-gateway",
            "time_range": "1h",
            "metrics": ["cpu_usage", "memory_usage", "request_rate"],
        },
        user_message="Analyze metrics for api-gateway over the last hour",
    )

    config = {"configurable": {"thread_id": "monitoring-001"}}

    logger.info("Executing monitoring workflow...")
    result = await workflow.ainvoke(initial_state, config)

    logger.info("\nWorkflow completed!")
    logger.info(f"Supervisor path: {result['supervisor_path']}")
    logger.info(f"Final phase: {result['current_phase']}")
    logger.info(f"Results: {list(result['subagent_results'].keys())}")

    return result


async def main():
    """Run all example workflows."""
    logger.info("DevOps Multi-Agent Workflow Examples")
    logger.info("=" * 80)

    # Run examples
    await run_deployment_workflow()
    await run_incident_workflow()
    await run_pipeline_workflow()
    await run_codegen_workflow()
    await run_database_workflow()
    await run_monitoring_workflow()

    logger.info("\n" + "=" * 80)
    logger.info("All examples completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
