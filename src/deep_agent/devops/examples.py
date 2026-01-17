"""Examples demonstrating how to use the DevOps tools.

This module provides example code showing how to integrate and use
the various DevOps tools in LangChain agents and workflows.
"""

import asyncio
import logging
from typing import List

from langchain_core.tools import StructuredTool

from .tools import (
    get_all_devops_tools,
    get_devops_tools_by_category,
    create_kubernetes_query_tool,
    create_harness_pipeline_tool,
    create_web_search_tool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_individual_tool_usage():
    """Example: Using individual DevOps tools."""
    logger.info("=== Example: Individual Tool Usage ===")

    # Create individual tools
    k8s_query = create_kubernetes_query_tool()
    harness = create_harness_pipeline_tool()
    web_search = create_web_search_tool()

    # Example 1: Query Kubernetes pods
    logger.info("\n1. Querying Kubernetes pods...")
    pods_result = await k8s_query.ainvoke({
        "resource_type": "pods",
        "namespace": "production",
        "label_selector": "app=api-server"
    })
    logger.info(f"Pods result: {pods_result}")

    # Example 2: Search web for solutions
    logger.info("\n2. Searching web for Kubernetes troubleshooting...")
    search_result = await web_search.ainvoke({
        "query": "Kubernetes pod CrashLoopBackOff troubleshooting",
        "max_results": 3
    })
    logger.info(f"Search result: {search_result}")

    # Example 3: Trigger Harness pipeline
    logger.info("\n3. Triggering Harness pipeline...")
    pipeline_result = await harness.ainvoke({
        "action": "trigger",
        "pipeline_id": "ci-pipeline-001",
        "inputs": {"branch": "main", "environment": "staging"}
    })
    logger.info(f"Pipeline result: {pipeline_result}")


async def example_all_tools_usage():
    """Example: Get all tools and use them in an agent."""
    logger.info("=== Example: Using All Tools ===")

    # Get all tools
    all_tools = get_all_devops_tools()
    logger.info(f"\nTotal tools available: {len(all_tools)}")

    for tool in all_tools:
        logger.info(f"  - {tool.name}: {tool.description[:80]}...")

    # Example usage with a hypothetical agent
    # from langchain.agents import AgentExecutor, create_openai_functions_agent
    # from langchain_openai import ChatOpenAI
    # from langchain.prompts import ChatPromptTemplate
    #
    # llm = ChatOpenAI(model="gpt-4", temperature=0)
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "You are a DevOps expert assistant."),
    #     ("user", "{input}"),
    #     ("assistant", "{agent_scratchpad}"),
    # ])
    #
    # agent = create_openai_functions_agent(llm, all_tools, prompt)
    # executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)
    #
    # result = await executor.ainvoke({
    #     "input": "Check if there are any pods failing in the production namespace"
    # })


async def example_categorized_tools_usage():
    """Example: Using tools organized by category."""
    logger.info("=== Example: Categorized Tools Usage ===")

    # Get tools by category
    tools_by_category = get_devops_tools_by_category()

    for category, tools in tools_by_category.items():
        logger.info(f"\n{category.upper()} Tools ({len(tools)}):")
        for tool in tools:
            logger.info(f"  - {tool.name}")

    # Use only observability tools for monitoring agent
    observability_tools = tools_by_category["observability"]
    logger.info(f"\nCreating monitoring agent with {len(observability_tools)} tools")

    # Example: Check metrics
    metrics_tool = observability_tools[0]
    result = await metrics_tool.ainvoke({
        "query": "rate(http_requests_total[5m])",
        "start_time": "-1h",
        "end_time": "now"
    })
    logger.info(f"Metrics result: {result}")


async def example_kubernetes_workflow():
    """Example: Complete Kubernetes troubleshooting workflow."""
    logger.info("=== Example: Kubernetes Troubleshooting Workflow ===")

    k8s_query = create_kubernetes_query_tool()
    k8s_action = create_kubernetes_action_tool()
    logs = create_log_search_tool()
    web_search = create_web_search_tool()

    # Step 1: Query failing pods
    logger.info("\n1. Checking for failing pods...")
    pods = await k8s_query.ainvoke({
        "resource_type": "pods",
        "namespace": "production",
        "label_selector": None
    })
    logger.info(f"Found {len(pods)} pods")

    # Step 2: Search logs for errors
    logger.info("\n2. Searching logs for errors...")
    error_logs = await logs.ainvoke({
        "query": "error",
        "service": "api-server",
        "time_range": "1h",
        "level": "error"
    })
    logger.info(f"Found {len(error_logs)} error log entries")

    # Step 3: Search web for solutions
    logger.info("\n3. Searching for solutions...")
    if error_logs and len(error_logs) > 0:
        error_msg = error_logs[0].get("message", "")
        solutions = await web_search.ainvoke({
            "query": f"Kubernetes {error_msg[:100]}",
            "max_results": 3
        })
        logger.info(f"Found {len(solutions)} potential solutions")

    # Step 4: Apply fix (example: restart deployment)
    logger.info("\n4. Applying fix - restarting deployment...")
    restart_result = await k8s_action.ainvoke({
        "action": "restart",
        "resource_type": "deployment",
        "resource_name": "api-server",
        "namespace": "production",
        "params": {}
    })
    logger.info(f"Restart result: {restart_result['status']}")


async def example_cicd_pipeline_workflow():
    """Example: CI/CD pipeline monitoring and management."""
    logger.info("=== Example: CI/CD Pipeline Workflow ===")

    harness = create_harness_pipeline_tool()
    metrics = create_metrics_query_tool()
    alerts = create_alert_manager_tool()

    # Step 1: List available pipelines
    logger.info("\n1. Listing available pipelines...")
    pipelines = await harness.ainvoke({
        "action": "list",
        "pipeline_id": None,
        "inputs": {}
    })
    logger.info(f"Available pipelines: {pipelines.get('pipelines', [])}")

    # Step 2: Trigger deployment pipeline
    logger.info("\n2. Triggering deployment pipeline...")
    execution = await harness.ainvoke({
        "action": "trigger",
        "pipeline_id": "deploy-to-staging",
        "inputs": {
            "image_tag": "v1.2.3",
            "environment": "staging",
            "rollback_enabled": True
        }
    })
    execution_id = execution.get("execution_id")
    logger.info(f"Pipeline triggered: {execution_id}")

    # Step 3: Monitor deployment metrics
    logger.info("\n3. Monitoring deployment metrics...")
    deployment_metrics = await metrics.ainvoke({
        "query": "deployment_status{environment='staging'}",
        "start_time": "-5m",
        "end_time": "now"
    })
    logger.info(f"Deployment metrics: {deployment_metrics.get('resultType')}")

    # Step 4: Check for alerts
    logger.info("\n4. Checking for deployment alerts...")
    active_alerts = await alerts.ainvoke({
        "action": "list",
        "alert_name": None
    })
    logger.info(f"Active alerts: {len(active_alerts.get('alerts', []))}")


async def main():
    """Run all examples."""
    logger.info("Starting DevOps Tools Examples\n")

    try:
        await example_individual_tool_usage()
        await asyncio.sleep(1)

        await example_all_tools_usage()
        await asyncio.sleep(1)

        await example_categorized_tools_usage()
        await asyncio.sleep(1)

        await example_kubernetes_workflow()
        await asyncio.sleep(1)

        await example_cicd_pipeline_workflow()

        logger.info("\n=== All Examples Completed Successfully ===")
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
