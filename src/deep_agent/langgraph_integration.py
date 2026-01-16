"""LangGraph integration for Deep Agent workflow orchestration.

This module provides the LangGraph workflow definition for coordinating
specialized subagents in infrastructure automation tasks.
"""

import logging
from typing import Annotated, Any, Dict, List, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the Deep Agent workflow.

    Attributes:
        messages: List of conversation messages
        task_type: Type of infrastructure task
        project_identifier: Harness project identifier
        context: Additional task context
        subagent_results: Results from subagent executions
        next_action: Next action to take in workflow
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    task_type: str
    project_identifier: str
    context: Dict[str, Any]
    subagent_results: Dict[str, Any]
    next_action: str


def analyze_task(state: AgentState) -> AgentState:
    """Analyze the incoming task and determine routing.

    Args:
        state: Current agent state

    Returns:
        Updated state with next_action
    """
    task_type = state["task_type"]
    logger.info(f"Analyzing task type: {task_type}")

    # Route based on task type
    if task_type in ["terraform", "iac", "infrastructure"]:
        state["next_action"] = "iac_architect"
    elif task_type in ["docker", "container", "dockerfile"]:
        state["next_action"] = "container_workflow"
    elif task_type in ["repository", "pipeline", "deployment"]:
        state["next_action"] = "team_accelerator"
    else:
        state["next_action"] = "general_orchestration"

    logger.info(f"Routing to: {state['next_action']}")
    return state


def iac_architect_node(state: AgentState) -> AgentState:
    """Execute infrastructure-as-code tasks.

    Delegates to iac-golden-architect subagent for Terraform operations.

    Args:
        state: Current agent state

    Returns:
        Updated state with IaC results
    """
    logger.info("Executing iac-golden-architect subagent")

    # Placeholder for actual IaC operations
    result = {
        "subagent": "iac-golden-architect",
        "action": "terraform_plan",
        "status": "success",
        "output": "Infrastructure plan generated successfully",
    }

    state["subagent_results"]["iac"] = result
    state["next_action"] = "complete"
    return state


def container_workflow_node(state: AgentState) -> AgentState:
    """Execute container-related tasks.

    Delegates to container-workflow subagent for Docker operations.

    Args:
        state: Current agent state

    Returns:
        Updated state with container results
    """
    logger.info("Executing container-workflow subagent")

    # Placeholder for actual container operations
    result = {
        "subagent": "container-workflow",
        "action": "dockerfile_review",
        "status": "success",
        "output": "Dockerfile validated and optimized",
    }

    state["subagent_results"]["container"] = result
    state["next_action"] = "complete"
    return state


def team_accelerator_node(state: AgentState) -> AgentState:
    """Execute team acceleration tasks.

    Delegates to team-accelerator subagent for repository and pipeline setup.

    Args:
        state: Current agent state

    Returns:
        Updated state with team accelerator results
    """
    logger.info("Executing team-accelerator subagent")

    # Placeholder for actual team acceleration operations
    result = {
        "subagent": "team-accelerator",
        "action": "repository_creation",
        "status": "success",
        "output": "Repository and pipeline configured",
    }

    state["subagent_results"]["team"] = result
    state["next_action"] = "complete"
    return state


def general_orchestration_node(state: AgentState) -> AgentState:
    """Handle general orchestration tasks.

    Coordinates multiple subagents for complex workflows.

    Args:
        state: Current agent state

    Returns:
        Updated state with orchestration results
    """
    logger.info("Executing general orchestration")

    result = {
        "subagent": "orchestrator",
        "action": "multi_agent_coordination",
        "status": "success",
        "output": "Task coordinated across subagents",
    }

    state["subagent_results"]["orchestration"] = result
    state["next_action"] = "complete"
    return state


def route_next_step(
    state: AgentState,
) -> Literal["iac_architect", "container_workflow", "team_accelerator", "general_orchestration", "end"]:
    """Route to the next workflow step based on state.

    Args:
        state: Current agent state

    Returns:
        Next node to execute
    """
    next_action = state.get("next_action", "end")

    if next_action == "complete":
        return "end"

    return next_action  # type: ignore


def create_agent_workflow() -> StateGraph:
    """Create the LangGraph workflow for the Deep Agent.

    Returns:
        Compiled workflow graph
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_task)
    workflow.add_node("iac_architect", iac_architect_node)
    workflow.add_node("container_workflow", container_workflow_node)
    workflow.add_node("team_accelerator", team_accelerator_node)
    workflow.add_node("general_orchestration", general_orchestration_node)

    # Define edges
    workflow.set_entry_point("analyze")

    workflow.add_conditional_edges(
        "analyze",
        route_next_step,
        {
            "iac_architect": "iac_architect",
            "container_workflow": "container_workflow",
            "team_accelerator": "team_accelerator",
            "general_orchestration": "general_orchestration",
            "end": END,
        },
    )

    # All subagent nodes route to end
    workflow.add_conditional_edges("iac_architect", route_next_step, {"end": END})
    workflow.add_conditional_edges("container_workflow", route_next_step, {"end": END})
    workflow.add_conditional_edges("team_accelerator", route_next_step, {"end": END})
    workflow.add_conditional_edges("general_orchestration", route_next_step, {"end": END})

    return workflow.compile()
