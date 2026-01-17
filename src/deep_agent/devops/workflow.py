"""LangGraph workflow for DevOps Engineer Multi-Agent System.

This module provides the LangGraph workflow definition for coordinating
specialized DevOps subagents with multi-level supervisor orchestration.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..memory_bus import AccessContext, MemoryBus
from .state import DevOpsAgentState
from .tools import get_devops_tools_by_category

logger = logging.getLogger(__name__)
CHECKPOINTER = MemorySaver()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_memory_bus(state: DevOpsAgentState) -> MemoryBus:
    """Get or initialize the shared memory bus in workflow state."""
    memory_bus = state.get("memory_bus")
    if isinstance(memory_bus, MemoryBus):
        return memory_bus
    memory_bus = MemoryBus()
    state["memory_bus"] = memory_bus
    return memory_bus


def record_supervisor_decision(
    state: DevOpsAgentState, supervisor: str, decision: str
) -> None:
    """Record a routing decision for multi-level supervisors."""
    state.setdefault("supervisor_path", [])
    state.setdefault("routing_trace", [])
    state["supervisor_path"].append(supervisor)
    state["routing_trace"].append(
        {
            "supervisor": supervisor,
            "decision": decision,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _utc_timestamp() -> str:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _aggregate_risk_tags(reflection_outcomes: Dict[str, Any]) -> List[str]:
    """Aggregate risk tags from reflection outcomes."""
    all_risks = []
    for outcome in reflection_outcomes.values():
        risks = outcome.get("risks", [])
        all_risks.extend(risks)
    return list(set(all_risks))


def _evaluate_agent_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate an agent result for quality, completeness, and risks."""
    status = result.get("status", "unknown")
    output = result.get("output", "")
    risks: List[str] = []
    quality = "high"
    completeness = "complete"

    if status != "success":
        quality = "low"
        completeness = "partial"
        risks.append("execution_failed")

    if not output:
        completeness = "partial"
        risks.append("missing_output")

    if isinstance(output, str) and "warning" in output.lower():
        risks.append("warnings_present")

    if isinstance(output, str) and "error" in output.lower():
        risks.append("errors_detected")
        quality = "low"

    return {"quality": quality, "completeness": completeness, "risks": risks}


# =============================================================================
# ROOT SUPERVISOR NODE
# =============================================================================


def devops_root_supervisor_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Analyze the incoming task and determine top-level routing.

    Routes to:
    - infrastructure_supervisor: For infra provisioning, scaffolding, deployments
    - development_supervisor: For code generation, database, testing
    - operations_supervisor: For monitoring, incidents, alerts

    Args:
        state: Current agent state

    Returns:
        Updated state with next_action
    """
    task_type = state["task_type"]
    context = state.get("context", {})

    logger.info(f"Root supervisor analyzing task type: {task_type}")

    # Determine routing based on task type
    if task_type in [
        "deploy",
        "provision",
        "scaffold",
        "pipeline",
        "infrastructure",
        "kubernetes",
        "k8s",
    ]:
        next_action = "infrastructure_supervisor"
    elif task_type in [
        "codegen",
        "database",
        "schema",
        "test",
        "migration",
        "api",
    ]:
        next_action = "development_supervisor"
    elif task_type in [
        "monitor",
        "incident_response",
        "alert",
        "debug",
        "incident",
        "logs",
        "metrics",
    ]:
        next_action = "operations_supervisor"
    else:
        # Default to infrastructure for general DevOps tasks
        next_action = "infrastructure_supervisor"

    state["next_action"] = next_action
    state["current_phase"] = "routing"
    record_supervisor_decision(state, "devops_root_supervisor", next_action)

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "workflow",
        "routing.root_supervisor",
        {
            "task_type": task_type,
            "next_action": next_action,
            "context": context,
            "timestamp": _utc_timestamp(),
        },
        access_context=AccessContext.for_workflow("devops_root_supervisor"),
    )

    logger.info(f"Root supervisor routing to: {next_action}")
    return state


# =============================================================================
# INFRASTRUCTURE SUPERVISOR NODE
# =============================================================================


def infrastructure_supervisor_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Route infrastructure tasks to the correct specialized agent.

    Routes to:
    - scaffold_agent: For project scaffolding and setup
    - harness_expert: For CI/CD pipeline creation
    - kubernetes_agent: For K8s operations and deployments

    Args:
        state: Current agent state

    Returns:
        Updated state with next_action
    """
    task_type = state["task_type"]
    context = state.get("context", {})

    logger.info(f"Infrastructure supervisor analyzing task type: {task_type}")

    if task_type in ["scaffold", "project_setup", "template"]:
        next_action = "scaffold_agent"
    elif task_type in ["pipeline", "deploy", "cicd", "harness"]:
        next_action = "harness_expert"
    elif task_type in ["kubernetes", "k8s", "provision", "cluster"]:
        next_action = "kubernetes_agent"
    else:
        # Default to harness for general infra tasks
        next_action = "harness_expert"

    state["next_action"] = next_action
    state["current_phase"] = "infrastructure_routing"
    record_supervisor_decision(state, "infrastructure_supervisor", next_action)

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "workflow",
        "routing.infrastructure_supervisor",
        {
            "task_type": task_type,
            "next_action": next_action,
            "context": context,
            "timestamp": _utc_timestamp(),
        },
        access_context=AccessContext.for_workflow("infrastructure_supervisor"),
    )

    logger.info(f"Infrastructure supervisor routing to: {next_action}")
    return state


# =============================================================================
# DEVELOPMENT SUPERVISOR NODE
# =============================================================================


def development_supervisor_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Route development tasks to the correct specialized agent.

    Routes to:
    - codegen_agent: For API client, model, and test generation
    - database_agent: For schema design, migrations, seeding
    - testing_agent: For test scaffolding and coverage analysis

    Args:
        state: Current agent state

    Returns:
        Updated state with next_action
    """
    task_type = state["task_type"]
    context = state.get("context", {})

    logger.info(f"Development supervisor analyzing task type: {task_type}")

    if task_type in ["codegen", "api", "client", "model"]:
        next_action = "codegen_agent"
    elif task_type in ["database", "schema", "migration", "seed"]:
        next_action = "database_agent"
    elif task_type in ["test", "coverage", "testing"]:
        next_action = "testing_agent"
    else:
        # Default to codegen for general dev tasks
        next_action = "codegen_agent"

    state["next_action"] = next_action
    state["current_phase"] = "development_routing"
    record_supervisor_decision(state, "development_supervisor", next_action)

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "workflow",
        "routing.development_supervisor",
        {
            "task_type": task_type,
            "next_action": next_action,
            "context": context,
            "timestamp": _utc_timestamp(),
        },
        access_context=AccessContext.for_workflow("development_supervisor"),
    )

    logger.info(f"Development supervisor routing to: {next_action}")
    return state


# =============================================================================
# OPERATIONS SUPERVISOR NODE
# =============================================================================


def operations_supervisor_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Route operations tasks to the correct specialized agent.

    Routes to:
    - monitoring_agent: For metrics, logs, and observability
    - incident_agent: For incident response and root cause analysis

    Args:
        state: Current agent state

    Returns:
        Updated state with next_action
    """
    task_type = state["task_type"]
    context = state.get("context", {})

    logger.info(f"Operations supervisor analyzing task type: {task_type}")

    if task_type in ["monitor", "metrics", "logs", "alert", "observability"]:
        next_action = "monitoring_agent"
    elif task_type in ["incident_response", "incident", "debug", "troubleshoot"]:
        next_action = "incident_agent"
    else:
        # Default to monitoring for general ops tasks
        next_action = "monitoring_agent"

    state["next_action"] = next_action
    state["current_phase"] = "operations_routing"
    record_supervisor_decision(state, "operations_supervisor", next_action)

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "workflow",
        "routing.operations_supervisor",
        {
            "task_type": task_type,
            "next_action": next_action,
            "context": context,
            "timestamp": _utc_timestamp(),
        },
        access_context=AccessContext.for_workflow("operations_supervisor"),
    )

    logger.info(f"Operations supervisor routing to: {next_action}")
    return state


# =============================================================================
# AGENT NODES
# =============================================================================


def scaffold_agent_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute project scaffolding tasks.

    Uses tools: harness_pipeline, documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with scaffolding results
    """
    logger.info("Executing scaffold-agent")

    context = state.get("context", {})

    # Placeholder for actual agent invocation
    # TODO: Integrate with actual scaffold-agent from templating plugin
    result = {
        "agent": "scaffold-agent",
        "action": "project_scaffolding",
        "status": "success",
        "output": f"Project scaffolded successfully with template: {context.get('template', 'default')}",
        "artifacts": ["project_structure", "config_files", "documentation"],
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["scaffold"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "scaffold_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "scaffold.result",
        result,
        access_context=AccessContext.for_agent("scaffold_agent"),
    )

    return state


def harness_expert_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute Harness CI/CD pipeline tasks.

    Uses tools: harness_pipeline, documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with Harness results
    """
    logger.info("Executing harness-expert")

    context = state.get("context", {})

    # Placeholder for actual agent invocation
    # TODO: Integrate with actual harness-expert from templating plugin
    result = {
        "agent": "harness-expert",
        "action": "pipeline_creation",
        "status": "success",
        "output": f"Pipeline created: {context.get('pipeline_name', 'default-pipeline')}",
        "pipeline_id": context.get("pipeline_id", "pipeline-001"),
        "artifacts": ["pipeline_yaml", "connectors", "secrets"],
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["harness"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "harness_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "harness.result",
        result,
        access_context=AccessContext.for_agent("harness_expert"),
    )

    return state


def kubernetes_agent_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute Kubernetes cluster operations.

    Uses tools: kubernetes_query, kubernetes_action, documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with K8s results
    """
    logger.info("Executing kubernetes-agent")

    context = state.get("context", {})
    infra_state = state.get("infrastructure_state", {})

    # Placeholder for actual agent invocation
    result = {
        "agent": "kubernetes-agent",
        "action": context.get("action", "deploy"),
        "status": "success",
        "output": f"Kubernetes operation completed: {context.get('action', 'deploy')}",
        "resources": infra_state.get("resources", ["deployment", "service", "ingress"]),
        "namespace": context.get("namespace", "default"),
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["kubernetes"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "kubernetes_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "kubernetes.result",
        result,
        access_context=AccessContext.for_agent("kubernetes_agent"),
    )

    return state


def codegen_agent_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute code generation tasks.

    Uses tools: documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with codegen results
    """
    logger.info("Executing codegen-agent")

    context = state.get("context", {})

    # Placeholder for actual agent invocation
    # TODO: Integrate with actual codegen-agent from templating plugin
    result = {
        "agent": "codegen-agent",
        "action": "code_generation",
        "status": "success",
        "output": f"Generated {context.get('generation_type', 'API client')}",
        "artifacts": ["api_client", "models", "tests"],
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["codegen"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "codegen_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "codegen.result",
        result,
        access_context=AccessContext.for_agent("codegen_agent"),
    )

    return state


def database_agent_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute database schema and migration tasks.

    Uses tools: documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with database results
    """
    logger.info("Executing database-agent")

    context = state.get("context", {})

    # Placeholder for actual agent invocation
    # TODO: Integrate with actual database-agent from templating plugin
    result = {
        "agent": "database-agent",
        "action": context.get("action", "schema_design"),
        "status": "success",
        "output": f"Database operation completed: {context.get('action', 'schema_design')}",
        "artifacts": ["schema", "migrations", "seeds"],
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["database"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "database_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "database.result",
        result,
        access_context=AccessContext.for_agent("database_agent"),
    )

    return state


def testing_agent_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute testing and coverage tasks.

    Uses tools: documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with testing results
    """
    logger.info("Executing testing-agent")

    context = state.get("context", {})

    # Placeholder for actual agent invocation
    # TODO: Integrate with actual testing-agent from templating plugin
    result = {
        "agent": "testing-agent",
        "action": context.get("action", "test_scaffolding"),
        "status": "success",
        "output": f"Testing operation completed: {context.get('action', 'test_scaffolding')}",
        "coverage": context.get("coverage", "85%"),
        "artifacts": ["test_suite", "coverage_report"],
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["testing"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "testing_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "testing.result",
        result,
        access_context=AccessContext.for_agent("testing_agent"),
    )

    return state


def monitoring_agent_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute monitoring and observability tasks.

    Uses tools: metrics_query, log_search, alert_manager, documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with monitoring results
    """
    logger.info("Executing monitoring-agent")

    context = state.get("context", {})
    metrics = state.get("metrics", {})
    logs = state.get("logs", [])

    # Placeholder for actual agent invocation
    result = {
        "agent": "monitoring-agent",
        "action": context.get("action", "metrics_analysis"),
        "status": "success",
        "output": "Monitoring analysis completed successfully",
        "metrics_summary": metrics,
        "log_entries": len(logs),
        "alerts_active": len(state.get("alerts", [])),
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["monitoring"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "monitoring_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "monitoring.result",
        result,
        access_context=AccessContext.for_agent("monitoring_agent"),
    )

    return state


def incident_agent_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Execute incident response and root cause analysis.

    Uses tools: log_search, metrics_query, kubernetes_query, web_search,
                documentation_search

    Args:
        state: Current agent state

    Returns:
        Updated state with incident response results
    """
    logger.info("Executing incident-agent")

    context = state.get("context", {})
    alerts = state.get("alerts", [])

    # Placeholder for actual agent invocation
    result = {
        "agent": "incident-agent",
        "action": "incident_response",
        "status": "success",
        "output": "Incident analysis completed",
        "root_cause": context.get("root_cause", "Under investigation"),
        "remediation_steps": context.get("remediation_steps", []),
        "active_alerts": len(alerts),
        "timestamp": _utc_timestamp(),
    }

    state.setdefault("subagent_results", {})
    state["subagent_results"]["incident"] = result
    state["next_action"] = "reflection"
    state["current_phase"] = "incident_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "agent",
        "incident.result",
        result,
        access_context=AccessContext.for_agent("incident_agent"),
    )

    return state


# =============================================================================
# REFLECTION NODE
# =============================================================================


def reflection_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Evaluate agent results and determine remediation or completion.

    Analyzes all agent results for quality, completeness, and risks.
    Generates remediation tasks if needed or routes to completion.

    Args:
        state: Current agent state

    Returns:
        Updated state with reflection outcomes
    """
    logger.info("Reflecting on agent results for quality checks")

    results = state.get("subagent_results", {})
    reflection_outcomes: Dict[str, Any] = {}
    remediation_tasks: List[Dict[str, Any]] = []
    needs_approval = False

    for result_key, result in results.items():
        outcome = _evaluate_agent_result(result)
        reflection_outcomes[result_key] = outcome

        # Check if result needs remediation
        if outcome["risks"] or outcome["quality"] != "high" or outcome["completeness"] != "complete":
            remediation_tasks.append(
                {
                    "agent": result.get("agent", result_key),
                    "issue": outcome,
                    "action": "review_and_retry",
                    "timestamp": _utc_timestamp(),
                }
            )

            # Record failure
            state.setdefault("failure_history", [])
            state["failure_history"].append(
                {
                    "timestamp": _utc_timestamp(),
                    "result_key": result_key,
                    "agent": result.get("agent", result_key),
                    "issue": outcome,
                }
            )

        # Check if result requires human approval
        action = result.get("action", "")
        if action in ["destroy", "delete", "production_deploy"]:
            needs_approval = True

    state["reflection_outcomes"] = reflection_outcomes

    if remediation_tasks:
        state.setdefault("remediation_tasks", [])
        state["remediation_tasks"].extend(remediation_tasks)

    # Update context signals for orchestration
    context_signals = state.setdefault("context_signals", {})
    risk_tags = _aggregate_risk_tags(reflection_outcomes)
    context_signals.update(
        {
            "needs_retry": bool(remediation_tasks),
            "needs_approval": needs_approval,
            "risk_tags": risk_tags,
            "recent_failure_count": len(state.get("failure_history", [])),
            "last_failure_agents": [
                entry["agent"] for entry in state.get("failure_history", [])[-3:]
            ],
        }
    )

    # Determine next action
    if needs_approval:
        state["next_action"] = "human_approval"
    elif remediation_tasks:
        # Re-route to appropriate supervisor for retry
        first_failed = remediation_tasks[0]["agent"]
        if first_failed in ["scaffold-agent", "harness-expert", "kubernetes-agent"]:
            state["next_action"] = "infrastructure_supervisor"
        elif first_failed in ["codegen-agent", "database-agent", "testing-agent"]:
            state["next_action"] = "development_supervisor"
        elif first_failed in ["monitoring-agent", "incident-agent"]:
            state["next_action"] = "operations_supervisor"
        else:
            state["next_action"] = "devops_root_supervisor"
    else:
        state["next_action"] = "complete"

    state["current_phase"] = "reflection_complete"

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "workflow",
        "reflection.outcomes",
        {
            "outcomes": reflection_outcomes,
            "remediation_tasks": remediation_tasks,
            "timestamp": _utc_timestamp(),
        },
        access_context=AccessContext.for_workflow("reflection"),
    )
    memory_bus.set(
        "workflow",
        "reflection.context_signals",
        context_signals,
        access_context=AccessContext.for_workflow("reflection"),
    )

    logger.info(f"Reflection completed; next action: {state['next_action']}")
    return state


# =============================================================================
# HUMAN APPROVAL GATE NODE
# =============================================================================


def human_approval_node(state: DevOpsAgentState) -> DevOpsAgentState:
    """Human approval gate for destructive operations.

    Pauses workflow for human review of:
    - Production deployments
    - Destructive operations (delete, destroy)
    - High-risk changes

    Args:
        state: Current agent state

    Returns:
        Updated state waiting for approval
    """
    logger.info("Human approval gate activated")

    results = state.get("subagent_results", {})

    # Collect operations requiring approval
    approval_items = []
    for result_key, result in results.items():
        action = result.get("action", "")
        if action in ["destroy", "delete", "production_deploy"]:
            approval_items.append(
                {
                    "agent": result.get("agent"),
                    "action": action,
                    "resource": result.get("resource", result_key),
                }
            )

    state["approval_required"] = approval_items
    state["current_phase"] = "awaiting_approval"

    # Add message to conversation
    approval_msg = (
        f"Human approval required for {len(approval_items)} operations:\n"
        + "\n".join(
            f"- {item['agent']}: {item['action']} on {item['resource']}"
            for item in approval_items
        )
    )
    state.setdefault("messages", [])
    state["messages"].append(AIMessage(content=approval_msg))

    memory_bus = get_memory_bus(state)
    memory_bus.set(
        "workflow",
        "approval.pending",
        {
            "items": approval_items,
            "timestamp": _utc_timestamp(),
        },
        access_context=AccessContext.for_workflow("human_approval"),
    )

    logger.info(f"Approval required for {len(approval_items)} operations")

    # In production, this would pause for actual human input
    # For now, default to approved
    state["next_action"] = "complete"

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_from_root_supervisor(
    state: DevOpsAgentState,
) -> Literal[
    "infrastructure_supervisor",
    "development_supervisor",
    "operations_supervisor",
    "end",
]:
    """Route from root supervisor to specialized supervisors."""
    next_action = state.get("next_action", "end")
    if next_action == "complete":
        return "end"
    return next_action  # type: ignore


def route_from_infra_supervisor(
    state: DevOpsAgentState,
) -> Literal["scaffold_agent", "harness_expert", "kubernetes_agent", "end"]:
    """Route from infrastructure supervisor to agents."""
    next_action = state.get("next_action", "end")
    if next_action == "complete":
        return "end"
    return next_action  # type: ignore


def route_from_dev_supervisor(
    state: DevOpsAgentState,
) -> Literal["codegen_agent", "database_agent", "testing_agent", "end"]:
    """Route from development supervisor to agents."""
    next_action = state.get("next_action", "end")
    if next_action == "complete":
        return "end"
    return next_action  # type: ignore


def route_from_ops_supervisor(
    state: DevOpsAgentState,
) -> Literal["monitoring_agent", "incident_agent", "end"]:
    """Route from operations supervisor to agents."""
    next_action = state.get("next_action", "end")
    if next_action == "complete":
        return "end"
    return next_action  # type: ignore


def route_after_reflection(
    state: DevOpsAgentState,
) -> Literal[
    "human_approval",
    "infrastructure_supervisor",
    "development_supervisor",
    "operations_supervisor",
    "devops_root_supervisor",
    "end",
]:
    """Route after reflection based on outcomes."""
    next_action = state.get("next_action", "end")
    if next_action == "complete":
        return "end"
    return next_action  # type: ignore


def route_after_agent(state: DevOpsAgentState) -> Literal["reflection", "end"]:
    """Route from agent nodes to reflection."""
    next_action = state.get("next_action", "end")
    if next_action == "reflection":
        return "reflection"
    return "end"


def route_after_approval(state: DevOpsAgentState) -> Literal["end"]:
    """Route after human approval."""
    # In production, check actual approval status
    # For now, always proceed to end
    return "end"


# =============================================================================
# WORKFLOW CREATION
# =============================================================================


def create_devops_workflow() -> StateGraph:
    """Create the LangGraph workflow for the DevOps Engineer Multi-Agent System.

    Builds a multi-level supervisor hierarchy with:
    - Root supervisor for top-level routing
    - Infrastructure supervisor for deployment/provisioning tasks
    - Development supervisor for code/database/testing tasks
    - Operations supervisor for monitoring/incident tasks
    - Specialized agent nodes for each capability
    - Reflection node for quality checks
    - Human approval gate for destructive operations

    Returns:
        Compiled workflow graph with checkpointing
    """
    logger.info("Creating DevOps multi-agent workflow")

    workflow = StateGraph(DevOpsAgentState)

    # Add supervisor nodes
    workflow.add_node("devops_root_supervisor", devops_root_supervisor_node)
    workflow.add_node("infrastructure_supervisor", infrastructure_supervisor_node)
    workflow.add_node("development_supervisor", development_supervisor_node)
    workflow.add_node("operations_supervisor", operations_supervisor_node)

    # Add infrastructure agent nodes
    workflow.add_node("scaffold_agent", scaffold_agent_node)
    workflow.add_node("harness_expert", harness_expert_node)
    workflow.add_node("kubernetes_agent", kubernetes_agent_node)

    # Add development agent nodes
    workflow.add_node("codegen_agent", codegen_agent_node)
    workflow.add_node("database_agent", database_agent_node)
    workflow.add_node("testing_agent", testing_agent_node)

    # Add operations agent nodes
    workflow.add_node("monitoring_agent", monitoring_agent_node)
    workflow.add_node("incident_agent", incident_agent_node)

    # Add reflection and approval nodes
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("human_approval", human_approval_node)

    # Set entry point
    workflow.set_entry_point("devops_root_supervisor")

    # Root supervisor routing
    workflow.add_conditional_edges(
        "devops_root_supervisor",
        route_from_root_supervisor,
        {
            "infrastructure_supervisor": "infrastructure_supervisor",
            "development_supervisor": "development_supervisor",
            "operations_supervisor": "operations_supervisor",
            "end": END,
        },
    )

    # Infrastructure supervisor routing
    workflow.add_conditional_edges(
        "infrastructure_supervisor",
        route_from_infra_supervisor,
        {
            "scaffold_agent": "scaffold_agent",
            "harness_expert": "harness_expert",
            "kubernetes_agent": "kubernetes_agent",
            "end": END,
        },
    )

    # Development supervisor routing
    workflow.add_conditional_edges(
        "development_supervisor",
        route_from_dev_supervisor,
        {
            "codegen_agent": "codegen_agent",
            "database_agent": "database_agent",
            "testing_agent": "testing_agent",
            "end": END,
        },
    )

    # Operations supervisor routing
    workflow.add_conditional_edges(
        "operations_supervisor",
        route_from_ops_supervisor,
        {
            "monitoring_agent": "monitoring_agent",
            "incident_agent": "incident_agent",
            "end": END,
        },
    )

    # Agent to reflection routing
    for agent in [
        "scaffold_agent",
        "harness_expert",
        "kubernetes_agent",
        "codegen_agent",
        "database_agent",
        "testing_agent",
        "monitoring_agent",
        "incident_agent",
    ]:
        workflow.add_conditional_edges(
            agent,
            route_after_agent,
            {"reflection": "reflection", "end": END},
        )

    # Reflection routing (includes retry loops and approval)
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {
            "human_approval": "human_approval",
            "infrastructure_supervisor": "infrastructure_supervisor",
            "development_supervisor": "development_supervisor",
            "operations_supervisor": "operations_supervisor",
            "devops_root_supervisor": "devops_root_supervisor",
            "end": END,
        },
    )

    # Human approval routing
    workflow.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"end": END},
    )

    logger.info("Compiling DevOps workflow with checkpointing")
    return workflow.compile(checkpointer=CHECKPOINTER)


__all__ = [
    "create_devops_workflow",
    "DevOpsAgentState",
]
