"""LangGraph Node Implementations for Deep Agent.

This module provides the core node functions for the deep agent workflow:
- Supervisor: Routes tasks to appropriate sub-agents
- Planning: Creates execution plans for complex tasks
- Retrieval: Performs RAG-based knowledge retrieval
- React Agent: Implements ReAct reasoning pattern
- Execution: Executes planned steps
- Reflection: Evaluates outputs and determines next steps
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, TypeVar

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, ValidationError

from .state import (
    AgentPhase,
    DeepAgentState,
)
from .llm import create_llm

logger = structlog.get_logger(__name__)


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor agent for a deep AI orchestration system.
Your role is to analyze tasks and route them to the appropriate specialized sub-agents.

Available sub-agents:
- planner: Creates detailed execution plans for complex tasks
- researcher: Performs research and retrieves relevant information
- coder: Writes and reviews code
- reviewer: Reviews work quality and provides feedback
- executor: Executes planned steps and actions
- iac_architect: Infrastructure as code expert (Terraform, CloudFormation)
- container_workflow: Container and Docker expertise
- team_accelerator: Repository and pipeline setup

Based on the task, decide which agent(s) should handle it and in what order.
Respond with JSON only using this schema:
{
    "analysis": "Brief analysis of the task",
    "agents": ["list", "of", "agents"],
    "reasoning": "Why these agents were selected"
}
Do not include markdown, comments, or extra keys.
"""

PLANNER_SYSTEM_PROMPT = """You are a Planning agent specialized in breaking down complex tasks.
Create detailed, step-by-step execution plans that other agents can follow.

For each step, specify:
1. A clear description of what needs to be done
2. Which agent type should execute it
3. Any dependencies on previous steps
4. Expected inputs and outputs

Return JSON only using this schema:
{
    "goal": "Overall goal",
    "assumptions": ["Assumption 1"],
    "constraints": ["Constraint 1"],
    "success_criteria": ["Criterion 1"],
    "steps": [
        {
            "description": "Step description",
            "agent_type": "agent_name",
            "dependencies": [],
            "inputs": {},
            "expected_outputs": []
        }
    ]
}
Do not include markdown, comments, or extra keys.
"""

REACT_SYSTEM_PROMPT = """You are a ReAct agent that reasons step-by-step and uses tools to accomplish tasks.

Follow this pattern:
1. Thought: Analyze what you need to do next
2. Action: Use an appropriate tool if needed
3. Observation: Process the tool result
4. Repeat until the task is complete

Be thorough but efficient. Only use tools when necessary.
"""

REFLECTION_SYSTEM_PROMPT = """You are a Reflection agent that evaluates work quality and completeness.

Analyze the outputs from previous steps and provide:
1. A quality score (0.0 to 1.0)
2. Assessment of completeness (complete, partial, incomplete)
3. List of potential risks or issues
4. Suggestions for improvements
5. Whether a retry is needed

Be constructive but honest. Focus on actionable feedback.
Return JSON only using this schema:
{
    "quality_score": 0.0,
    "completeness": "complete",
    "risks": ["risk_1"],
    "improvements": ["improvement_1"],
    "should_retry": false,
    "retry_reason": "optional short reason"
}
Do not include markdown, comments, or extra keys.
"""


# =============================================================================
# STRUCTURED OUTPUT MODELS
# =============================================================================

class SupervisorDecision(BaseModel):
    """Structured supervisor routing decision."""

    analysis: str = ""
    agents: List[str] = Field(default_factory=lambda: ["planner"])
    reasoning: str = ""


class PlanStepSpec(BaseModel):
    """Structured plan step specification."""

    description: str = ""
    agent_type: str = "executor"
    dependencies: List[str] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: List[str] = Field(default_factory=list)


class PlanSpec(BaseModel):
    """Structured execution plan specification."""

    goal: str = ""
    assumptions: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    steps: List[PlanStepSpec] = Field(default_factory=list)


class ReflectionSpec(BaseModel):
    """Structured reflection outcome."""

    quality_score: float = Field(default=0.7, ge=0.0, le=1.0)
    completeness: Literal["complete", "partial", "incomplete"] = "partial"
    risks: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    should_retry: bool = False
    retry_reason: Optional[str] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

TModel = TypeVar("TModel", bound=BaseModel)


def _utc_now() -> str:
    """Get current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _record_routing(
    state: DeepAgentState,
    supervisor: str,
    decision: str,
) -> DeepAgentState:
    """Record a routing decision in state."""
    path = list(state.get("supervisor_path", []))
    trace = list(state.get("routing_trace", []))

    path.append(supervisor)
    trace.append({
        "supervisor": supervisor,
        "decision": decision,
        "timestamp": _utc_now(),
    })

    return {
        **state,
        "supervisor_path": path,
        "routing_trace": trace,
    }


def _increment_iteration(state: DeepAgentState) -> DeepAgentState:
    """Increment iteration count and check limits."""
    count = state.get("iteration_count", 0) + 1
    max_iter = state.get("max_iterations", 10)

    if count >= max_iter:
        logger.warning(
            "Max iterations reached",
            count=count,
            max=max_iter,
        )

    return {
        **state,
        "iteration_count": count,
    }


def _extract_json_payload(content: str) -> str:
    """Extract the most likely JSON payload from a model response."""
    content = content.strip()
    if "```json" in content:
        return content.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in content:
        return content.split("```", 1)[1].split("```", 1)[0].strip()

    object_start = content.find("{")
    array_start = content.find("[")
    if object_start == -1 and array_start == -1:
        return content

    if object_start != -1 and (array_start == -1 or object_start < array_start):
        end = content.rfind("}")
        if end > object_start:
            return content[object_start : end + 1].strip()
    if array_start != -1:
        end = content.rfind("]")
        if end > array_start:
            return content[array_start : end + 1].strip()

    return content


def _parse_model(model: Type[TModel], content: str, default: TModel) -> Tuple[TModel, bool]:
    """Parse a JSON response into a Pydantic model with a fallback default."""
    payload = _extract_json_payload(content)
    try:
        return model.model_validate_json(payload), True
    except (ValidationError, ValueError):
        try:
            data = json.loads(payload)
            return model.model_validate(data), True
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            return default, False


def _parse_plan_spec(content: str, task: str) -> PlanSpec:
    """Parse a plan response into a PlanSpec, allowing legacy list output."""
    payload = _extract_json_payload(content)
    if payload.lstrip().startswith("["):
        try:
            steps_data = json.loads(payload)
            steps = [PlanStepSpec.model_validate(step) for step in steps_data]
            return PlanSpec(goal=task, steps=steps)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError):
            return PlanSpec(goal=task, steps=[PlanStepSpec(description=task)])

    spec, _ = _parse_model(PlanSpec, payload, PlanSpec(goal=task))
    if not spec.goal:
        spec.goal = task
    return spec


def _normalize_plan(task: str, spec: PlanSpec) -> Dict[str, Any]:
    """Build the workflow plan dict from a structured PlanSpec."""
    steps = spec.steps or [PlanStepSpec(description=task)]
    plan: Dict[str, Any] = {
        "goal": spec.goal or task,
        "assumptions": spec.assumptions,
        "constraints": spec.constraints,
        "success_criteria": spec.success_criteria,
        "steps": [],
        "created_at": _utc_now(),
        "status": "approved",
    }

    for i, step in enumerate(steps):
        description = step.description or f"Step {i + 1}"
        plan["steps"].append({
            "step_id": f"step_{i + 1}",
            "description": description,
            "agent_type": step.agent_type or "executor",
            "dependencies": step.dependencies,
            "inputs": step.inputs,
            "expected_outputs": step.expected_outputs,
            "status": "pending",
        })

    return plan


def _heuristic_reflection(content: str) -> ReflectionSpec:
    """Fallback reflection parsing based on simple heuristics."""
    quality_score = 0.7
    should_retry = False
    completeness: Literal["complete", "partial", "incomplete"] = "partial"

    lowered = content.lower()
    if "complete" in lowered and "incomplete" not in lowered:
        completeness = "complete"
        quality_score = 0.9
    elif "incomplete" in lowered:
        completeness = "incomplete"
        quality_score = 0.4
        should_retry = True

    if "retry" in lowered or "again" in lowered:
        should_retry = True

    return ReflectionSpec(
        quality_score=quality_score,
        completeness=completeness,
        should_retry=should_retry,
    )


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================

async def supervisor_node(state: DeepAgentState) -> DeepAgentState:
    """Supervisor node that analyzes tasks and routes to sub-agents.

    Args:
        state: Current workflow state

    Returns:
        Updated state with routing decision
    """
    logger.info("Supervisor analyzing task", task=state.get("task", "")[:100])

    llm = create_llm(model="claude-sonnet-4-5-20250514", temperature=0.0)

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {state.get('task', '')}"),
    ]

    # Add context if available
    context = state.get("context", {})
    if context:
        messages.append(HumanMessage(content=f"Context: {json.dumps(context, indent=2)}"))

    try:
        response = await llm.ainvoke(messages)
        content = response.content

        decision, parsed = _parse_model(
            SupervisorDecision,
            content,
            SupervisorDecision(),
        )
        agents = decision.agents or ["planner"]
        analysis = decision.analysis or (content if not parsed else "")

        # Determine next action
        next_action = agents[0] if agents else "planning"

        # Map agent names to phases
        phase_map = {
            "planner": "planning",
            "researcher": "retrieval",
            "coder": "execution",
            "reviewer": "reflection",
            "executor": "execution",
            "iac_architect": "execution",
            "container_workflow": "execution",
            "team_accelerator": "execution",
        }

        routing_decision = phase_map.get(next_action, "planning")

        updated_state = {
            **state,
            "phase": AgentPhase.PLANNING,
            "routing_decision": routing_decision,
            "active_agents": agents,
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Supervisor analysis: {analysis}\nRouting to: {agents}")
            ],
        }

        return _record_routing(updated_state, "supervisor", routing_decision)

    except Exception as e:
        logger.error("Supervisor error", error=str(e))
        return {
            **state,
            "phase": AgentPhase.ERROR,
            "errors": state.get("errors", []) + [f"Supervisor error: {str(e)}"],
            "routing_decision": "complete",
        }


async def planning_node(state: DeepAgentState) -> DeepAgentState:
    """Planning node that creates execution plans.

    Args:
        state: Current workflow state

    Returns:
        Updated state with execution plan
    """
    logger.info("Planning node creating execution plan")

    llm = create_llm(model="claude-sonnet-4-5-20250514", temperature=0.0)

    task = state.get("task", "")
    context = state.get("context", {})
    retrieved = state.get("retrieved_context", "")

    prompt_content = f"""Create an execution plan for the following task:

Task: {task}

Context: {json.dumps(context, indent=2)}

Retrieved Knowledge: {retrieved[:2000] if retrieved else "None"}

Provide a detailed plan with specific steps. Each step should be actionable.
"""

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content

        plan_spec = _parse_plan_spec(content, task)
        plan = _normalize_plan(task, plan_spec)

        return {
            **state,
            "phase": AgentPhase.RETRIEVAL,
            "plan": plan,
            "current_step_index": 0,
            "routing_decision": "retrieval",
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Created plan with {len(plan['steps'])} steps")
            ],
        }

    except Exception as e:
        logger.error("Planning error", error=str(e))
        return {
            **state,
            "phase": AgentPhase.ERROR,
            "errors": state.get("errors", []) + [f"Planning error: {str(e)}"],
            "routing_decision": "complete",
        }


async def rag_retrieval_node(state: DeepAgentState) -> DeepAgentState:
    """RAG retrieval node that fetches relevant knowledge.

    Args:
        state: Current workflow state

    Returns:
        Updated state with retrieved context
    """
    logger.info("RAG retrieval node fetching knowledge")

    task = state.get("task", "")
    plan = state.get("plan", {})

    # Build retrieval query from task and plan
    queries = [task]
    if plan and plan.get("steps"):
        for step in plan["steps"][:3]:  # Top 3 steps
            queries.append(step.get("description", ""))

    # In production, this would call the actual RAG system
    # For now, we'll simulate retrieval
    retrieval_results = []
    retrieved_context = ""

    try:
        # Simulate RAG retrieval (replace with actual RAG call)
        for query in queries:
            retrieval_results.append({
                "query": query,
                "documents": [
                    {"content": f"Relevant information for: {query[:50]}", "score": 0.85}
                ],
            })

        # Combine retrieved context
        retrieved_context = "\n\n".join([
            f"Query: {r['query']}\nResult: {r['documents'][0]['content']}"
            for r in retrieval_results
        ])

        return {
            **state,
            "phase": AgentPhase.REASONING,
            "retrieval_results": retrieval_results,
            "retrieved_context": retrieved_context,
            "routing_decision": "reasoning",
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Retrieved {len(retrieval_results)} knowledge chunks")
            ],
        }

    except Exception as e:
        logger.error("Retrieval error", error=str(e))
        # Continue without retrieval on error
        return {
            **state,
            "phase": AgentPhase.REASONING,
            "retrieval_results": [],
            "retrieved_context": "",
            "routing_decision": "reasoning",
        }


async def react_agent_node(
    state: DeepAgentState,
    tools: Optional[List[BaseTool]] = None,
) -> DeepAgentState:
    """ReAct agent node for reasoning and tool use.

    Args:
        state: Current workflow state
        tools: Available tools for the agent

    Returns:
        Updated state with reasoning results
    """
    logger.info("ReAct agent reasoning")

    llm = create_llm(model="claude-sonnet-4-5-20250514", temperature=0.0)

    task = state.get("task", "")
    plan = state.get("plan", {})
    retrieved = state.get("retrieved_context", "")
    current_step = state.get("current_step_index", 0)

    # Get current step from plan
    step_description = task
    if plan and plan.get("steps") and current_step < len(plan["steps"]):
        step_description = plan["steps"][current_step].get("description", task)

    prompt_content = f"""Execute the following step using ReAct reasoning:

Current Step: {step_description}

Overall Task: {task}

Retrieved Context:
{retrieved[:2000] if retrieved else "No additional context"}

Use the Thought-Action-Observation pattern to complete this step.
"""

    messages = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        *state.get("messages", [])[-5:],  # Include recent messages
        HumanMessage(content=prompt_content),
    ]

    try:
        if tools:
            # Use prebuilt ReAct agent with tools
            react_agent = create_react_agent(llm, tools)
            result = await react_agent.ainvoke({
                "messages": messages,
            })
            response_content = result["messages"][-1].content if result["messages"] else ""
        else:
            # Simple reasoning without tools
            response = await llm.ainvoke(messages)
            response_content = response.content

        # Update step status if plan exists
        updated_plan = plan
        if plan and plan.get("steps") and current_step < len(plan["steps"]):
            updated_plan = dict(plan)
            updated_plan["steps"] = list(plan["steps"])
            updated_plan["steps"][current_step] = {
                **plan["steps"][current_step],
                "status": "completed",
                "result": {"output": response_content[:500]},
            }

        return {
            **state,
            "phase": AgentPhase.EXECUTION,
            "plan": updated_plan,
            "routing_decision": "execution",
            "messages": state.get("messages", []) + [
                AIMessage(content=response_content)
            ],
            "subagent_results": {
                **state.get("subagent_results", {}),
                f"step_{current_step}": {
                    "status": "success",
                    "output": response_content,
                },
            },
        }

    except Exception as e:
        logger.error("ReAct agent error", error=str(e))
        return {
            **state,
            "phase": AgentPhase.REFLECTION,
            "errors": state.get("errors", []) + [f"ReAct error: {str(e)}"],
            "routing_decision": "reflection",
        }


async def execution_node(state: DeepAgentState) -> DeepAgentState:
    """Execution node that runs planned steps.

    Args:
        state: Current workflow state

    Returns:
        Updated state with execution results
    """
    logger.info("Execution node processing")

    plan = state.get("plan", {})
    current_step = state.get("current_step_index", 0)
    steps = plan.get("steps", []) if plan else []

    # Check if all steps completed
    if current_step >= len(steps):
        return {
            **state,
            "phase": AgentPhase.REFLECTION,
            "routing_decision": "reflection",
            "messages": state.get("messages", []) + [
                AIMessage(content="All plan steps completed. Moving to reflection.")
            ],
        }

    # Execute current step
    step = steps[current_step]
    logger.info(
        "Executing step",
        step_id=step.get("step_id"),
        description=step.get("description", "")[:50],
    )

    # Delegate to ReAct for actual execution
    state_with_increment = _increment_iteration(state)

    return {
        **state_with_increment,
        "current_step_index": current_step + 1,
        "routing_decision": "reasoning",
        "messages": state.get("messages", []) + [
            AIMessage(content=f"Executing step {current_step + 1}: {step.get('description', '')[:100]}")
        ],
    }


async def reflection_node(state: DeepAgentState) -> DeepAgentState:
    """Reflection node that evaluates work quality.

    Args:
        state: Current workflow state

    Returns:
        Updated state with reflection results
    """
    logger.info("Reflection node evaluating outputs")

    llm = create_llm(model="claude-sonnet-4-5-20250514", temperature=0.0)

    task = state.get("task", "")
    subagent_results = state.get("subagent_results", {})
    errors = state.get("errors", [])
    plan = state.get("plan", {})

    # Summarize results for reflection
    results_summary = json.dumps(subagent_results, indent=2, default=str)[:2000]

    prompt_content = f"""Evaluate the following work:

Original Task: {task}

Execution Results:
{results_summary}

Errors Encountered: {errors if errors else "None"}

Provide a quality assessment and determine if the task is complete.
"""

    messages = [
        SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content

        default_reflection = _heuristic_reflection(content)
        reflection_spec, _ = _parse_model(ReflectionSpec, content, default_reflection)

        reflection = {
            "quality_score": reflection_spec.quality_score,
            "completeness": reflection_spec.completeness,
            "should_retry": reflection_spec.should_retry,
            "retry_reason": reflection_spec.retry_reason,
            "risks": reflection_spec.risks,
            "improvements": reflection_spec.improvements,
            "feedback": content[:500],
            "timestamp": _utc_now(),
        }

        # Determine next action
        if reflection_spec.should_retry and state.get("iteration_count", 0) < state.get("max_iterations", 10):
            routing_decision = "planning"
            next_phase = AgentPhase.PLANNING
        else:
            routing_decision = "complete"
            next_phase = AgentPhase.COMPLETE

        return {
            **state,
            "phase": next_phase,
            "reflection": reflection,
            "quality_scores": {
                **state.get("quality_scores", {}),
                f"iteration_{state.get('iteration_count', 0)}": reflection_spec.quality_score,
            },
            "routing_decision": routing_decision,
            "messages": state.get("messages", []) + [
                AIMessage(
                    content=(
                        "Reflection: "
                        f"Quality={reflection_spec.quality_score:.2f}, "
                        f"Completeness={reflection_spec.completeness}"
                    )
                )
            ],
        }

    except Exception as e:
        logger.error("Reflection error", error=str(e))
        return {
            **state,
            "phase": AgentPhase.COMPLETE,
            "routing_decision": "complete",
            "errors": state.get("errors", []) + [f"Reflection error: {str(e)}"],
        }


async def output_node(state: DeepAgentState) -> DeepAgentState:
    """Final output node that aggregates results.

    Args:
        state: Current workflow state

    Returns:
        Updated state with final output
    """
    logger.info("Output node generating final response")

    llm = create_llm(model="claude-sonnet-4-5-20250514", temperature=0.0)

    task = state.get("task", "")
    subagent_results = state.get("subagent_results", {})
    reflection = state.get("reflection", {})
    output_format = state.get("output_format", "markdown")

    # Gather all AI messages for summary
    messages_content = []
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage):
            messages_content.append(msg.content)

    prompt_content = f"""Generate a final response for the following completed task:

Task: {task}

Work Summary:
{chr(10).join(messages_content[-5:])}

Format the response as {output_format}.
Be concise but comprehensive. Include key findings, actions taken, and results.
"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a helpful assistant generating a final summary."),
            HumanMessage(content=prompt_content),
        ])

        final_output = response.content

        return {
            **state,
            "phase": AgentPhase.COMPLETE,
            "final_output": final_output,
            "routing_decision": "end",
            "messages": state.get("messages", []) + [
                AIMessage(content=final_output)
            ],
        }

    except Exception as e:
        logger.error("Output generation error", error=str(e))
        # Fallback to simple summary
        fallback_output = f"Task completed: {task}\n\nResults: {json.dumps(subagent_results, indent=2, default=str)[:1000]}"

        return {
            **state,
            "phase": AgentPhase.COMPLETE,
            "final_output": fallback_output,
            "routing_decision": "end",
        }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_supervisor(
    state: DeepAgentState,
) -> Literal["planning", "retrieval", "execution", "reflection", "complete"]:
    """Route after supervisor node."""
    decision = state.get("routing_decision", "planning")
    if decision in ["planning", "retrieval", "execution", "reflection", "complete"]:
        return decision
    return "planning"


def route_after_planning(
    state: DeepAgentState,
) -> Literal["retrieval", "reasoning", "complete"]:
    """Route after planning node."""
    decision = state.get("routing_decision", "retrieval")
    if decision in ["retrieval", "reasoning", "complete"]:
        return decision
    return "retrieval"


def route_after_retrieval(
    state: DeepAgentState,
) -> Literal["reasoning", "execution", "complete"]:
    """Route after retrieval node."""
    decision = state.get("routing_decision", "reasoning")
    if decision in ["reasoning", "execution", "complete"]:
        return decision
    return "reasoning"


def route_after_reasoning(
    state: DeepAgentState,
) -> Literal["execution", "reflection", "complete"]:
    """Route after reasoning node."""
    decision = state.get("routing_decision", "execution")
    if decision in ["execution", "reflection", "complete"]:
        return decision
    return "execution"


def route_after_execution(
    state: DeepAgentState,
) -> Literal["reasoning", "reflection", "complete"]:
    """Route after execution node."""
    plan = state.get("plan", {})
    current_step = state.get("current_step_index", 0)
    steps = plan.get("steps", []) if plan else []

    # Check iteration limit
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return "reflection"

    # More steps to execute
    if current_step < len(steps):
        return "reasoning"

    return "reflection"


def route_after_reflection(
    state: DeepAgentState,
) -> Literal["planning", "complete"]:
    """Route after reflection node."""
    reflection = state.get("reflection", {})
    should_retry = reflection.get("should_retry", False)

    if should_retry and state.get("iteration_count", 0) < state.get("max_iterations", 10):
        return "planning"

    return "complete"
