"""Deep Agent State Management.

This module defines the core state structures for the LangGraph-based deep agent system.
Follows LangGraph 0.2+ patterns with typed state and message handling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentPhase(str, Enum):
    """Current phase of the deep agent workflow."""

    INIT = "init"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    COMPLETE = "complete"
    ERROR = "error"


class TaskPriority(str, Enum):
    """Priority levels for tasks."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class SubAgentType(str, Enum):
    """Types of specialized sub-agents."""

    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"
    EXECUTOR = "executor"
    IAC_ARCHITECT = "iac_architect"
    CONTAINER_WORKFLOW = "container_workflow"
    TEAM_ACCELERATOR = "team_accelerator"


@dataclass
class MessageHistory:
    """Wrapper for conversation message history with utilities."""

    messages: List[BaseMessage] = field(default_factory=list)

    def add_human(self, content: str) -> None:
        """Add a human message."""
        self.messages.append(HumanMessage(content=content))

    def add_ai(self, content: str, tool_calls: Optional[List[Dict]] = None) -> None:
        """Add an AI message."""
        if tool_calls:
            self.messages.append(AIMessage(content=content, tool_calls=tool_calls))
        else:
            self.messages.append(AIMessage(content=content))

    def add_system(self, content: str) -> None:
        """Add a system message."""
        self.messages.append(SystemMessage(content=content))

    def add_tool(self, content: str, tool_call_id: str, name: str) -> None:
        """Add a tool result message."""
        self.messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))

    def get_last_n(self, n: int) -> List[BaseMessage]:
        """Get the last n messages."""
        return self.messages[-n:] if len(self.messages) >= n else self.messages.copy()

    def to_list(self) -> List[BaseMessage]:
        """Convert to list of messages."""
        return self.messages.copy()


@dataclass
class AgentContext:
    """Shared context for agent execution."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    task_description: str = ""
    task_priority: TaskPriority = TaskPriority.NORMAL
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    agent_type: SubAgentType
    dependencies: List[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "completed", "failed", "skipped"] = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExecutionPlan(BaseModel):
    """Complete execution plan with steps."""

    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str
    steps: List[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["draft", "approved", "executing", "completed", "failed"] = "draft"


class RetrievalResult(BaseModel):
    """Result from RAG retrieval."""

    query: str
    documents: List[Dict[str, Any]] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReflectionOutcome(BaseModel):
    """Outcome from reflection on agent outputs."""

    quality_score: float = Field(ge=0.0, le=1.0)
    completeness: Literal["complete", "partial", "incomplete"] = "partial"
    risks: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    should_retry: bool = False
    retry_reason: Optional[str] = None


class SubAgentResult(BaseModel):
    """Result from a sub-agent execution."""

    agent_type: SubAgentType
    agent_name: str
    task: str
    status: Literal["success", "partial", "failed"] = "success"
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeepAgentState(TypedDict):
    """Complete state for the Deep Agent workflow.

    This follows LangGraph 0.2+ patterns with typed state management.
    The state flows through the graph, accumulating results from each node.

    Attributes:
        messages: Conversation history with message accumulation
        phase: Current workflow phase
        task: Primary task description
        context: Shared execution context
        plan: Current execution plan
        retrieval_results: RAG retrieval results
        subagent_results: Results from sub-agent executions
        reflection: Latest reflection outcome
        routing_decision: Next node/agent to route to
        iteration_count: Number of iterations (for loop prevention)
        max_iterations: Maximum allowed iterations
        errors: List of errors encountered
        final_output: Final aggregated output
    """

    # Core message handling with LangGraph accumulation
    messages: Annotated[List[BaseMessage], add_messages]

    # Workflow control
    phase: AgentPhase
    task: str
    context: Dict[str, Any]

    # Planning
    plan: Optional[Dict[str, Any]]
    current_step_index: int

    # Retrieval
    retrieval_results: List[Dict[str, Any]]
    retrieved_context: str

    # Sub-agent execution
    subagent_results: Dict[str, Any]
    active_agents: List[str]

    # Reflection and quality
    reflection: Optional[Dict[str, Any]]
    quality_scores: Dict[str, float]

    # Routing and control flow
    routing_decision: str
    supervisor_path: List[str]
    routing_trace: List[Dict[str, str]]

    # Iteration control
    iteration_count: int
    max_iterations: int

    # Error handling
    errors: List[str]
    failure_history: List[Dict[str, Any]]

    # Output
    final_output: Optional[str]
    output_format: str


def create_initial_state(
    task: str,
    context: Optional[Dict[str, Any]] = None,
    max_iterations: int = 10,
) -> DeepAgentState:
    """Create initial state for a new deep agent workflow.

    Args:
        task: The primary task description
        context: Optional shared context
        max_iterations: Maximum iterations before stopping

    Returns:
        Initialized DeepAgentState
    """
    return DeepAgentState(
        messages=[HumanMessage(content=task)],
        phase=AgentPhase.INIT,
        task=task,
        context=context or {},
        plan=None,
        current_step_index=0,
        retrieval_results=[],
        retrieved_context="",
        subagent_results={},
        active_agents=[],
        reflection=None,
        quality_scores={},
        routing_decision="planning",
        supervisor_path=[],
        routing_trace=[],
        iteration_count=0,
        max_iterations=max_iterations,
        errors=[],
        failure_history=[],
        final_output=None,
        output_format="markdown",
    )


def get_state_summary(state: DeepAgentState) -> Dict[str, Any]:
    """Generate a summary of the current state for debugging/logging.

    Args:
        state: Current agent state

    Returns:
        Summary dictionary
    """
    return {
        "phase": state.get("phase", AgentPhase.INIT),
        "task": state.get("task", "")[:100],
        "message_count": len(state.get("messages", [])),
        "iteration": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 10),
        "has_plan": state.get("plan") is not None,
        "retrieval_count": len(state.get("retrieval_results", [])),
        "subagent_count": len(state.get("subagent_results", {})),
        "error_count": len(state.get("errors", [])),
        "has_output": state.get("final_output") is not None,
    }
