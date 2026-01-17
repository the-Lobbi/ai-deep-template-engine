"""LangGraph Deep Agent Graph Definition.

This module defines the complete LangGraph workflow for the deep agent system.
It connects all nodes with appropriate routing logic and provides factory functions.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Type, Union

import structlog
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .state import (
    AgentPhase,
    DeepAgentState,
    create_initial_state,
    get_state_summary,
)
from .nodes import (
    supervisor_node,
    planning_node,
    rag_retrieval_node,
    react_agent_node,
    execution_node,
    reflection_node,
    output_node,
    route_after_supervisor,
    route_after_planning,
    route_after_retrieval,
    route_after_reasoning,
    route_after_execution,
    route_after_reflection,
)

logger = structlog.get_logger(__name__)


class DeepAgentGraph:
    """Wrapper class for the Deep Agent LangGraph workflow.

    This class provides a high-level interface for:
    - Building and compiling the graph
    - Invoking the agent with tasks
    - Streaming execution results
    - Managing checkpoints and state

    Example:
        >>> graph = DeepAgentGraph()
        >>> result = await graph.invoke("Analyze the codebase and suggest improvements")
        >>> print(result["final_output"])
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        max_iterations: int = 10,
    ):
        """Initialize the Deep Agent Graph.

        Args:
            tools: List of tools available to the agent
            checkpointer: State checkpointer (defaults to MemorySaver)
            max_iterations: Maximum workflow iterations
        """
        self.tools = tools or []
        self.checkpointer = checkpointer or MemorySaver()
        self.max_iterations = max_iterations

        self._graph: Optional[CompiledStateGraph] = None
        self._build_graph()

        logger.info(
            "DeepAgentGraph initialized",
            num_tools=len(self.tools),
            max_iterations=max_iterations,
        )

    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""
        workflow = StateGraph(DeepAgentState)

        # Add nodes
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_node("planning", planning_node)
        workflow.add_node("retrieval", rag_retrieval_node)
        workflow.add_node("reasoning", self._create_reasoning_node())
        workflow.add_node("execution", execution_node)
        workflow.add_node("reflection", reflection_node)
        workflow.add_node("output", output_node)

        # Set entry point
        workflow.add_edge(START, "supervisor")

        # Add conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {
                "planning": "planning",
                "retrieval": "retrieval",
                "execution": "execution",
                "reflection": "reflection",
                "complete": "output",
            },
        )

        # Add edges from planning
        workflow.add_conditional_edges(
            "planning",
            route_after_planning,
            {
                "retrieval": "retrieval",
                "reasoning": "reasoning",
                "complete": "output",
            },
        )

        # Add edges from retrieval
        workflow.add_conditional_edges(
            "retrieval",
            route_after_retrieval,
            {
                "reasoning": "reasoning",
                "execution": "execution",
                "complete": "output",
            },
        )

        # Add edges from reasoning
        workflow.add_conditional_edges(
            "reasoning",
            route_after_reasoning,
            {
                "execution": "execution",
                "reflection": "reflection",
                "complete": "output",
            },
        )

        # Add edges from execution
        workflow.add_conditional_edges(
            "execution",
            route_after_execution,
            {
                "reasoning": "reasoning",
                "reflection": "reflection",
                "complete": "output",
            },
        )

        # Add edges from reflection
        workflow.add_conditional_edges(
            "reflection",
            route_after_reflection,
            {
                "planning": "planning",
                "complete": "output",
            },
        )

        # Output goes to END
        workflow.add_edge("output", END)

        # Compile with checkpointer
        self._graph = workflow.compile(checkpointer=self.checkpointer)

        logger.debug("Graph compiled successfully")

    def _create_reasoning_node(self) -> Callable:
        """Create a reasoning node with tools bound."""
        tools = self.tools

        async def reasoning_with_tools(state: DeepAgentState) -> DeepAgentState:
            return await react_agent_node(state, tools=tools if tools else None)

        return reasoning_with_tools

    async def invoke(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DeepAgentState:
        """Invoke the agent with a task.

        Args:
            task: The task description
            context: Optional context dictionary
            thread_id: Thread ID for state persistence
            **kwargs: Additional arguments

        Returns:
            Final state after execution
        """
        if self._graph is None:
            raise RuntimeError("Graph not initialized")

        initial_state = create_initial_state(
            task=task,
            context=context,
            max_iterations=self.max_iterations,
        )

        config = {"configurable": {"thread_id": thread_id or "default"}}
        if kwargs:
            config.update(kwargs)

        logger.info("Invoking deep agent", task=task[:100], thread_id=thread_id)

        result = await self._graph.ainvoke(initial_state, config)

        logger.info(
            "Agent execution complete",
            summary=get_state_summary(result),
        )

        return result

    async def stream(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
    ):
        """Stream agent execution results.

        Args:
            task: The task description
            context: Optional context dictionary
            thread_id: Thread ID for state persistence

        Yields:
            State updates as the agent executes
        """
        if self._graph is None:
            raise RuntimeError("Graph not initialized")

        initial_state = create_initial_state(
            task=task,
            context=context,
            max_iterations=self.max_iterations,
        )

        config = {"configurable": {"thread_id": thread_id or "default"}}

        logger.info("Starting stream", task=task[:100])

        async for event in self._graph.astream(initial_state, config):
            yield event

    def get_graph(self) -> CompiledStateGraph:
        """Get the compiled graph instance."""
        if self._graph is None:
            raise RuntimeError("Graph not initialized")
        return self._graph

    def get_state_schema(self) -> Type[DeepAgentState]:
        """Get the state schema type."""
        return DeepAgentState


def create_deep_agent_graph(
    tools: Optional[List[BaseTool]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    max_iterations: int = 10,
) -> DeepAgentGraph:
    """Factory function to create a Deep Agent Graph.

    Args:
        tools: List of tools available to the agent
        checkpointer: State checkpointer
        max_iterations: Maximum workflow iterations

    Returns:
        Configured DeepAgentGraph instance

    Example:
        >>> graph = create_deep_agent_graph(tools=[search_tool, code_tool])
        >>> result = await graph.invoke("Build a REST API")
    """
    return DeepAgentGraph(
        tools=tools,
        checkpointer=checkpointer,
        max_iterations=max_iterations,
    )


# =============================================================================
# SPECIALIZED GRAPH VARIANTS
# =============================================================================

def create_planning_only_graph() -> CompiledStateGraph:
    """Create a graph that only does planning (no execution).

    Useful for getting a plan before execution.
    """
    workflow = StateGraph(DeepAgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("retrieval", rag_retrieval_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_edge("supervisor", "planning")
    workflow.add_edge("planning", "retrieval")
    workflow.add_edge("retrieval", END)

    return workflow.compile()


def create_react_only_graph(
    tools: Optional[List[BaseTool]] = None,
) -> CompiledStateGraph:
    """Create a simple ReAct-only graph.

    Single-step reasoning without planning/reflection loop.
    """
    workflow = StateGraph(DeepAgentState)

    async def react_node(state: DeepAgentState) -> DeepAgentState:
        result = await react_agent_node(state, tools=tools)
        return {
            **result,
            "routing_decision": "complete",
            "phase": AgentPhase.COMPLETE,
        }

    workflow.add_node("react", react_node)
    workflow.add_node("output", output_node)

    workflow.add_edge(START, "react")
    workflow.add_edge("react", "output")
    workflow.add_edge("output", END)

    return workflow.compile()


# =============================================================================
# MULTI-AGENT SUPERVISOR GRAPH
# =============================================================================

def create_multi_agent_supervisor_graph(
    agent_configs: Dict[str, Dict[str, Any]],
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> CompiledStateGraph:
    """Create a multi-agent supervisor graph.

    This creates a graph where a supervisor routes to multiple
    specialized sub-agents based on task analysis.

    Args:
        agent_configs: Dictionary mapping agent names to their configs
        checkpointer: State checkpointer

    Returns:
        Compiled multi-agent graph
    """
    workflow = StateGraph(DeepAgentState)

    # Add supervisor
    workflow.add_node("supervisor", supervisor_node)

    # Add nodes for each configured agent
    for agent_name, config in agent_configs.items():
        agent_type = config.get("type", "react")

        if agent_type == "react":
            tools = config.get("tools", [])

            async def agent_node(state: DeepAgentState, t=tools) -> DeepAgentState:
                return await react_agent_node(state, tools=t)

            workflow.add_node(agent_name, agent_node)
        elif agent_type == "planning":
            workflow.add_node(agent_name, planning_node)
        elif agent_type == "reflection":
            workflow.add_node(agent_name, reflection_node)

    # Add output node
    workflow.add_node("output", output_node)

    # Entry point
    workflow.add_edge(START, "supervisor")

    # Build routing map from supervisor to agents
    routing_map = {name: name for name in agent_configs.keys()}
    routing_map["complete"] = "output"

    def supervisor_router(state: DeepAgentState) -> str:
        active = state.get("active_agents", [])
        if active:
            first_agent = active[0]
            if first_agent in agent_configs:
                return first_agent
        return "output"

    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        routing_map,
    )

    # Route all agents back to supervisor or output
    for agent_name in agent_configs.keys():

        def agent_router(state: DeepAgentState) -> str:
            phase = state.get("phase", AgentPhase.INIT)
            if phase == AgentPhase.COMPLETE:
                return "output"
            # Could route to another agent or back to supervisor
            return "supervisor"

        workflow.add_conditional_edges(
            agent_name,
            agent_router,
            {"supervisor": "supervisor", "output": "output"},
        )

    workflow.add_edge("output", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DeepAgentGraph",
    "create_deep_agent_graph",
    "create_planning_only_graph",
    "create_react_only_graph",
    "create_multi_agent_supervisor_graph",
]
