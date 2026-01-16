"""
LangGraph Studio compatible graph definition.
This file provides a simple interface for LangGraph Studio visualization.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """State for the Harness Deep Agent workflow"""
    messages: Annotated[list[BaseMessage], add_messages]
    current_phase: str
    repo_url: str
    harness_config: dict
    status: str


def analyze_repo(state: AgentState) -> dict:
    """Phase 1: Repository Analysis"""
    return {
        "current_phase": "analysis",
        "status": "analyzing repository"
    }


def design_harness(state: AgentState) -> dict:
    """Phase 2: Design Harness Configuration"""
    return {
        "current_phase": "design",
        "status": "designing harness setup"
    }


def create_resources(state: AgentState) -> dict:
    """Phase 3: Create Harness Resources"""
    return {
        "current_phase": "create",
        "status": "creating harness resources"
    }


def deploy_verify(state: AgentState) -> dict:
    """Phase 4: Deploy and Verify"""
    return {
        "current_phase": "deploy",
        "status": "deploying and verifying"
    }


def route_phase(state: AgentState) -> str:
    """Route to next phase based on current state"""
    phase = state.get("current_phase", "")

    if phase == "analysis":
        return "design"
    elif phase == "design":
        return "create"
    elif phase == "create":
        return "deploy"
    else:
        return END


# Build the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("analyze", analyze_repo)
graph.add_node("design", design_harness)
graph.add_node("create", create_resources)
graph.add_node("deploy", deploy_verify)

# Add edges
graph.add_edge(START, "analyze")
graph.add_conditional_edges("analyze", route_phase)
graph.add_conditional_edges("design", route_phase)
graph.add_conditional_edges("create", route_phase)
graph.add_edge("deploy", END)

# Compile the graph
app = graph.compile()
