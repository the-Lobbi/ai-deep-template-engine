"""
Properly connected LangGraph Studio visualization with all nodes linked.
This version ensures every node has clear incoming and outgoing edges.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage


class AgentState(TypedDict):
    """State for the Deep Agent workflow"""
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    repo_url: str
    current_phase: str

    # Analysis results
    languages: list[str]
    frameworks: list[str]
    has_docker: bool

    # Design results
    pipeline_stages: list[str]
    connectors: list[str]

    # Creation results
    resources: list[str]
    pipeline_id: str

    # Deployment results
    execution_id: str
    status: str

    # Routing
    needs_approval: bool


# =============================================================================
# ORCHESTRATOR NODES
# =============================================================================

def start_workflow(state: AgentState) -> dict:
    """Entry: Receive task and plan approach"""
    return {
        "current_phase": "planning",
        "messages": [AIMessage(content="🎯 Starting Deep Agent workflow")]
    }


def plan_execution(state: AgentState) -> dict:
    """Plan: Break down task into phases"""
    return {
        "current_phase": "ready_for_analysis",
        "messages": [AIMessage(content="📋 Planning complete - routing to Repo Analyst")]
    }


# =============================================================================
# REPO-ANALYST SUBAGENT (7 nodes)
# =============================================================================

def analyst_start(state: AgentState) -> dict:
    """Repo Analyst: Initialize"""
    return {
        "messages": [AIMessage(content="🔬 [REPO-ANALYST] Starting repository analysis")]
    }


def clone_repo(state: AgentState) -> dict:
    """Clone the repository"""
    return {
        "messages": [AIMessage(content=f"📥 [REPO-ANALYST] Cloning {state.get('repo_url', 'repository')}")]
    }


def run_repomix(state: AgentState) -> dict:
    """Extract code patterns"""
    return {
        "messages": [AIMessage(content="🔍 [REPO-ANALYST] Running Repomix for pattern extraction")]
    }


def detect_languages(state: AgentState) -> dict:
    """Detect programming languages"""
    return {
        "languages": ["python", "typescript", "go"],
        "messages": [AIMessage(content="🐍 [REPO-ANALYST] Detected: Python, TypeScript, Go")]
    }


def detect_frameworks(state: AgentState) -> dict:
    """Detect frameworks"""
    return {
        "frameworks": ["fastapi", "react", "kubernetes"],
        "has_docker": True,
        "messages": [AIMessage(content="⚡ [REPO-ANALYST] Found: FastAPI, React, Kubernetes")]
    }


def find_configs(state: AgentState) -> dict:
    """Find configuration files"""
    return {
        "messages": [AIMessage(content="📄 [REPO-ANALYST] Found: Dockerfile, k8s/, .github/")]
    }


def analyst_complete(state: AgentState) -> dict:
    """Repo Analyst: Exit"""
    return {
        "current_phase": "analysis_complete",
        "messages": [AIMessage(content="✅ [REPO-ANALYST] Analysis complete - handoff to Harness Expert")]
    }


# =============================================================================
# HARNESS-EXPERT SUBAGENT - DESIGN (5 nodes)
# =============================================================================

def expert_design_start(state: AgentState) -> dict:
    """Harness Expert: Start design phase"""
    return {
        "messages": [AIMessage(content="⚙️ [HARNESS-EXPERT] Starting pipeline design")]
    }


def design_pipeline(state: AgentState) -> dict:
    """Design CI/CD pipeline"""
    return {
        "pipeline_stages": ["build", "test", "scan", "deploy"],
        "messages": [AIMessage(content="🏗️ [HARNESS-EXPERT] Designed 4-stage pipeline")]
    }


def design_connectors(state: AgentState) -> dict:
    """Design required connectors"""
    return {
        "connectors": ["github", "docker-hub", "k8s-cluster"],
        "messages": [AIMessage(content="🔌 [HARNESS-EXPERT] Planned 3 connectors")]
    }


def design_environments(state: AgentState) -> dict:
    """Design environments"""
    return {
        "messages": [AIMessage(content="🌍 [HARNESS-EXPERT] Designed: dev, staging, prod")]
    }


def expert_design_complete(state: AgentState) -> dict:
    """Harness Expert: Design phase complete"""
    return {
        "current_phase": "design_complete",
        "needs_approval": True,
        "messages": [AIMessage(content="✅ [HARNESS-EXPERT] Design complete - requesting approval")]
    }


# =============================================================================
# APPROVAL GATE
# =============================================================================

def approval_gate(state: AgentState) -> dict:
    """Human approval checkpoint"""
    return {
        "current_phase": "approved",
        "messages": [AIMessage(content="✋ [APPROVAL] Review design before creation (auto-approved in demo)")]
    }


# =============================================================================
# HARNESS-EXPERT SUBAGENT - CREATE (6 nodes)
# =============================================================================

def expert_create_start(state: AgentState) -> dict:
    """Harness Expert: Start creation phase"""
    return {
        "resources": [],
        "messages": [AIMessage(content="⚙️ [HARNESS-EXPERT] Starting resource creation")]
    }


def create_secrets(state: AgentState) -> dict:
    """Create secrets"""
    resources = state.get("resources", [])
    return {
        "resources": resources + ["secret:github-pat", "secret:docker-creds"],
        "messages": [AIMessage(content="🔐 [HARNESS-EXPERT] Created 2 secrets")]
    }


def create_connectors(state: AgentState) -> dict:
    """Create connectors"""
    resources = state.get("resources", [])
    return {
        "resources": resources + ["connector:github", "connector:docker", "connector:k8s"],
        "messages": [AIMessage(content="🔌 [HARNESS-EXPERT] Created 3 connectors")]
    }


def create_environments(state: AgentState) -> dict:
    """Create environments"""
    resources = state.get("resources", [])
    return {
        "resources": resources + ["env:dev", "env:staging", "env:prod"],
        "messages": [AIMessage(content="🌍 [HARNESS-EXPERT] Created 3 environments")]
    }


def create_service(state: AgentState) -> dict:
    """Create service definition"""
    resources = state.get("resources", [])
    return {
        "resources": resources + ["service:my-app"],
        "messages": [AIMessage(content="📦 [HARNESS-EXPERT] Created service definition")]
    }


def create_pipeline(state: AgentState) -> dict:
    """Create pipeline"""
    resources = state.get("resources", [])
    return {
        "resources": resources + ["pipeline:my-app-cicd"],
        "pipeline_id": "my-app-cicd",
        "messages": [AIMessage(content="🚀 [HARNESS-EXPERT] Created pipeline: my-app-cicd")]
    }


def expert_create_complete(state: AgentState) -> dict:
    """Harness Expert: Creation complete"""
    return {
        "current_phase": "resources_created",
        "messages": [AIMessage(content="✅ [HARNESS-EXPERT] All resources created - handoff to Deployer")]
    }


# =============================================================================
# DEPLOYER SUBAGENT (5 nodes)
# =============================================================================

def deployer_start(state: AgentState) -> dict:
    """Deployer: Initialize"""
    return {
        "messages": [AIMessage(content="🚀 [DEPLOYER] Starting deployment verification")]
    }


def trigger_pipeline(state: AgentState) -> dict:
    """Trigger execution"""
    return {
        "execution_id": "exec-abc123",
        "messages": [AIMessage(content=f"▶️ [DEPLOYER] Triggered: {state.get('pipeline_id', 'pipeline')}")]
    }


def monitor_deployment(state: AgentState) -> dict:
    """Monitor progress"""
    return {
        "messages": [AIMessage(content="⏳ [DEPLOYER] Monitoring: Build ✓ → Test ✓ → Deploy...")]
    }


def verify_health(state: AgentState) -> dict:
    """Verify deployment"""
    return {
        "status": "success",
        "messages": [AIMessage(content="🏥 [DEPLOYER] Health check passed - service is live")]
    }


def deployer_complete(state: AgentState) -> dict:
    """Deployer: Exit"""
    return {
        "current_phase": "deployment_verified",
        "messages": [AIMessage(content="✅ [DEPLOYER] Verification complete")]
    }


# =============================================================================
# FINAL NODES
# =============================================================================

def finalize(state: AgentState) -> dict:
    """Final validation and summary"""
    return {
        "current_phase": "complete",
        "messages": [AIMessage(content="🎉 Workflow complete! All phases successful.")]
    }


# =============================================================================
# BUILD GRAPH - FULLY CONNECTED
# =============================================================================

graph = StateGraph(AgentState)

# Add all nodes
nodes = {
    # Orchestrator
    "start": start_workflow,
    "plan": plan_execution,
    "approval": approval_gate,
    "finalize": finalize,

    # Repo-Analyst (7)
    "analyst_start": analyst_start,
    "clone_repo": clone_repo,
    "run_repomix": run_repomix,
    "detect_languages": detect_languages,
    "detect_frameworks": detect_frameworks,
    "find_configs": find_configs,
    "analyst_complete": analyst_complete,

    # Harness-Expert Design (5)
    "expert_design_start": expert_design_start,
    "design_pipeline": design_pipeline,
    "design_connectors": design_connectors,
    "design_environments": design_environments,
    "expert_design_complete": expert_design_complete,

    # Harness-Expert Create (6)
    "expert_create_start": expert_create_start,
    "create_secrets": create_secrets,
    "create_connectors": create_connectors,
    "create_environments": create_environments,
    "create_service": create_service,
    "create_pipeline": create_pipeline,
    "expert_create_complete": expert_create_complete,

    # Deployer (5)
    "deployer_start": deployer_start,
    "trigger_pipeline": trigger_pipeline,
    "monitor_deployment": monitor_deployment,
    "verify_health": verify_health,
    "deployer_complete": deployer_complete,
}

for name, func in nodes.items():
    graph.add_node(name, func)

# Add edges - FULLY CONNECTED FLOW
# Main flow
graph.add_edge(START, "start")
graph.add_edge("start", "plan")
graph.add_edge("plan", "analyst_start")

# Repo-Analyst flow (linear)
graph.add_edge("analyst_start", "clone_repo")
graph.add_edge("clone_repo", "run_repomix")
graph.add_edge("run_repomix", "detect_languages")
graph.add_edge("detect_languages", "detect_frameworks")
graph.add_edge("detect_frameworks", "find_configs")
graph.add_edge("find_configs", "analyst_complete")
graph.add_edge("analyst_complete", "expert_design_start")

# Harness-Expert Design flow (linear)
graph.add_edge("expert_design_start", "design_pipeline")
graph.add_edge("design_pipeline", "design_connectors")
graph.add_edge("design_connectors", "design_environments")
graph.add_edge("design_environments", "expert_design_complete")
graph.add_edge("expert_design_complete", "approval")

# Approval gate
graph.add_edge("approval", "expert_create_start")

# Harness-Expert Create flow (linear)
graph.add_edge("expert_create_start", "create_secrets")
graph.add_edge("create_secrets", "create_connectors")
graph.add_edge("create_connectors", "create_environments")
graph.add_edge("create_environments", "create_service")
graph.add_edge("create_service", "create_pipeline")
graph.add_edge("create_pipeline", "expert_create_complete")
graph.add_edge("expert_create_complete", "deployer_start")

# Deployer flow (linear)
graph.add_edge("deployer_start", "trigger_pipeline")
graph.add_edge("trigger_pipeline", "monitor_deployment")
graph.add_edge("monitor_deployment", "verify_health")
graph.add_edge("verify_health", "deployer_complete")
graph.add_edge("deployer_complete", "finalize")

# Final
graph.add_edge("finalize", END)

# Compile
app = graph.compile()
