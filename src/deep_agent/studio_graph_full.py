"""
Full LangGraph Studio visualization showing the complete Deep Agent architecture
with subagents, tool routing, and decision points.
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class DeepAgentState(TypedDict):
    """State for the full Deep Agent workflow"""
    messages: Annotated[list[BaseMessage], add_messages]

    # Task info
    task: str
    repo_url: str
    harness_config: dict

    # Workflow control
    current_phase: str
    subagent_target: str

    # Analysis results
    repo_analysis: dict
    detected_languages: list[str]
    detected_frameworks: list[str]
    has_dockerfile: bool
    has_kubernetes: bool

    # Design results
    pipeline_design: dict
    required_connectors: list[str]
    required_secrets: list[str]
    environments: list[str]

    # Creation results
    created_resources: list[str]
    pipeline_id: str

    # Deployment results
    execution_id: str
    deployment_status: str

    # Control
    needs_approval: bool
    approved: bool
    errors: list[str]


# =============================================================================
# MAIN ORCHESTRATOR NODES
# =============================================================================

def plan_task(state: DeepAgentState) -> dict:
    """Initial planning: Break down task and determine approach"""
    return {
        "current_phase": "planning",
        "messages": [AIMessage(content="Planning automation workflow and determining required subagents")]
    }


def route_to_subagent(state: DeepAgentState) -> dict:
    """Routing layer: Decide which subagent to invoke"""
    phase = state.get("current_phase", "")

    if phase == "planning":
        target = "repo-analyst"
    elif phase == "analysis_complete":
        target = "harness-expert"
    elif phase == "design_complete":
        target = "harness-expert"
    elif phase == "resources_created":
        target = "deployer"
    else:
        target = None

    return {
        "subagent_target": target,
        "messages": [AIMessage(content=f"Routing to subagent: {target}")]
    }


def aggregate_results(state: DeepAgentState) -> dict:
    """Aggregate results from subagents"""
    return {
        "messages": [AIMessage(content="Aggregating results from subagent execution")]
    }


def human_approval_gate(state: DeepAgentState) -> dict:
    """Human-in-the-loop approval for production operations"""
    return {
        "needs_approval": True,
        "messages": [AIMessage(content="Requesting human approval for production resource creation")]
    }


def finalize_workflow(state: DeepAgentState) -> dict:
    """Final validation and reporting"""
    return {
        "current_phase": "complete",
        "messages": [AIMessage(content="Workflow complete! All resources created and verified.")]
    }


# =============================================================================
# REPO-ANALYST SUBAGENT NODES
# =============================================================================

def repo_analyst_entry(state: DeepAgentState) -> dict:
    """Repo Analyst: Entry point"""
    return {
        "messages": [AIMessage(content="[REPO-ANALYST] Starting repository analysis")]
    }


def clone_repository(state: DeepAgentState) -> dict:
    """Clone the target repository"""
    return {
        "messages": [AIMessage(content=f"[REPO-ANALYST] Cloning repository: {state.get('repo_url')}")]
    }


def run_repomix(state: DeepAgentState) -> dict:
    """Run Repomix for code pattern extraction"""
    return {
        "messages": [AIMessage(content="[REPO-ANALYST] Running Repomix with compression for pattern extraction")]
    }


def detect_languages(state: DeepAgentState) -> dict:
    """Detect programming languages and versions"""
    return {
        "detected_languages": ["python", "typescript"],
        "messages": [AIMessage(content="[REPO-ANALYST] Detected languages: Python 3.12, TypeScript 5.0")]
    }


def detect_frameworks(state: DeepAgentState) -> dict:
    """Detect frameworks and build tools"""
    return {
        "detected_frameworks": ["fastapi", "react"],
        "messages": [AIMessage(content="[REPO-ANALYST] Detected frameworks: FastAPI, React")]
    }


def find_configs(state: DeepAgentState) -> dict:
    """Find configuration files (Dockerfile, K8s, Terraform)"""
    return {
        "has_dockerfile": True,
        "has_kubernetes": True,
        "messages": [AIMessage(content="[REPO-ANALYST] Found: Dockerfile, k8s/deployment.yaml")]
    }


def repo_analyst_exit(state: DeepAgentState) -> dict:
    """Repo Analyst: Exit with results"""
    return {
        "current_phase": "analysis_complete",
        "repo_analysis": {
            "languages": state.get("detected_languages", []),
            "frameworks": state.get("detected_frameworks", []),
            "has_docker": state.get("has_dockerfile", False),
            "has_k8s": state.get("has_kubernetes", False)
        },
        "messages": [AIMessage(content="[REPO-ANALYST] Analysis complete. Repository profiled.")]
    }


# =============================================================================
# HARNESS-EXPERT SUBAGENT NODES
# =============================================================================

def harness_expert_entry(state: DeepAgentState) -> dict:
    """Harness Expert: Entry point"""
    phase = state.get("current_phase", "")
    if phase == "analysis_complete":
        return {
            "messages": [AIMessage(content="[HARNESS-EXPERT] Starting pipeline design phase")]
        }
    else:
        return {
            "messages": [AIMessage(content="[HARNESS-EXPERT] Starting resource creation phase")]
        }


def design_pipeline(state: DeepAgentState) -> dict:
    """Design the CI/CD pipeline structure"""
    return {
        "pipeline_design": {
            "stages": ["build", "test", "scan", "deploy-dev", "deploy-prod"],
            "type": "ci-cd"
        },
        "messages": [AIMessage(content="[HARNESS-EXPERT] Designed 5-stage CI/CD pipeline")]
    }


def design_templates(state: DeepAgentState) -> dict:
    """Design reusable step and stage templates"""
    return {
        "messages": [AIMessage(content="[HARNESS-EXPERT] Designed 8 reusable templates (steps + stages)")]
    }


def list_requirements(state: DeepAgentState) -> dict:
    """List required connectors, secrets, environments"""
    return {
        "required_connectors": ["github", "docker-hub", "kubernetes"],
        "required_secrets": ["github-token", "docker-credentials"],
        "environments": ["dev", "staging", "prod"],
        "messages": [AIMessage(content="[HARNESS-EXPERT] Listed requirements: 3 connectors, 2 secrets, 3 environments")]
    }


def create_secrets(state: DeepAgentState) -> dict:
    """Create secrets in Harness"""
    return {
        "created_resources": state.get("created_resources", []) + ["secret:github-token", "secret:docker-creds"],
        "messages": [AIMessage(content="[HARNESS-EXPERT] Created 2 secrets")]
    }


def create_connectors(state: DeepAgentState) -> dict:
    """Create connectors (Git, Docker, K8s, Cloud)"""
    return {
        "created_resources": state.get("created_resources", []) + ["connector:github", "connector:docker", "connector:k8s"],
        "messages": [AIMessage(content="[HARNESS-EXPERT] Created 3 connectors")]
    }


def create_environments(state: DeepAgentState) -> dict:
    """Create environments and infrastructure definitions"""
    return {
        "created_resources": state.get("created_resources", []) + ["env:dev", "env:staging", "env:prod"],
        "messages": [AIMessage(content="[HARNESS-EXPERT] Created 3 environments")]
    }


def create_service(state: DeepAgentState) -> dict:
    """Create service definition with manifests"""
    return {
        "created_resources": state.get("created_resources", []) + ["service:my-app"],
        "messages": [AIMessage(content="[HARNESS-EXPERT] Created service definition")]
    }


def create_pipeline(state: DeepAgentState) -> dict:
    """Create the complete CI/CD pipeline"""
    return {
        "pipeline_id": "my-app-cicd",
        "created_resources": state.get("created_resources", []) + ["pipeline:my-app-cicd"],
        "messages": [AIMessage(content="[HARNESS-EXPERT] Created CI/CD pipeline: my-app-cicd")]
    }


def harness_expert_exit(state: DeepAgentState) -> dict:
    """Harness Expert: Exit with results"""
    phase = state.get("current_phase", "")
    if phase == "analysis_complete":
        return {
            "current_phase": "design_complete",
            "messages": [AIMessage(content="[HARNESS-EXPERT] Design phase complete")]
        }
    else:
        return {
            "current_phase": "resources_created",
            "messages": [AIMessage(content="[HARNESS-EXPERT] All resources created successfully")]
        }


# =============================================================================
# DEPLOYER SUBAGENT NODES
# =============================================================================

def deployer_entry(state: DeepAgentState) -> dict:
    """Deployer: Entry point"""
    return {
        "messages": [AIMessage(content="[DEPLOYER] Starting deployment verification")]
    }


def trigger_execution(state: DeepAgentState) -> dict:
    """Trigger pipeline execution"""
    return {
        "execution_id": "exec-12345",
        "messages": [AIMessage(content=f"[DEPLOYER] Triggered pipeline: {state.get('pipeline_id')}")]
    }


def monitor_execution(state: DeepAgentState) -> dict:
    """Monitor deployment progress"""
    return {
        "messages": [AIMessage(content="[DEPLOYER] Monitoring execution... Build → Test → Deploy stages running")]
    }


def verify_deployment(state: DeepAgentState) -> dict:
    """Verify deployment success"""
    return {
        "deployment_status": "success",
        "messages": [AIMessage(content="[DEPLOYER] ✅ Deployment verified! Service is healthy.")]
    }


def deployer_exit(state: DeepAgentState) -> dict:
    """Deployer: Exit with results"""
    return {
        "current_phase": "deployment_verified",
        "messages": [AIMessage(content="[DEPLOYER] Verification complete")]
    }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_planning(state: DeepAgentState) -> str:
    """Route after planning phase"""
    return "route_to_subagent"


def route_to_subagent_target(state: DeepAgentState) -> str:
    """Route to appropriate subagent"""
    target = state.get("subagent_target")
    if target == "repo-analyst":
        return "repo_analyst_entry"
    elif target == "harness-expert":
        return "harness_expert_entry"
    elif target == "deployer":
        return "deployer_entry"
    return "aggregate_results"


def route_after_aggregation(state: DeepAgentState) -> str:
    """Route after aggregating subagent results"""
    phase = state.get("current_phase", "")

    if phase == "analysis_complete":
        return "route_to_subagent"  # Design phase
    elif phase == "design_complete":
        return "human_approval_gate"  # Need approval before creation
    elif phase == "resources_created":
        return "route_to_subagent"  # Verification phase
    elif phase == "deployment_verified":
        return "finalize_workflow"
    else:
        return "route_to_subagent"


def route_after_approval(state: DeepAgentState) -> str:
    """Route after human approval"""
    if state.get("approved", True):  # Default to approved in demo
        return "route_to_subagent"  # Proceed with creation
    else:
        return END  # User rejected


def route_harness_expert_phase(state: DeepAgentState) -> str:
    """Route harness expert to correct phase"""
    phase = state.get("current_phase", "")
    if phase == "analysis_complete":
        return "design_pipeline"  # Design phase
    else:
        return "create_secrets"  # Creation phase


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

graph = StateGraph(DeepAgentState)

# Main orchestrator nodes
graph.add_node("plan_task", plan_task)
graph.add_node("route_to_subagent", route_to_subagent)
graph.add_node("aggregate_results", aggregate_results)
graph.add_node("human_approval_gate", human_approval_gate)
graph.add_node("finalize_workflow", finalize_workflow)

# Repo-Analyst subagent nodes
graph.add_node("repo_analyst_entry", repo_analyst_entry)
graph.add_node("clone_repository", clone_repository)
graph.add_node("run_repomix", run_repomix)
graph.add_node("detect_languages", detect_languages)
graph.add_node("detect_frameworks", detect_frameworks)
graph.add_node("find_configs", find_configs)
graph.add_node("repo_analyst_exit", repo_analyst_exit)

# Harness-Expert subagent nodes
graph.add_node("harness_expert_entry", harness_expert_entry)
graph.add_node("design_pipeline", design_pipeline)
graph.add_node("design_templates", design_templates)
graph.add_node("list_requirements", list_requirements)
graph.add_node("create_secrets", create_secrets)
graph.add_node("create_connectors", create_connectors)
graph.add_node("create_environments", create_environments)
graph.add_node("create_service", create_service)
graph.add_node("create_pipeline", create_pipeline)
graph.add_node("harness_expert_exit", harness_expert_exit)

# Deployer subagent nodes
graph.add_node("deployer_entry", deployer_entry)
graph.add_node("trigger_execution", trigger_execution)
graph.add_node("monitor_execution", monitor_execution)
graph.add_node("verify_deployment", verify_deployment)
graph.add_node("deployer_exit", deployer_exit)

# Main flow
graph.add_edge(START, "plan_task")
graph.add_conditional_edges("plan_task", route_after_planning)
graph.add_conditional_edges("route_to_subagent", route_to_subagent_target)
graph.add_conditional_edges("aggregate_results", route_after_aggregation)
graph.add_conditional_edges("human_approval_gate", route_after_approval)
graph.add_edge("finalize_workflow", END)

# Repo-Analyst flow
graph.add_edge("repo_analyst_entry", "clone_repository")
graph.add_edge("clone_repository", "run_repomix")
graph.add_edge("run_repomix", "detect_languages")
graph.add_edge("detect_languages", "detect_frameworks")
graph.add_edge("detect_frameworks", "find_configs")
graph.add_edge("find_configs", "repo_analyst_exit")
graph.add_edge("repo_analyst_exit", "aggregate_results")

# Harness-Expert flow (design OR creation)
graph.add_conditional_edges("harness_expert_entry", route_harness_expert_phase)

# Design phase
graph.add_edge("design_pipeline", "design_templates")
graph.add_edge("design_templates", "list_requirements")
graph.add_edge("list_requirements", "harness_expert_exit")

# Creation phase
graph.add_edge("create_secrets", "create_connectors")
graph.add_edge("create_connectors", "create_environments")
graph.add_edge("create_environments", "create_service")
graph.add_edge("create_service", "create_pipeline")
graph.add_edge("create_pipeline", "harness_expert_exit")

# Deployer flow
graph.add_edge("deployer_entry", "trigger_execution")
graph.add_edge("trigger_execution", "monitor_execution")
graph.add_edge("monitor_execution", "verify_deployment")
graph.add_edge("verify_deployment", "deployer_exit")
graph.add_edge("deployer_exit", "aggregate_results")

# Compile the graph
app = graph.compile()
