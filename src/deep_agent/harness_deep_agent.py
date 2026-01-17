"""Core Deep Agent implementation for Harness automation.

This module provides the main HarnessDeepAgent class that orchestrates
infrastructure tasks using specialized subagents.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import httpx

from .agent_registry import AgentRegistry, SubagentSpec, TaskRequirements, default_subagent_factory

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the Harness Deep Agent."""

    harness_account_id: str
    harness_api_url: str
    harness_api_token: str
    org_identifier: str = "default"
    project_identifier: Optional[str] = None
    enabled_subagents: Sequence[str] = field(
        default_factory=lambda: (
            "iac-golden-architect",
            "container-workflow",
            "team-accelerator",
        )
    )
    mcp_server_host: str = "0.0.0.0"
    mcp_server_port: int = 8000
    log_level: str = "INFO"


class HarnessDeepAgent:
    """Deep Agent for Harness infrastructure automation.

    This agent orchestrates complex infrastructure tasks by delegating to
    specialized subagents: iac-golden-architect, container-workflow, and
    team-accelerator.

    Attributes:
        config: Agent configuration
        client: HTTP client for Harness API calls
    """

    def __init__(self, config: AgentConfig):
        """Initialize the Deep Agent.

        Args:
            config: Agent configuration with Harness credentials
        """
        self.config = config
        self.registry = AgentRegistry()
        self._register_default_subagents()
        self.client = httpx.AsyncClient(
            base_url=config.harness_api_url,
            headers={
                "x-api-key": config.harness_api_token,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logging.basicConfig(level=config.log_level)
        logger.info(
            "Initialized HarnessDeepAgent with subagents: %s",
            ", ".join(self.registry.list_names()),
        )

    def _register_default_subagents(self) -> None:
        """Register baseline subagents, honoring enabled_subagents if provided."""
        enabled = set(self.config.enabled_subagents or ())
        default_specs = [
            SubagentSpec(
                name="iac-golden-architect",
                description="Infrastructure as code planning and Terraform guidance.",
                capabilities=("terraform", "iac", "planning"),
                supported_tasks=("terraform_plan", "iac_design", "iac_review"),
                factory=default_subagent_factory("iac-golden-architect"),
            ),
            SubagentSpec(
                name="container-workflow",
                description="Container build, packaging, and deployment workflows.",
                capabilities=("docker", "containers", "build"),
                supported_tasks=("containerize", "docker_build", "image_scan"),
                factory=default_subagent_factory("container-workflow"),
            ),
            SubagentSpec(
                name="team-accelerator",
                description="Repository setup and pipeline automation.",
                capabilities=("repositories", "pipelines", "harness"),
                supported_tasks=("repo_setup", "pipeline_create", "ci_bootstrap"),
                factory=default_subagent_factory("team-accelerator"),
            ),
        ]
        for spec in default_specs:
            if not enabled or spec.name in enabled:
                self.registry.register(spec)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    async def create_repository(
        self,
        repo_name: str,
        project_identifier: str,
        description: str = "",
        default_branch: str = "main",
    ) -> Dict[str, Any]:
        """Create a new repository in Harness Code.

        Args:
            repo_name: Name of the repository
            project_identifier: Harness project identifier
            description: Repository description
            default_branch: Default branch name

        Returns:
            Repository creation response
        """
        logger.info(f"Creating repository: {repo_name} in project: {project_identifier}")

        payload = {
            "default_branch": default_branch,
            "description": description,
            "identifier": repo_name,
            "is_public": False,
        }

        url = f"/code/api/v1/repos/{self.config.org_identifier}/{project_identifier}"
        response = await self.client.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Repository created successfully: {result.get('path')}")
        return result

    async def create_pipeline(
        self,
        pipeline_name: str,
        project_identifier: str,
        pipeline_yaml: str,
    ) -> Dict[str, Any]:
        """Create a new pipeline in Harness.

        Args:
            pipeline_name: Name of the pipeline
            project_identifier: Harness project identifier
            pipeline_yaml: Pipeline YAML definition

        Returns:
            Pipeline creation response
        """
        logger.info(f"Creating pipeline: {pipeline_name}")

        payload = {
            "identifier": pipeline_name,
            "name": pipeline_name,
            "org_identifier": self.config.org_identifier,
            "project_identifier": project_identifier,
            "yaml": pipeline_yaml,
        }

        url = f"/pipeline/api/pipelines/v2"
        response = await self.client.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        logger.info(f"Pipeline created successfully: {pipeline_name}")
        return result

    async def get_repositories(
        self, project_identifier: str, page: int = 1, limit: int = 50
    ) -> Dict[str, Any]:
        """List repositories in a Harness Code project.

        Args:
            project_identifier: Harness project identifier
            page: Page number for pagination
            limit: Number of results per page

        Returns:
            List of repositories
        """
        url = f"/code/api/v1/repos/{self.config.org_identifier}/{project_identifier}"
        params = {"page": page, "limit": limit}

        response = await self.client.get(url, params=params)
        response.raise_for_status()

        return response.json()

    async def delegate_to_subagent(
        self, subagent: str, task: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate a task to a specialized subagent.

        Args:
            subagent: Name of the subagent (e.g., 'iac-golden-architect')
            task: Task description
            context: Task context and parameters

        Returns:
            Subagent execution result
        """
        if not self.registry.get(subagent):
            raise ValueError(f"Subagent '{subagent}' is not enabled")

        logger.info(f"Delegating to {subagent}: {task}")

        # This is a placeholder for actual subagent delegation
        # In production, this would route to LangGraph workflow nodes
        agent_instance = self.registry.instantiate(subagent, context)
        return {
            "subagent": subagent,
            "task": task,
            "status": "delegated",
            "context": context,
            "instance": agent_instance,
        }

    def plan_subagents_for_node(
        self, node_name: str, task: str, context: Dict[str, Any], capabilities: Sequence[str]
    ):
        """Plan subagent invocations when approaching a workflow node.

        This hook is typically called by the workflow engine (for example,
        a LangGraph node) immediately before executing a specific node so
        that the appropriate subagents can be scheduled.

        Args:
            node_name: Logical name or identifier of the workflow/LangGraph
                node that is about to be executed.
            task: Natural-language description of the work to perform at
                this node (e.g., "provision QA environment").
            context: Shared task context and parameters passed between
                nodes (such as repository info, environment details, or
                previous results).
            capabilities: Capabilities that candidate subagents must
                support for this node (for example, ["git", "terraform"]).

        Returns:
            List of SubagentInvocation objects describing which subagents
            should be invoked for this node, in what order, and with what
            configuration.

        Example:
            invocations = agent.plan_subagents_for_node(
                node_name="provision_infra",
                task="Create a new QA environment",
                context=current_context,
                capabilities=["terraform", "cloud"],
            )
            # The returned list can then be used by the workflow engine to
            # dispatch work to each selected subagent.
        """
        requirements = TaskRequirements(task=task, capabilities=capabilities, allow_team=True)
        return self.registry.plan_for_node(node_name, requirements, context)

    def plan_subagents_for_edge(
        self,
        source: str,
        destination: str,
        task: str,
        context: Dict[str, Any],
        capabilities: Sequence[str],
    ):
        """Plan subagent invocations when traversing a workflow edge.

        This hook is typically called by the workflow engine when execution is
        about to move from one node to another. It inspects the task description
        and required capabilities to determine which subagents should be invoked
        along that edge.

        Args:
            source: Name or identifier of the source workflow node.
            destination: Name or identifier of the destination workflow node.
            task: Natural language or structured description of the work to be
                performed while transitioning between ``source`` and
                ``destination``.
            context: Additional context and parameters for the task, such as
                environment details, repository metadata, or user inputs. This
                dictionary is forwarded to the selected subagents.
            capabilities: List of capability identifiers required to complete
                the task on this edge (for example, infrastructure provisioning
                or code analysis capabilities).

        Returns:
            A list of subagent invocation plans (for example,
            ``SubagentInvocation`` objects) describing which subagents should be
            run for this edge and with what configuration.
        """
        requirements = TaskRequirements(task=task, capabilities=capabilities, allow_team=True)
        return self.registry.plan_for_edge(source, destination, requirements, context)

    async def health_check(self) -> Dict[str, str]:
        """Check health of the agent and Harness API connectivity.

        Returns:
            Health status
        """
        try:
            # Test Harness API connectivity
            url = f"/ng/api/user/currentUser"
            response = await self.client.get(url)
            response.raise_for_status()

            return {"status": "healthy", "harness_api": "connected"}
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
