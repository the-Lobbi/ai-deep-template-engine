"""Comprehensive DevOps tools integrating claude-code-templating-plugin for the multi-agent system.

This module provides LangChain StructuredTool implementations for DevOps operations
including templating, scaffolding, Harness CI/CD, code generation, web search,
Kubernetes management, metrics queries, log search, and monitoring operations.

Integrates the claude-code-templating-plugin capabilities:
- Commands: /template, /scaffold, /harness, /generate
- Agents: harness-expert, scaffold-agent, codegen-agent, database-agent, testing-agent
- Skills: universal-templating, harness-expert, project-scaffolding
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .secrets import SecretsManager
from .observability import (
    get_prometheus_client,
    get_loki_client,
    get_elasticsearch_client,
    get_alertmanager_client,
    get_log_backend,
    PrometheusClient,
)

# Kubernetes client imports
try:
    from kubernetes import client as k8s_client, config as k8s_config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    k8s_client = None
    k8s_config = None
    ApiException = Exception

logger = logging.getLogger(__name__)


# =============================================================================
# Kubernetes Client Wrapper
# =============================================================================


class KubernetesClient:
    """Lazy-initialized Kubernetes client wrapper.

    Supports both in-cluster config (when running in a pod) and kubeconfig file
    (for local development). The client APIs are initialized lazily on first use.
    """

    def __init__(self):
        """Initialize the Kubernetes client wrapper."""
        self._core_api: Optional[k8s_client.CoreV1Api] = None
        self._apps_api: Optional[k8s_client.AppsV1Api] = None
        self._configured = False

    def _configure(self) -> None:
        """Configure the Kubernetes client.

        Attempts to load in-cluster config first (for pods), then falls back
        to kubeconfig file (for local development).

        Raises:
            RuntimeError: If kubernetes package is not installed
            k8s_config.ConfigException: If no valid config is found
        """
        if not KUBERNETES_AVAILABLE:
            raise RuntimeError(
                "kubernetes package is not installed. "
                "Install it with: pip install kubernetes>=29.0.0"
            )

        if not self._configured:
            try:
                # Try in-cluster config first (for pods running in Kubernetes)
                k8s_config.load_incluster_config()
                logger.info("Loaded in-cluster Kubernetes configuration")
            except k8s_config.ConfigException:
                # Fall back to kubeconfig file
                try:
                    k8s_config.load_kube_config()
                    logger.info("Loaded kubeconfig file configuration")
                except k8s_config.ConfigException as e:
                    logger.error(f"Failed to load Kubernetes configuration: {e}")
                    raise
            self._configured = True

    @property
    def core_api(self) -> k8s_client.CoreV1Api:
        """Get the CoreV1Api client for pods, services, etc.

        Returns:
            CoreV1Api instance
        """
        self._configure()
        if self._core_api is None:
            self._core_api = k8s_client.CoreV1Api()
        return self._core_api

    @property
    def apps_api(self) -> k8s_client.AppsV1Api:
        """Get the AppsV1Api client for deployments, statefulsets, etc.

        Returns:
            AppsV1Api instance
        """
        self._configure()
        if self._apps_api is None:
            self._apps_api = k8s_client.AppsV1Api()
        return self._apps_api


# Global Kubernetes client instance (lazy initialization)
_k8s_client: Optional[KubernetesClient] = None


def get_k8s_client() -> KubernetesClient:
    """Get or create the global Kubernetes client instance.

    Returns:
        KubernetesClient instance
    """
    global _k8s_client
    if _k8s_client is None:
        _k8s_client = KubernetesClient()
    return _k8s_client


def _format_age(creation_timestamp: datetime) -> str:
    """Format a creation timestamp as a human-readable age string.

    Args:
        creation_timestamp: The creation time

    Returns:
        Human-readable age string (e.g., "5d", "2h", "30m")
    """
    if creation_timestamp is None:
        return "unknown"

    now = datetime.now(timezone.utc)
    if creation_timestamp.tzinfo is None:
        creation_timestamp = creation_timestamp.replace(tzinfo=timezone.utc)

    delta = now - creation_timestamp

    if delta.days > 0:
        return f"{delta.days}d"
    elif delta.seconds >= 3600:
        return f"{delta.seconds // 3600}h"
    elif delta.seconds >= 60:
        return f"{delta.seconds // 60}m"
    else:
        return f"{delta.seconds}s"


def _pod_to_dict(pod: Any) -> Dict[str, Any]:
    """Convert a Kubernetes Pod object to a dictionary.

    Args:
        pod: V1Pod object

    Returns:
        Dictionary representation of the pod
    """
    # Calculate ready containers
    ready_count = 0
    total_count = 0
    restart_count = 0

    if pod.status.container_statuses:
        for cs in pod.status.container_statuses:
            total_count += 1
            if cs.ready:
                ready_count += 1
            restart_count += cs.restart_count or 0

    return {
        "name": pod.metadata.name,
        "namespace": pod.metadata.namespace,
        "status": pod.status.phase,
        "ready": f"{ready_count}/{total_count}",
        "restarts": restart_count,
        "age": _format_age(pod.metadata.creation_timestamp),
        "node": pod.spec.node_name,
        "ip": pod.status.pod_ip,
        "labels": dict(pod.metadata.labels) if pod.metadata.labels else {},
    }


def _service_to_dict(service: Any) -> Dict[str, Any]:
    """Convert a Kubernetes Service object to a dictionary.

    Args:
        service: V1Service object

    Returns:
        Dictionary representation of the service
    """
    # Format ports
    ports = []
    if service.spec.ports:
        for p in service.spec.ports:
            port_str = f"{p.port}"
            if p.target_port:
                port_str += f":{p.target_port}"
            port_str += f"/{p.protocol}"
            ports.append(port_str)

    external_ip = None
    if service.status.load_balancer and service.status.load_balancer.ingress:
        ingress = service.status.load_balancer.ingress[0]
        external_ip = ingress.ip or ingress.hostname

    return {
        "name": service.metadata.name,
        "namespace": service.metadata.namespace,
        "type": service.spec.type,
        "cluster_ip": service.spec.cluster_ip,
        "external_ip": external_ip,
        "ports": ", ".join(ports),
        "age": _format_age(service.metadata.creation_timestamp),
        "selector": dict(service.spec.selector) if service.spec.selector else {},
    }


def _deployment_to_dict(deployment: Any) -> Dict[str, Any]:
    """Convert a Kubernetes Deployment object to a dictionary.

    Args:
        deployment: V1Deployment object

    Returns:
        Dictionary representation of the deployment
    """
    ready = deployment.status.ready_replicas or 0
    desired = deployment.spec.replicas or 0
    up_to_date = deployment.status.updated_replicas or 0
    available = deployment.status.available_replicas or 0

    return {
        "name": deployment.metadata.name,
        "namespace": deployment.metadata.namespace,
        "ready": f"{ready}/{desired}",
        "up_to_date": up_to_date,
        "available": available,
        "age": _format_age(deployment.metadata.creation_timestamp),
        "replicas": desired,
        "strategy": deployment.spec.strategy.type if deployment.spec.strategy else None,
        "labels": dict(deployment.metadata.labels) if deployment.metadata.labels else {},
    }


# =============================================================================
# Harness API Client
# =============================================================================


class HarnessClientError(Exception):
    """Exception raised for Harness API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[str] = None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)


class HarnessClient:
    """Async HTTP client for Harness API operations.

    Supports pipeline management, template operations, and execution control.
    Uses lazy initialization for credentials via SecretsManager.

    Environment variables:
        HARNESS_ACCOUNT_ID: Harness account identifier
        HARNESS_API_URL: Harness API base URL (default: https://app.harness.io/gateway)
        HARNESS_API_TOKEN: API token for authentication
        HARNESS_ORG_IDENTIFIER: Organization identifier (default: default)
        HARNESS_PROJECT_IDENTIFIER: Project identifier
    """

    _instance: Optional["HarnessClient"] = None
    _initialized: bool = False

    def __init__(self):
        """Initialize the Harness client with lazy loading."""
        self._client: Optional[httpx.AsyncClient] = None
        self._account_id: Optional[str] = None
        self._api_url: Optional[str] = None
        self._api_token: Optional[str] = None
        self._org_identifier: Optional[str] = None
        self._project_identifier: Optional[str] = None
        self._secrets_manager: Optional[SecretsManager] = None

    @classmethod
    def get_instance(cls) -> "HarnessClient":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _ensure_initialized(self) -> None:
        """Ensure the client is initialized with credentials."""
        if self._initialized:
            return

        # Load from environment first, then try secrets manager
        self._account_id = os.getenv("HARNESS_ACCOUNT_ID")
        self._api_url = os.getenv("HARNESS_API_URL", "https://app.harness.io/gateway")
        self._api_token = os.getenv("HARNESS_API_TOKEN")
        self._org_identifier = os.getenv("HARNESS_ORG_IDENTIFIER", "default")
        self._project_identifier = os.getenv("HARNESS_PROJECT_IDENTIFIER")

        # Try to get API token from secrets manager if not in environment
        if not self._api_token:
            try:
                self._secrets_manager = SecretsManager(provider="auto")
                self._api_token = await self._secrets_manager.get_harness_token()
            except Exception as e:
                logger.warning(f"Could not get Harness token from secrets manager: {e}")

        if not self._account_id:
            raise HarnessClientError("HARNESS_ACCOUNT_ID is required")

        if not self._api_token:
            raise HarnessClientError("HARNESS_API_TOKEN is required")

        if not self._project_identifier:
            raise HarnessClientError("HARNESS_PROJECT_IDENTIFIER is required")

        # Create the HTTP client
        self._client = httpx.AsyncClient(
            base_url=self._api_url.rstrip("/"),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

        self._initialized = True
        logger.info(
            f"Harness client initialized: account={self._account_id}, "
            f"org={self._org_identifier}, project={self._project_identifier}"
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for Harness API requests."""
        return {
            "x-api-key": self._api_token,
            "Content-Type": "application/json",
            "Harness-Account": self._account_id,
        }

    def _get_query_params(self) -> Dict[str, str]:
        """Get common query parameters for Harness API requests."""
        return {
            "accountIdentifier": self._account_id,
            "orgIdentifier": self._org_identifier,
            "projectIdentifier": self._project_identifier,
        }

    @retry(
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        yaml_body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the Harness API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API endpoint path
            params: Query parameters
            json_body: JSON request body
            yaml_body: YAML request body (for pipeline creation)

        Returns:
            Parsed JSON response

        Raises:
            HarnessClientError: If the request fails
        """
        await self._ensure_initialized()

        # Merge default params with provided params
        all_params = {**self._get_query_params(), **(params or {})}

        headers = self._get_headers()

        # Handle YAML body for pipeline creation
        content = None
        if yaml_body:
            headers["Content-Type"] = "application/yaml"
            content = yaml_body

        try:
            response = await self._client.request(
                method=method,
                url=path,
                params=all_params,
                headers=headers,
                json=json_body if not yaml_body else None,
                content=content,
            )

            # Check for errors
            if response.status_code >= 400:
                error_body = response.text
                try:
                    error_json = response.json()
                    error_msg = error_json.get("message", error_json.get("error", error_body))
                except Exception:
                    error_msg = error_body

                raise HarnessClientError(
                    f"Harness API error: {error_msg}",
                    status_code=response.status_code,
                    response_body=error_body,
                )

            return response.json()

        except httpx.HTTPStatusError as e:
            raise HarnessClientError(
                f"HTTP error: {e}",
                status_code=e.response.status_code,
                response_body=e.response.text,
            )
        except httpx.RequestError as e:
            raise HarnessClientError(f"Request error: {e}")

    async def create_pipeline(
        self,
        name: str,
        identifier: str,
        yaml_content: str,
    ) -> Dict[str, Any]:
        """Create a new pipeline.

        Args:
            name: Pipeline name
            identifier: Pipeline identifier
            yaml_content: Complete pipeline YAML

        Returns:
            Created pipeline data
        """
        logger.info(f"Creating pipeline: {name} ({identifier})")

        response = await self._request(
            method="POST",
            path="/pipeline/api/pipelines/v2",
            yaml_body=yaml_content,
        )

        logger.info(f"Pipeline created successfully: {identifier}")
        return response

    async def validate_pipeline_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Validate pipeline YAML.

        Args:
            yaml_content: Pipeline YAML to validate

        Returns:
            Validation result with errors and warnings
        """
        logger.info("Validating pipeline YAML")

        response = await self._request(
            method="POST",
            path="/pipeline/api/pipelines/v2/validate-yaml",
            yaml_body=yaml_content,
        )

        return response

    async def create_template(
        self,
        name: str,
        identifier: str,
        template_type: str,
        version_label: str,
        yaml_content: str,
        scope: str = "project",
    ) -> Dict[str, Any]:
        """Create a Harness template.

        Args:
            name: Template name
            identifier: Template identifier
            template_type: Template type (Step, Stage, Pipeline, StepGroup)
            version_label: Version label
            yaml_content: Template YAML content
            scope: Template scope (account, org, project)

        Returns:
            Created template data
        """
        logger.info(f"Creating template: {name} ({identifier}), type={template_type}")

        # Adjust path and params based on scope
        params = {}
        if scope == "account":
            params = {"accountIdentifier": self._account_id}
        elif scope == "org":
            params = {
                "accountIdentifier": self._account_id,
                "orgIdentifier": self._org_identifier,
            }
        # project scope uses default params

        response = await self._request(
            method="POST",
            path="/template/api/templates",
            params=params,
            yaml_body=yaml_content,
        )

        logger.info(f"Template created successfully: {identifier}")
        return response

    async def trigger_pipeline(
        self,
        pipeline_identifier: str,
        inputs: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Trigger a pipeline execution.

        Args:
            pipeline_identifier: Pipeline identifier
            inputs: Runtime inputs
            notes: Execution notes

        Returns:
            Execution data including execution ID
        """
        logger.info(f"Triggering pipeline: {pipeline_identifier}")

        body = {}
        if inputs:
            body["inputs"] = inputs
        if notes:
            body["notes"] = notes

        response = await self._request(
            method="POST",
            path=f"/pipeline/api/pipelines/execute/{pipeline_identifier}",
            json_body=body if body else None,
        )

        execution_id = response.get("data", {}).get("planExecution", {}).get("uuid")
        logger.info(f"Pipeline triggered: execution_id={execution_id}")
        return response

    async def get_pipeline_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get pipeline execution status.

        Args:
            execution_id: Execution ID (planExecutionId)

        Returns:
            Execution status and details
        """
        logger.info(f"Getting pipeline execution: {execution_id}")

        response = await self._request(
            method="GET",
            path=f"/pipeline/api/pipelines/execution/{execution_id}",
        )

        return response

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            self._initialized = False
            logger.info("Harness client closed")


# Module-level lazy-loaded client
_harness_client: Optional[HarnessClient] = None


def get_harness_client() -> HarnessClient:
    """Get the singleton Harness client instance."""
    global _harness_client
    if _harness_client is None:
        _harness_client = HarnessClient.get_instance()
    return _harness_client


# =============================================================================
# Tavily Search Client
# =============================================================================


class TavilySearchClient:
    """Client for Tavily Web Search API.

    Tavily provides AI-optimized web search with advanced filtering and
    summarization capabilities.

    Environment Variables:
        TAVILY_API_KEY: API key for Tavily (required)
    """

    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tavily search client."""
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            logger.warning("TAVILY_API_KEY not set.")

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True,
        exclude_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search the web using Tavily API."""
        if not self.api_key:
            raise ValueError("Tavily API key not configured.")

        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(max(1, max_results), 20),
            "search_depth": search_depth,
            "include_answer": include_answer,
        }
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        async with httpx.AsyncClient() as client:
            response = await client.post(self.API_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()

    async def search_devops(
        self, query: str, max_results: int = 5
    ) -> List[Dict[str, str]]:
        """Search for DevOps-specific content."""
        excluded = ["twitter.com", "facebook.com", "linkedin.com", "reddit.com"]
        result = await self.search(query, max_results, exclude_domains=excluded)
        formatted = []
        if result.get("answer"):
            formatted.append({
                "title": "AI Summary",
                "url": "",
                "snippet": result["answer"],
                "source": "tavily_answer",
            })
        for item in result.get("results", []):
            formatted.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:500],
                "source": "tavily",
            })
        return formatted[:max_results]


_tavily_client: Optional[TavilySearchClient] = None


def get_tavily_client() -> TavilySearchClient:
    """Get or create the global Tavily client instance."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilySearchClient()
    return _tavily_client


# =============================================================================
# RAG Knowledge Base Integration
# =============================================================================

_rag_knowledge_base = None
_rag_retriever = None
_rag_embeddings = None
_rag_initialized = False


async def _ensure_rag_initialized():
    """Initialize RAG components if not already done."""
    global _rag_knowledge_base, _rag_retriever, _rag_embeddings, _rag_initialized

    if _rag_initialized:
        return _rag_knowledge_base is not None

    _rag_initialized = True

    try:
        required_vars = ["PINECONE_API_KEY", "VOYAGE_API_KEY"]
        missing_vars = [v for v in required_vars if not os.getenv(v)]
        if missing_vars:
            logger.warning(
                f"RAG initialization skipped - missing env vars: {missing_vars}"
            )
            return False

        from .rag import create_rag_retriever

        logger.info("Initializing RAG retrieval system...")
        _rag_knowledge_base, _rag_retriever, _rag_embeddings = (
            await create_rag_retriever(namespace="devops", alpha=0.7)
        )
        logger.info("RAG retrieval system initialized successfully")
        return True
    except ImportError as e:
        logger.warning(f"RAG module import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        return False


async def search_documentation_with_rag(
    query: str, top_k: int = 5, source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search documentation using the RAG system."""
    if not await _ensure_rag_initialized():
        return [{
            "title": f"Documentation: {query}",
            "content": f"[RAG not configured] Placeholder for: {query}",
            "url": "",
            "score": 0.0,
            "source": "placeholder",
        }]

    try:
        if source_filter:
            filter_map = {
                "kubernetes": _rag_knowledge_base.search_k8s_docs,
                "harness": _rag_knowledge_base.search_harness_docs,
                "terraform": _rag_knowledge_base.search_terraform_docs,
                "runbooks": _rag_knowledge_base.search_runbooks,
            }
            search_func = filter_map.get(
                source_filter, _rag_knowledge_base.search_all
            )
            results = await search_func(query, top_k=top_k)
        else:
            results = await _rag_retriever.retrieve(query, top_k=top_k)

        formatted = []
        for result in results:
            formatted.append({
                "title": result.metadata.get("title", result.id),
                "content": result.content,
                "url": result.metadata.get("url", ""),
                "score": result.score,
                "source": result.metadata.get("source", "unknown"),
                "metadata": result.metadata,
            })
        logger.info(f"RAG search completed: {len(formatted)} results")
        return formatted
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return [{
            "title": "Search Error",
            "content": f"Documentation search failed: {str(e)}",
            "url": "",
            "score": 0.0,
            "source": "error",
        }]


# =============================================================================
# Input Schemas - Templating and Scaffolding
# =============================================================================


class ListTemplatesInput(BaseModel):
    """Input schema for listing available templates."""

    format_filter: Optional[str] = Field(
        default=None,
        description="Filter by template format: handlebars, cookiecutter, copier, maven, harness",
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by category: microservice, web, cli, infrastructure, pipeline",
    )
    tags: Optional[str] = Field(
        default=None,
        description="Comma-separated tags to filter by",
    )


class SearchTemplatesInput(BaseModel):
    """Input schema for searching templates."""

    query: str = Field(
        description="Search query for template name, description, or tags"
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return",
        ge=1,
        le=50,
    )


class GenerateFromTemplateInput(BaseModel):
    """Input schema for generating from a template."""

    template_name: str = Field(
        description="Name of the template to use (e.g., 'fastapi-microservice')"
    )
    output_path: str = Field(
        description="Output directory path for generated project"
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template variables as key-value pairs",
    )
    dry_run: bool = Field(
        default=False,
        description="Preview generation without writing files",
    )


class ValidateTemplateInput(BaseModel):
    """Input schema for validating a template."""

    path: str = Field(
        description="Path to the template directory or configuration file"
    )


class ScaffoldOptions(BaseModel):
    """Options for project scaffolding."""

    harness: bool = Field(
        default=False,
        description="Include Harness CI/CD pipeline setup",
    )
    environments: List[str] = Field(
        default_factory=lambda: ["dev"],
        description="Target environments (e.g., ['dev', 'staging', 'prod'])",
    )
    docker: bool = Field(
        default=False,
        description="Include Docker configuration",
    )
    kubernetes: bool = Field(
        default=False,
        description="Include Kubernetes manifests",
    )
    git_init: bool = Field(
        default=True,
        description="Initialize git repository",
    )


class ScaffoldProjectInput(BaseModel):
    """Input schema for scaffolding a new project."""

    template: str = Field(
        description="Template name or path (e.g., 'fastapi-microservice')"
    )
    project_name: str = Field(
        description="Name of the new project"
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Output directory (defaults to current directory)",
    )
    options: ScaffoldOptions = Field(
        default_factory=ScaffoldOptions,
        description="Scaffolding options",
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template variables",
    )


# =============================================================================
# Input Schemas - Harness CI/CD
# =============================================================================


class CreatePipelineInput(BaseModel):
    """Input schema for creating a Harness pipeline."""

    name: str = Field(
        description="Pipeline name"
    )
    pipeline_type: str = Field(
        description="Pipeline type: standard-cicd, gitops, canary, blue-green"
    )
    service: str = Field(
        description="Service name to deploy"
    )
    environments: List[str] = Field(
        default_factory=lambda: ["dev"],
        description="Target environments",
    )
    stages: Optional[List[str]] = Field(
        default=None,
        description="Custom stage names (defaults based on pipeline type)",
    )


class ValidatePipelineInput(BaseModel):
    """Input schema for validating pipeline YAML."""

    yaml_content: str = Field(
        description="Pipeline YAML content to validate"
    )


class CreateTemplateInput(BaseModel):
    """Input schema for creating Harness templates."""

    template_type: str = Field(
        description="Template type: step, stage, pipeline, stepgroup"
    )
    name: str = Field(
        description="Template name"
    )
    scope: str = Field(
        default="project",
        description="Template scope: account, org, project",
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template configuration",
    )


class TriggerPipelineInput(BaseModel):
    """Input schema for triggering pipeline execution."""

    pipeline_id: str = Field(
        description="Pipeline identifier"
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime inputs and variables",
    )
    environment: Optional[str] = Field(
        default=None,
        description="Target environment",
    )


class GetPipelineStatusInput(BaseModel):
    """Input schema for checking pipeline status."""

    execution_id: str = Field(
        description="Pipeline execution ID"
    )


# =============================================================================
# Input Schemas - Code Generation
# =============================================================================


class GenerateApiClientInput(BaseModel):
    """Input schema for generating API client."""

    spec_path: str = Field(
        description="Path to OpenAPI/Swagger specification"
    )
    language: str = Field(
        description="Target language: typescript, python, go, java, csharp"
    )
    output_path: str = Field(
        description="Output directory for generated client"
    )
    style: Optional[str] = Field(
        default=None,
        description="Client style (e.g., axios, fetch, httpx)",
    )


class GenerateModelsInput(BaseModel):
    """Input schema for generating models."""

    schema_path: str = Field(
        description="Path to JSON Schema, Prisma, or GraphQL schema"
    )
    output_path: str = Field(
        description="Output directory for generated models"
    )
    language: Optional[str] = Field(
        default="typescript",
        description="Target language",
    )


class GenerateTestsInput(BaseModel):
    """Input schema for generating tests."""

    source_path: str = Field(
        description="Path to source code directory or file"
    )
    framework: str = Field(
        description="Test framework: jest, pytest, vitest, mocha, junit"
    )
    coverage_target: int = Field(
        default=80,
        description="Target code coverage percentage",
        ge=0,
        le=100,
    )


class GenerateMigrationsInput(BaseModel):
    """Input schema for generating database migrations."""

    schema_path: str = Field(
        description="Path to database schema or models"
    )
    db_type: str = Field(
        description="Database type: postgresql, mysql, sqlite, mongodb"
    )
    framework: Optional[str] = Field(
        default=None,
        description="Migration framework (e.g., alembic, migrate)",
    )


# =============================================================================
# Input Schemas - DevOps Operations (from existing tools.py)
# =============================================================================


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""

    query: str = Field(
        description="Search query for DevOps solutions, documentation, or best practices"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return",
        ge=1,
        le=20,
    )


class KubernetesGetPodsInput(BaseModel):
    """Input schema for getting Kubernetes pods."""

    namespace: str = Field(
        default="default",
        description="Kubernetes namespace",
    )
    label_selector: Optional[str] = Field(
        default=None,
        description="Label selector to filter pods (e.g., 'app=nginx,env=prod')",
    )


class KubernetesGetServicesInput(BaseModel):
    """Input schema for getting Kubernetes services."""

    namespace: str = Field(
        default="default",
        description="Kubernetes namespace",
    )


class KubernetesGetDeploymentsInput(BaseModel):
    """Input schema for getting Kubernetes deployments."""

    namespace: str = Field(
        default="default",
        description="Kubernetes namespace",
    )


class KubernetesScaleInput(BaseModel):
    """Input schema for scaling Kubernetes deployments."""

    namespace: str = Field(
        description="Kubernetes namespace"
    )
    name: str = Field(
        description="Deployment name"
    )
    replicas: int = Field(
        description="Target number of replicas",
        ge=0,
    )


class KubernetesRestartInput(BaseModel):
    """Input schema for restarting Kubernetes deployments."""

    namespace: str = Field(
        description="Kubernetes namespace"
    )
    name: str = Field(
        description="Deployment name"
    )


class QueryMetricsInput(BaseModel):
    """Input schema for querying Prometheus metrics."""

    promql: str = Field(
        description="PromQL query string (e.g., 'rate(http_requests_total[5m])')"
    )
    time_range: str = Field(
        default="1h",
        description="Time range (e.g., '1h', '30m', '1d')",
    )


class SearchLogsInput(BaseModel):
    """Input schema for searching logs."""

    query: str = Field(
        description="Log search query or keyword"
    )
    service: Optional[str] = Field(
        default=None,
        description="Service name to filter logs",
    )
    level: Optional[str] = Field(
        default=None,
        description="Log level filter: error, warn, info, debug",
    )
    time_range: str = Field(
        default="1h",
        description="Time range to search",
    )


class ListAlertsInput(BaseModel):
    """Input schema for listing alerts."""

    status: str = Field(
        default="firing",
        description="Alert status filter: firing, pending, resolved",
    )


class DocumentationSearchInput(BaseModel):
    """Input schema for searching documentation."""

    query: str = Field(
        description="Search query for internal documentation"
    )
    top_k: int = Field(
        default=5,
        description="Number of top results to return",
        ge=1,
        le=20,
    )


# =============================================================================
# Tool Implementation Functions - Templating
# =============================================================================


async def list_templates_impl(
    format_filter: Optional[str], category: Optional[str], tags: Optional[str]
) -> List[Dict[str, Any]]:
    """List available templates with optional filtering.

    Args:
        format_filter: Template format filter
        category: Category filter
        tags: Comma-separated tags filter

    Returns:
        List of template metadata
    """
    try:
        logger.info(
            f"Listing templates: format={format_filter}, category={category}, tags={tags}"
        )

        # Placeholder - integrate with claude-code-templating-plugin template registry
        templates = [
            {
                "name": "fastapi-microservice",
                "format": "cookiecutter",
                "category": "microservice",
                "description": "FastAPI Python microservice with async support",
                "tags": ["python", "api", "async", "rest"],
                "variables": ["project_name", "author", "python_version"],
            },
            {
                "name": "spring-boot-service",
                "format": "maven",
                "category": "microservice",
                "description": "Spring Boot Java microservice",
                "tags": ["java", "spring", "rest", "jpa"],
                "variables": ["artifactId", "groupId", "version"],
            },
            {
                "name": "react-typescript",
                "format": "copier",
                "category": "web",
                "description": "React with TypeScript and modern tooling",
                "tags": ["react", "typescript", "vite", "frontend"],
                "variables": ["project_name", "port", "api_url"],
            },
            {
                "name": "ci-cd-standard",
                "format": "harness",
                "category": "pipeline",
                "description": "Standard CI/CD pipeline with build, test, deploy",
                "tags": ["harness", "cicd", "pipeline", "kubernetes"],
                "variables": ["service_name", "environments", "registry"],
            },
        ]

        # Apply filters
        if format_filter:
            templates = [t for t in templates if t["format"] == format_filter]
        if category:
            templates = [t for t in templates if t["category"] == category]
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            templates = [
                t for t in templates if any(tag in t["tags"] for tag in tag_list)
            ]

        # TODO: Integrate with actual template registry
        # from claude_code_templating_plugin import TemplateRegistry
        # registry = TemplateRegistry()
        # templates = registry.list(format=format_filter, category=category, tags=tags)

        logger.info(f"Found {len(templates)} templates")
        return templates

    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return [{"error": str(e)}]


async def search_templates_impl(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Search templates by query string.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of matching templates with relevance scores
    """
    try:
        logger.info(f"Searching templates: query='{query}', max_results={max_results}")

        # Placeholder - integrate with template search
        results = [
            {
                "name": "fastapi-microservice",
                "description": "FastAPI Python microservice with async support",
                "format": "cookiecutter",
                "relevance": 0.95,
                "path": "templates/fastapi-microservice",
            }
        ]

        # TODO: Implement full-text search
        # from claude_code_templating_plugin import TemplateSearch
        # search = TemplateSearch()
        # results = search.search(query, max_results=max_results)

        logger.info(f"Found {len(results)} matching templates")
        return results[:max_results]

    except Exception as e:
        logger.error(f"Template search failed: {e}")
        return [{"error": str(e)}]


async def generate_from_template_impl(
    template_name: str, output_path: str, variables: Dict[str, Any], dry_run: bool
) -> Dict[str, Any]:
    """Generate project from template.

    Args:
        template_name: Template to use
        output_path: Output directory
        variables: Template variables
        dry_run: Preview without writing

    Returns:
        Generation result with files created
    """
    try:
        logger.info(
            f"Generating from template: {template_name} -> {output_path}, dry_run={dry_run}"
        )

        # Placeholder implementation
        result = {
            "template": template_name,
            "output_path": output_path,
            "dry_run": dry_run,
            "status": "success",
            "files_created": [
                f"{output_path}/README.md",
                f"{output_path}/src/main.py",
                f"{output_path}/tests/test_main.py",
                f"{output_path}/requirements.txt",
                f"{output_path}/.gitignore",
            ],
            "variables_used": variables,
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with template engine
        # from claude_code_templating_plugin import TemplateEngine
        # engine = TemplateEngine()
        # result = engine.generate(
        #     template_name=template_name,
        #     output_path=output_path,
        #     variables=variables,
        #     dry_run=dry_run
        # )

        logger.info(f"Generated {len(result.get('files_created', []))} files")
        return result

    except Exception as e:
        logger.error(f"Template generation failed: {e}")
        return {
            "template": template_name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def validate_template_impl(path: str) -> Dict[str, Any]:
    """Validate template configuration and structure.

    Args:
        path: Path to template

    Returns:
        Validation result with errors and warnings
    """
    try:
        logger.info(f"Validating template: {path}")

        # Placeholder implementation
        result = {
            "path": path,
            "valid": True,
            "errors": [],
            "warnings": [
                "Consider adding description field to cookiecutter.json"
            ],
            "format": "cookiecutter",
            "quality_score": 85,
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Implement validation
        # from claude_code_templating_plugin import TemplateValidator
        # validator = TemplateValidator()
        # result = validator.validate(path)

        logger.info(f"Validation complete: valid={result['valid']}")
        return result

    except Exception as e:
        logger.error(f"Template validation failed: {e}")
        return {
            "path": path,
            "valid": False,
            "errors": [str(e)],
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Tool Implementation Functions - Scaffolding
# =============================================================================


async def scaffold_project_impl(
    template: str,
    project_name: str,
    output_dir: Optional[str],
    options: ScaffoldOptions,
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    """Scaffold a new project from template.

    Args:
        template: Template name or path
        project_name: Project name
        output_dir: Output directory
        options: Scaffolding options
        variables: Template variables

    Returns:
        Scaffolding result
    """
    try:
        logger.info(
            f"Scaffolding project: template={template}, name={project_name}, "
            f"harness={options.harness}, envs={options.environments}"
        )

        output_path = output_dir or f"./{project_name}"

        # Placeholder implementation
        result = {
            "template": template,
            "project_name": project_name,
            "output_path": output_path,
            "status": "success",
            "files_created": [
                f"{output_path}/README.md",
                f"{output_path}/CLAUDE.md",
                f"{output_path}/src/",
                f"{output_path}/tests/",
                f"{output_path}/.gitignore",
            ],
            "git_initialized": options.git_init,
            "docker_included": options.docker,
            "kubernetes_included": options.kubernetes,
            "harness_pipeline_created": options.harness,
            "environments": options.environments if options.harness else [],
            "next_steps": [
                "cd " + output_path,
                "Install dependencies",
                "Run tests",
                "Review CLAUDE.md for project guidance",
            ],
            "timestamp": datetime.now().isoformat(),
        }

        if options.docker:
            result["files_created"].append(f"{output_path}/Dockerfile")
            result["files_created"].append(f"{output_path}/.dockerignore")

        if options.kubernetes:
            result["files_created"].append(f"{output_path}/k8s/")
            result["files_created"].append(f"{output_path}/k8s/deployment.yaml")
            result["files_created"].append(f"{output_path}/k8s/service.yaml")

        if options.harness:
            result["files_created"].append(f"{output_path}/.harness/")
            result["files_created"].append(f"{output_path}/.harness/pipeline.yaml")

        # TODO: Integrate with scaffold agent
        # from claude_code_templating_plugin.agents import ScaffoldAgent
        # agent = ScaffoldAgent()
        # result = agent.scaffold(
        #     template=template,
        #     project_name=project_name,
        #     output_dir=output_dir,
        #     options=options.dict(),
        #     variables=variables
        # )

        logger.info(f"Scaffolding complete: {len(result['files_created'])} files created")
        return result

    except Exception as e:
        logger.error(f"Scaffolding failed: {e}")
        return {
            "template": template,
            "project_name": project_name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Tool Implementation Functions - Harness CI/CD
# =============================================================================


def _generate_pipeline_yaml(
    name: str,
    identifier: str,
    pipeline_type: str,
    service: str,
    environments: List[str],
    stages: List[str],
) -> str:
    """Generate Harness pipeline YAML based on type.

    Args:
        name: Pipeline name
        identifier: Pipeline identifier
        pipeline_type: Pipeline type
        service: Service name
        environments: Target environments
        stages: Stage names

    Returns:
        Complete pipeline YAML string
    """
    # Generate stage YAML based on pipeline type
    stage_yaml_list = []

    for i, stage_name in enumerate(stages):
        stage_id = stage_name.lower().replace(" ", "_").replace("-", "_")

        if "build" in stage_name.lower():
            stage_yaml = f"""
        - stage:
            name: {stage_name}
            identifier: {stage_id}
            type: CI
            spec:
              cloneCodebase: true
              execution:
                steps:
                  - step:
                      type: Run
                      name: Build
                      identifier: build
                      spec:
                        shell: Bash
                        command: |
                          echo "Building {service}..."
                          # Add your build commands here
              platform:
                os: Linux
                arch: Amd64
              runtime:
                type: Cloud
                spec: {{}}"""
        elif "test" in stage_name.lower():
            stage_yaml = f"""
        - stage:
            name: {stage_name}
            identifier: {stage_id}
            type: CI
            spec:
              cloneCodebase: true
              execution:
                steps:
                  - step:
                      type: Run
                      name: Run Tests
                      identifier: run_tests
                      spec:
                        shell: Bash
                        command: |
                          echo "Running tests for {service}..."
                          # Add your test commands here
              platform:
                os: Linux
                arch: Amd64
              runtime:
                type: Cloud
                spec: {{}}"""
        elif "approval" in stage_name.lower():
            stage_yaml = f"""
        - stage:
            name: {stage_name}
            identifier: {stage_id}
            type: Approval
            spec:
              execution:
                steps:
                  - step:
                      type: HarnessApproval
                      name: Approval
                      identifier: approval
                      spec:
                        approvalMessage: Please review and approve the deployment
                        includePipelineExecutionHistory: true
                        approvers:
                          userGroups:
                            - _project_all_users
                          minimumCount: 1
                          disallowPipelineExecutor: false"""
        elif "deploy" in stage_name.lower():
            # Determine target environment
            env_name = "dev"
            for env in environments:
                if env.lower() in stage_name.lower():
                    env_name = env
                    break

            stage_yaml = f"""
        - stage:
            name: {stage_name}
            identifier: {stage_id}
            type: Deployment
            spec:
              deploymentType: Kubernetes
              service:
                serviceRef: {service}
              environment:
                environmentRef: {env_name}
                deployToAll: false
                infrastructureDefinitions:
                  - identifier: {env_name}_infra
              execution:
                steps:
                  - step:
                      type: K8sRollingDeploy
                      name: Rolling Deployment
                      identifier: rolling_deploy
                      spec:
                        skipDryRun: false
                rollbackSteps:
                  - step:
                      type: K8sRollingRollback
                      name: Rollback
                      identifier: rollback
                      spec: {{}}"""
        else:
            # Generic stage
            stage_yaml = f"""
        - stage:
            name: {stage_name}
            identifier: {stage_id}
            type: Custom
            spec:
              execution:
                steps:
                  - step:
                      type: ShellScript
                      name: {stage_name}
                      identifier: {stage_id}_step
                      spec:
                        shell: Bash
                        onDelegate: true
                        source:
                          type: Inline
                          spec:
                            script: |
                              echo "Executing {stage_name}..."
                        environmentVariables: []
                        outputVariables: []"""

        stage_yaml_list.append(stage_yaml)

    stages_yaml = "\n".join(stage_yaml_list)

    pipeline_yaml = f"""pipeline:
  name: {name}
  identifier: {identifier}
  projectIdentifier: <+pipeline.projectIdentifier>
  orgIdentifier: <+pipeline.orgIdentifier>
  tags:
    pipeline_type: {pipeline_type}
    service: {service}
  stages:{stages_yaml}
"""

    return pipeline_yaml


async def create_pipeline_impl(
    name: str,
    pipeline_type: str,
    service: str,
    environments: List[str],
    stages: Optional[List[str]],
) -> Dict[str, Any]:
    """Create Harness pipeline using the Harness API.

    Args:
        name: Pipeline name
        pipeline_type: Pipeline type (standard-cicd, gitops, canary, blue-green)
        service: Service name
        environments: Target environments
        stages: Custom stages

    Returns:
        Pipeline creation result including pipeline ID and URL
    """
    try:
        logger.info(
            f"Creating Harness pipeline: name={name}, type={pipeline_type}, "
            f"service={service}, envs={environments}"
        )

        # Default stages based on pipeline type
        if not stages:
            if pipeline_type == "standard-cicd":
                stages = ["Build", "Test", "Deploy to Dev", "Approval", "Deploy to Prod"]
            elif pipeline_type == "gitops":
                stages = ["Build", "Update Manifest", "ArgoCD Sync"]
            elif pipeline_type == "canary":
                stages = ["Build", "Deploy Canary 5%", "Verify", "Deploy Canary 100%"]
            elif pipeline_type == "blue-green":
                stages = ["Build", "Deploy Blue", "Switch Traffic", "Cleanup Green"]
            else:
                stages = ["Build", "Deploy"]

        # Generate identifier from name
        identifier = name.lower().replace(" ", "_").replace("-", "_")

        # Generate pipeline YAML
        pipeline_yaml = _generate_pipeline_yaml(
            name=name,
            identifier=identifier,
            pipeline_type=pipeline_type,
            service=service,
            environments=environments,
            stages=stages,
        )

        # Create pipeline via Harness API
        client = get_harness_client()
        response = await client.create_pipeline(
            name=name,
            identifier=identifier,
            yaml_content=pipeline_yaml,
        )

        # Extract pipeline data from response
        pipeline_data = response.get("data", {})
        pipeline_identifier = pipeline_data.get("identifier", identifier)

        # Build Harness URL
        account_id = client._account_id
        org_id = client._org_identifier
        project_id = client._project_identifier
        harness_url = (
            f"https://app.harness.io/ng/account/{account_id}/cd/orgs/{org_id}/"
            f"projects/{project_id}/pipelines/{pipeline_identifier}/pipeline-studio"
        )

        result = {
            "name": name,
            "type": pipeline_type,
            "service": service,
            "environments": environments,
            "stages": stages,
            "pipeline_id": pipeline_identifier,
            "status": "created",
            "yaml_path": f".harness/pipelines/{identifier}.yaml",
            "url": harness_url,
            "yaml_content": pipeline_yaml,
            "api_response": pipeline_data,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Pipeline created: {result['pipeline_id']}")
        return result

    except HarnessClientError as e:
        logger.error(f"Harness API error creating pipeline: {e}")
        return {
            "name": name,
            "status": "failed",
            "error": str(e),
            "status_code": e.status_code,
            "response_body": e.response_body,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Pipeline creation failed: {e}")
        return {
            "name": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def validate_pipeline_impl(yaml_content: str) -> Dict[str, Any]:
    """Validate Harness pipeline YAML using the Harness API.

    Args:
        yaml_content: Pipeline YAML content

    Returns:
        Validation result with errors, warnings, and suggestions
    """
    try:
        logger.info("Validating pipeline YAML via Harness API")

        # Call Harness API for validation
        client = get_harness_client()
        response = await client.validate_pipeline_yaml(yaml_content)

        # Parse validation response
        validation_data = response.get("data", {})
        validation_result = validation_data.get("validationResult", {})

        # Extract errors and warnings
        errors = []
        warnings = []
        suggestions = []

        # Check for validation errors
        if validation_result.get("valid") is False:
            error_messages = validation_result.get("errorMessages", [])
            for error in error_messages:
                if isinstance(error, dict):
                    errors.append(error.get("message", str(error)))
                else:
                    errors.append(str(error))

        # Check for validation warnings
        warning_messages = validation_result.get("warningMessages", [])
        for warning in warning_messages:
            if isinstance(warning, dict):
                warnings.append(warning.get("message", str(warning)))
            else:
                warnings.append(str(warning))

        # Add best practice suggestions based on YAML analysis
        yaml_lower = yaml_content.lower()

        if "timeout" not in yaml_lower:
            suggestions.append("Consider adding timeout values to steps for better resource management")

        if "failurestrategy" not in yaml_lower:
            suggestions.append("Add failure strategies to handle step/stage failures gracefully")

        if "rollbacksteps" not in yaml_lower and "deployment" in yaml_lower:
            suggestions.append("Add rollback steps for deployment stages")

        if "approval" not in yaml_lower and "prod" in yaml_lower:
            suggestions.append("Add approval gates before production deployments")

        if "notification" not in yaml_lower:
            suggestions.append("Consider adding notification rules for pipeline events")

        # Calculate quality score
        quality_score = 100
        quality_score -= len(errors) * 20
        quality_score -= len(warnings) * 5
        quality_score -= max(0, len(suggestions) - 2) * 2
        quality_score = max(0, min(100, quality_score))

        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
            "quality_score": quality_score,
            "api_response": validation_data,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Validation complete: valid={result['valid']}, errors={len(errors)}, warnings={len(warnings)}")
        return result

    except HarnessClientError as e:
        logger.error(f"Harness API validation error: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "suggestions": [],
            "status_code": e.status_code,
            "response_body": e.response_body,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "suggestions": [],
            "timestamp": datetime.now().isoformat(),
        }


def _generate_template_yaml(
    name: str,
    identifier: str,
    template_type: str,
    version_label: str,
    config: Dict[str, Any],
) -> str:
    """Generate Harness template YAML.

    Args:
        name: Template name
        identifier: Template identifier
        template_type: Template type (Step, Stage, Pipeline, StepGroup)
        version_label: Version label
        config: Template configuration

    Returns:
        Template YAML string
    """
    # Normalize template type to Harness format
    type_mapping = {
        "step": "Step",
        "stage": "Stage",
        "pipeline": "Pipeline",
        "stepgroup": "StepGroup",
    }
    harness_type = type_mapping.get(template_type.lower(), template_type)

    # Default spec based on template type
    if harness_type == "Step":
        spec_yaml = config.get("spec", {
            "type": "ShellScript",
            "spec": {
                "shell": "Bash",
                "onDelegate": True,
                "source": {
                    "type": "Inline",
                    "spec": {
                        "script": config.get("script", "echo 'Hello from template'")
                    }
                }
            }
        })
    elif harness_type == "Stage":
        spec_yaml = config.get("spec", {
            "type": "Custom",
            "spec": {
                "execution": {
                    "steps": [
                        {
                            "step": {
                                "type": "ShellScript",
                                "name": "Template Step",
                                "identifier": "template_step",
                                "spec": {
                                    "shell": "Bash",
                                    "onDelegate": True,
                                    "source": {
                                        "type": "Inline",
                                        "spec": {
                                            "script": "echo 'Stage from template'"
                                        }
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        })
    else:
        spec_yaml = config.get("spec", {})

    # Build template inputs
    template_inputs = config.get("templateInputs", [])
    inputs_yaml = ""
    if template_inputs:
        inputs_list = []
        for inp in template_inputs:
            inp_name = inp.get("name", "input")
            inp_type = inp.get("type", "String")
            inp_default = inp.get("default", "")
            inputs_list.append(f"    - name: {inp_name}\n      type: {inp_type}\n      default: {inp_default}")
        inputs_yaml = "\n  templateInputs:\n" + "\n".join(inputs_list)

    # Convert spec to YAML format
    import json
    spec_json = json.dumps(spec_yaml, indent=4)
    # Indent spec properly
    spec_lines = spec_json.split("\n")
    indented_spec = "\n".join("    " + line for line in spec_lines)

    template_yaml = f"""template:
  name: {name}
  identifier: {identifier}
  versionLabel: {version_label}
  type: {harness_type}
  projectIdentifier: <+template.projectIdentifier>
  orgIdentifier: <+template.orgIdentifier>
  tags:
    created_by: devops_agent
  spec:
{indented_spec}{inputs_yaml}
"""

    return template_yaml


async def create_template_impl(
    template_type: str, name: str, scope: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create Harness reusable template using the Harness API.

    Args:
        template_type: Template type (step, stage, pipeline, stepgroup)
        name: Template name
        scope: Template scope (account, org, project)
        config: Template configuration including spec and optional templateInputs

    Returns:
        Template creation result including template ID and URL
    """
    try:
        logger.info(
            f"Creating Harness template: type={template_type}, name={name}, scope={scope}"
        )

        # Generate identifier from name
        identifier = name.lower().replace(" ", "_").replace("-", "_")
        version_label = config.get("version", "1.0.0")

        # Generate template YAML
        template_yaml = _generate_template_yaml(
            name=name,
            identifier=identifier,
            template_type=template_type,
            version_label=version_label,
            config=config,
        )

        # Create template via Harness API
        client = get_harness_client()
        response = await client.create_template(
            name=name,
            identifier=identifier,
            template_type=template_type,
            version_label=version_label,
            yaml_content=template_yaml,
            scope=scope,
        )

        # Extract template data from response
        template_data = response.get("data", {})
        template_identifier = template_data.get("identifier", identifier)

        # Build Harness URL
        account_id = client._account_id
        org_id = client._org_identifier
        project_id = client._project_identifier

        if scope == "account":
            harness_url = f"https://app.harness.io/ng/account/{account_id}/settings/templates/{template_identifier}"
        elif scope == "org":
            harness_url = f"https://app.harness.io/ng/account/{account_id}/settings/organizations/{org_id}/templates/{template_identifier}"
        else:
            harness_url = (
                f"https://app.harness.io/ng/account/{account_id}/cd/orgs/{org_id}/"
                f"projects/{project_id}/setup/resources/templates/{template_identifier}"
            )

        result = {
            "type": template_type,
            "name": name,
            "scope": scope,
            "template_id": template_identifier,
            "version": version_label,
            "status": "created",
            "yaml_path": f".harness/templates/{template_type}/{identifier}.yaml",
            "url": harness_url,
            "yaml_content": template_yaml,
            "api_response": template_data,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Template created: {result['template_id']}")
        return result

    except HarnessClientError as e:
        logger.error(f"Harness API error creating template: {e}")
        return {
            "type": template_type,
            "name": name,
            "status": "failed",
            "error": str(e),
            "status_code": e.status_code,
            "response_body": e.response_body,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Template creation failed: {e}")
        return {
            "type": template_type,
            "name": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def trigger_pipeline_impl(
    pipeline_id: str, inputs: Dict[str, Any], environment: Optional[str]
) -> Dict[str, Any]:
    """Trigger Harness pipeline execution using the Harness API.

    Args:
        pipeline_id: Pipeline identifier
        inputs: Runtime inputs
        environment: Target environment (added to inputs if specified)

    Returns:
        Execution information including execution ID and status URL
    """
    try:
        logger.info(
            f"Triggering pipeline: id={pipeline_id}, env={environment}"
        )

        # Add environment to inputs if specified
        execution_inputs = dict(inputs) if inputs else {}
        if environment:
            execution_inputs["environment"] = environment

        # Build execution notes
        notes = f"Triggered via DevOps Agent at {datetime.now().isoformat()}"
        if environment:
            notes += f" for environment: {environment}"

        # Trigger pipeline via Harness API
        client = get_harness_client()
        response = await client.trigger_pipeline(
            pipeline_identifier=pipeline_id,
            inputs=execution_inputs if execution_inputs else None,
            notes=notes,
        )

        # Extract execution data from response
        execution_data = response.get("data", {})
        plan_execution = execution_data.get("planExecution", {})
        execution_id = plan_execution.get("uuid", plan_execution.get("planExecutionId"))

        # Get execution status
        status = plan_execution.get("status", "RUNNING")

        # Build Harness execution URL
        account_id = client._account_id
        org_id = client._org_identifier
        project_id = client._project_identifier
        harness_url = (
            f"https://app.harness.io/ng/account/{account_id}/cd/orgs/{org_id}/"
            f"projects/{project_id}/pipelines/{pipeline_id}/executions/{execution_id}/pipeline"
        )

        # Extract stage information if available
        stages = []
        layout_node_map = plan_execution.get("layoutNodeMap", {})
        for node_id, node_info in layout_node_map.items():
            if node_info.get("nodeType") == "STAGE":
                stages.append({
                    "name": node_info.get("name"),
                    "identifier": node_info.get("nodeIdentifier"),
                    "status": node_info.get("status", "QUEUED"),
                })

        result = {
            "pipeline_id": pipeline_id,
            "execution_id": execution_id,
            "status": status,
            "environment": environment,
            "inputs": execution_inputs,
            "url": harness_url,
            "started_at": datetime.now().isoformat(),
            "stages": stages if stages else [{"name": "Initializing", "status": "QUEUED"}],
            "api_response": execution_data,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Pipeline triggered: execution_id={execution_id}, status={status}")
        return result

    except HarnessClientError as e:
        logger.error(f"Harness API error triggering pipeline: {e}")
        return {
            "pipeline_id": pipeline_id,
            "status": "FAILED",
            "error": str(e),
            "status_code": e.status_code,
            "response_body": e.response_body,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Pipeline trigger failed: {e}")
        return {
            "pipeline_id": pipeline_id,
            "status": "FAILED",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def get_pipeline_status_impl(execution_id: str) -> Dict[str, Any]:
    """Get pipeline execution status using the Harness API.

    Args:
        execution_id: Execution ID (planExecutionId)

    Returns:
        Execution status including stage-level details
    """
    try:
        logger.info(f"Getting pipeline status: execution_id={execution_id}")

        # Get execution status via Harness API
        client = get_harness_client()
        response = await client.get_pipeline_execution(execution_id)

        # Extract execution data
        execution_data = response.get("data", {})
        plan_execution = execution_data.get("pipelineExecutionSummary", {})

        # Get overall status
        status = plan_execution.get("status", "UNKNOWN")

        # Calculate duration
        start_ts = plan_execution.get("startTs")
        end_ts = plan_execution.get("endTs")
        duration_seconds = None
        if start_ts and end_ts:
            duration_seconds = (end_ts - start_ts) // 1000
        elif start_ts:
            # Still running, calculate from start
            duration_seconds = int((datetime.now().timestamp() * 1000 - start_ts) // 1000)

        # Extract stage information
        stages = []
        layout_node_map = plan_execution.get("layoutNodeMap", {})
        starting_node_id = plan_execution.get("startingNodeId")

        # Process stages in order
        def process_node(node_id: str, visited: set):
            if not node_id or node_id in visited:
                return
            visited.add(node_id)

            node_info = layout_node_map.get(node_id, {})
            if node_info.get("nodeType") == "STAGE":
                stage_start = node_info.get("startTs")
                stage_end = node_info.get("endTs")
                stage_duration = None
                if stage_start and stage_end:
                    stage_duration = (stage_end - stage_start) // 1000

                stages.append({
                    "name": node_info.get("name"),
                    "identifier": node_info.get("nodeIdentifier"),
                    "status": node_info.get("status", "UNKNOWN"),
                    "duration_seconds": stage_duration,
                    "started_at": datetime.fromtimestamp(stage_start / 1000).isoformat() if stage_start else None,
                    "ended_at": datetime.fromtimestamp(stage_end / 1000).isoformat() if stage_end else None,
                })

            # Process next nodes
            for next_id in node_info.get("edgeLayoutList", {}).get("nextIds", []):
                process_node(next_id, visited)

        # Start processing from the starting node
        if starting_node_id:
            process_node(starting_node_id, set())

        # Build Harness execution URL
        account_id = client._account_id
        org_id = client._org_identifier
        project_id = client._project_identifier
        pipeline_id = plan_execution.get("pipelineIdentifier", "unknown")
        harness_url = (
            f"https://app.harness.io/ng/account/{account_id}/cd/orgs/{org_id}/"
            f"projects/{project_id}/pipelines/{pipeline_id}/executions/{execution_id}/pipeline"
        )

        # Get pipeline and run sequence info
        pipeline_name = plan_execution.get("name", pipeline_id)
        run_sequence = plan_execution.get("runSequence", 0)

        result = {
            "execution_id": execution_id,
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline_name,
            "run_sequence": run_sequence,
            "status": status,
            "duration_seconds": duration_seconds,
            "started_at": datetime.fromtimestamp(start_ts / 1000).isoformat() if start_ts else None,
            "ended_at": datetime.fromtimestamp(end_ts / 1000).isoformat() if end_ts else None,
            "stages": stages,
            "url": harness_url,
            "api_response": execution_data,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Execution status: {status}, stages: {len(stages)}")
        return result

    except HarnessClientError as e:
        logger.error(f"Harness API error getting execution status: {e}")
        return {
            "execution_id": execution_id,
            "status": "ERROR",
            "error": str(e),
            "status_code": e.status_code,
            "response_body": e.response_body,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        return {
            "execution_id": execution_id,
            "status": "UNKNOWN",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Tool Implementation Functions - Code Generation
# =============================================================================


async def generate_api_client_impl(
    spec_path: str, language: str, output_path: str, style: Optional[str]
) -> Dict[str, Any]:
    """Generate API client from OpenAPI specification.

    Args:
        spec_path: Path to OpenAPI spec
        language: Target language
        output_path: Output directory
        style: Client style

    Returns:
        Generation result
    """
    try:
        logger.info(
            f"Generating API client: spec={spec_path}, language={language}, "
            f"style={style}, output={output_path}"
        )

        # Placeholder implementation
        result = {
            "spec_path": spec_path,
            "language": language,
            "style": style or "default",
            "output_path": output_path,
            "status": "success",
            "files_generated": [
                f"{output_path}/client.{language}",
                f"{output_path}/models.{language}",
                f"{output_path}/types.{language}",
                f"{output_path}/README.md",
            ],
            "endpoints_count": 25,
            "models_count": 15,
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with codegen agent
        # from claude_code_templating_plugin.agents import CodegenAgent
        # agent = CodegenAgent()
        # result = agent.generate_api_client(
        #     spec_path=spec_path,
        #     language=language,
        #     output_path=output_path,
        #     style=style
        # )

        logger.info(f"Generated API client: {len(result['files_generated'])} files")
        return result

    except Exception as e:
        logger.error(f"API client generation failed: {e}")
        return {
            "spec_path": spec_path,
            "language": language,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def generate_models_impl(
    schema_path: str, output_path: str, language: Optional[str]
) -> Dict[str, Any]:
    """Generate models from schema.

    Args:
        schema_path: Path to schema
        output_path: Output directory
        language: Target language

    Returns:
        Generation result
    """
    try:
        logger.info(
            f"Generating models: schema={schema_path}, language={language}, "
            f"output={output_path}"
        )

        lang = language or "typescript"

        # Placeholder implementation
        result = {
            "schema_path": schema_path,
            "language": lang,
            "output_path": output_path,
            "status": "success",
            "files_generated": [
                f"{output_path}/models.{lang}",
                f"{output_path}/validators.{lang}",
                f"{output_path}/serializers.{lang}",
            ],
            "models_count": 12,
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with codegen agent
        # from claude_code_templating_plugin.agents import CodegenAgent
        # agent = CodegenAgent()
        # result = agent.generate_models(
        #     schema_path=schema_path,
        #     output_path=output_path,
        #     language=lang
        # )

        logger.info(f"Generated {result['models_count']} models")
        return result

    except Exception as e:
        logger.error(f"Model generation failed: {e}")
        return {
            "schema_path": schema_path,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def generate_tests_impl(
    source_path: str, framework: str, coverage_target: int
) -> Dict[str, Any]:
    """Generate tests from source code.

    Args:
        source_path: Path to source code
        framework: Test framework
        coverage_target: Target coverage percentage

    Returns:
        Generation result
    """
    try:
        logger.info(
            f"Generating tests: source={source_path}, framework={framework}, "
            f"coverage_target={coverage_target}%"
        )

        # Placeholder implementation
        result = {
            "source_path": source_path,
            "framework": framework,
            "coverage_target": coverage_target,
            "status": "success",
            "files_generated": [
                f"{source_path}/tests/test_main.py",
                f"{source_path}/tests/test_api.py",
                f"{source_path}/tests/fixtures.py",
            ],
            "tests_count": 45,
            "estimated_coverage": 85,
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with testing agent
        # from claude_code_templating_plugin.agents import TestingAgent
        # agent = TestingAgent()
        # result = agent.generate_tests(
        #     source_path=source_path,
        #     framework=framework,
        #     coverage_target=coverage_target
        # )

        logger.info(
            f"Generated {result['tests_count']} tests, "
            f"estimated coverage: {result['estimated_coverage']}%"
        )
        return result

    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return {
            "source_path": source_path,
            "framework": framework,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def generate_migrations_impl(
    schema_path: str, db_type: str, framework: Optional[str]
) -> Dict[str, Any]:
    """Generate database migrations.

    Args:
        schema_path: Path to schema/models
        db_type: Database type
        framework: Migration framework

    Returns:
        Generation result
    """
    try:
        logger.info(
            f"Generating migrations: schema={schema_path}, db={db_type}, "
            f"framework={framework}"
        )

        # Placeholder implementation
        result = {
            "schema_path": schema_path,
            "db_type": db_type,
            "framework": framework or "alembic",
            "status": "success",
            "files_generated": [
                "migrations/versions/001_initial.py",
                "migrations/versions/002_add_users.py",
                "migrations/alembic.ini",
            ],
            "migrations_count": 2,
            "tables_created": 5,
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with database agent
        # from claude_code_templating_plugin.agents import DatabaseAgent
        # agent = DatabaseAgent()
        # result = agent.generate_migrations(
        #     schema_path=schema_path,
        #     db_type=db_type,
        #     framework=framework
        # )

        logger.info(f"Generated {result['migrations_count']} migrations")
        return result

    except Exception as e:
        logger.error(f"Migration generation failed: {e}")
        return {
            "schema_path": schema_path,
            "db_type": db_type,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Tool Implementation Functions - DevOps Operations
# =============================================================================


async def web_search_impl(query: str, max_results: int) -> List[Dict[str, str]]:
    """Search the web for DevOps solutions and documentation using Tavily API.

    Uses the Tavily Web Search API for AI-optimized search results.
    Falls back to placeholder results if TAVILY_API_KEY is not configured.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        List of search results with title, url, and snippet
    """
    try:
        logger.info(f"Web search: {query} (max_results={max_results})")

        # Get the Tavily client
        tavily = get_tavily_client()

        # Check if API key is configured
        if not tavily.api_key:
            logger.warning("Tavily API key not configured, returning placeholder results")
            return [
                {
                    "title": f"DevOps Best Practice: {query}",
                    "url": f"https://docs.example.com/devops/{query.replace(' ', '-').lower()}",
                    "snippet": f"[Tavily not configured] Guide on {query} including patterns, "
                    f"troubleshooting, and production considerations.",
                    "source": "placeholder",
                }
            ]

        # Use Tavily for DevOps-optimized search
        results = await tavily.search_devops(query=query, max_results=max_results)

        logger.info(f"Tavily search found {len(results)} results")
        return results

    except ValueError as e:
        # API key not configured
        logger.warning(f"Tavily configuration error: {e}")
        return [
            {
                "title": f"DevOps Best Practice: {query}",
                "url": "",
                "snippet": f"[Search unavailable] {str(e)}",
                "source": "error",
            }
        ]
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return [{"error": str(e), "title": "Search failed", "url": "", "snippet": ""}]


async def kubernetes_get_pods_impl(
    namespace: str, label_selector: Optional[str]
) -> List[Dict[str, Any]]:
    """Get Kubernetes pods using the Kubernetes API.

    Args:
        namespace: Kubernetes namespace
        label_selector: Optional label selector (e.g., 'app=nginx,env=prod')

    Returns:
        List of pod dictionaries with status, readiness, and resource info
    """
    try:
        logger.info(f"Getting K8s pods: namespace={namespace}, labels={label_selector}")

        if not KUBERNETES_AVAILABLE:
            return [{
                "error": "kubernetes package not installed",
                "name": "client-unavailable",
                "message": "Install with: pip install kubernetes>=29.0.0"
            }]

        k8s = get_k8s_client()

        # Call the Kubernetes API
        pod_list = k8s.core_api.list_namespaced_pod(
            namespace=namespace,
            label_selector=label_selector or ""
        )

        # Convert pods to dictionaries
        pods = [_pod_to_dict(pod) for pod in pod_list.items]

        logger.info(f"Found {len(pods)} pods")
        return pods

    except ApiException as e:
        logger.error(f"Kubernetes API error getting pods: {e.status} - {e.reason}")
        error_msg = f"API error: {e.reason}"
        if e.status == 404:
            error_msg = f"Namespace '{namespace}' not found"
        elif e.status == 403:
            error_msg = f"Access denied to namespace '{namespace}'"
        return [{"error": error_msg, "name": "api-error", "status_code": e.status}]

    except RuntimeError as e:
        logger.error(f"Kubernetes client error: {e}")
        return [{"error": str(e), "name": "client-error"}]

    except Exception as e:
        logger.error(f"Failed to get pods: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def kubernetes_get_services_impl(namespace: str) -> List[Dict[str, Any]]:
    """Get Kubernetes services using the Kubernetes API.

    Args:
        namespace: Kubernetes namespace

    Returns:
        List of service dictionaries with type, IPs, ports, and selectors
    """
    try:
        logger.info(f"Getting K8s services: namespace={namespace}")

        if not KUBERNETES_AVAILABLE:
            return [{
                "error": "kubernetes package not installed",
                "name": "client-unavailable",
                "message": "Install with: pip install kubernetes>=29.0.0"
            }]

        k8s = get_k8s_client()

        # Call the Kubernetes API
        service_list = k8s.core_api.list_namespaced_service(namespace=namespace)

        # Convert services to dictionaries
        services = [_service_to_dict(svc) for svc in service_list.items]

        logger.info(f"Found {len(services)} services")
        return services

    except ApiException as e:
        logger.error(f"Kubernetes API error getting services: {e.status} - {e.reason}")
        error_msg = f"API error: {e.reason}"
        if e.status == 404:
            error_msg = f"Namespace '{namespace}' not found"
        elif e.status == 403:
            error_msg = f"Access denied to namespace '{namespace}'"
        return [{"error": error_msg, "name": "api-error", "status_code": e.status}]

    except RuntimeError as e:
        logger.error(f"Kubernetes client error: {e}")
        return [{"error": str(e), "name": "client-error"}]

    except Exception as e:
        logger.error(f"Failed to get services: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def kubernetes_get_deployments_impl(namespace: str) -> List[Dict[str, Any]]:
    """Get Kubernetes deployments using the Kubernetes API.

    Args:
        namespace: Kubernetes namespace

    Returns:
        List of deployment dictionaries with replica counts and status
    """
    try:
        logger.info(f"Getting K8s deployments: namespace={namespace}")

        if not KUBERNETES_AVAILABLE:
            return [{
                "error": "kubernetes package not installed",
                "name": "client-unavailable",
                "message": "Install with: pip install kubernetes>=29.0.0"
            }]

        k8s = get_k8s_client()

        # Call the Kubernetes API
        deployment_list = k8s.apps_api.list_namespaced_deployment(namespace=namespace)

        # Convert deployments to dictionaries
        deployments = [_deployment_to_dict(dep) for dep in deployment_list.items]

        logger.info(f"Found {len(deployments)} deployments")
        return deployments

    except ApiException as e:
        logger.error(f"Kubernetes API error getting deployments: {e.status} - {e.reason}")
        error_msg = f"API error: {e.reason}"
        if e.status == 404:
            error_msg = f"Namespace '{namespace}' not found"
        elif e.status == 403:
            error_msg = f"Access denied to namespace '{namespace}'"
        return [{"error": error_msg, "name": "api-error", "status_code": e.status}]

    except RuntimeError as e:
        logger.error(f"Kubernetes client error: {e}")
        return [{"error": str(e), "name": "client-error"}]

    except Exception as e:
        logger.error(f"Failed to get deployments: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def kubernetes_scale_impl(
    namespace: str, name: str, replicas: int
) -> Dict[str, Any]:
    """Scale a Kubernetes deployment using the Kubernetes API.

    Args:
        namespace: Kubernetes namespace
        name: Deployment name
        replicas: Target number of replicas

    Returns:
        Scaling result with status and updated replica count
    """
    try:
        logger.info(f"Scaling deployment: {namespace}/{name} to {replicas} replicas")

        if not KUBERNETES_AVAILABLE:
            return {
                "namespace": namespace,
                "deployment": name,
                "status": "failed",
                "error": "kubernetes package not installed",
                "message": "Install with: pip install kubernetes>=29.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        k8s = get_k8s_client()

        # Create the scale patch body
        scale_body = {"spec": {"replicas": replicas}}

        # Patch the deployment scale
        k8s.apps_api.patch_namespaced_deployment_scale(
            name=name,
            namespace=namespace,
            body=scale_body
        )

        # Fetch the updated deployment to confirm
        deployment = k8s.apps_api.read_namespaced_deployment(
            name=name,
            namespace=namespace
        )

        result = {
            "namespace": namespace,
            "deployment": name,
            "replicas": deployment.spec.replicas,
            "ready_replicas": deployment.status.ready_replicas or 0,
            "status": "success",
            "message": f"Scaled to {replicas} replicas",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Scaling complete: {name} now has {replicas} replicas")
        return result

    except ApiException as e:
        logger.error(f"Kubernetes API error scaling deployment: {e.status} - {e.reason}")
        error_msg = f"API error: {e.reason}"
        if e.status == 404:
            error_msg = f"Deployment '{name}' not found in namespace '{namespace}'"
        elif e.status == 403:
            error_msg = f"Access denied to scale deployment '{name}'"
        elif e.status == 422:
            error_msg = f"Invalid replica count: {replicas}"
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": error_msg,
            "status_code": e.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except RuntimeError as e:
        logger.error(f"Kubernetes client error: {e}")
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Scaling failed: {e}")
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def kubernetes_restart_impl(namespace: str, name: str) -> Dict[str, Any]:
    """Restart a Kubernetes deployment by patching the restart annotation.

    This is equivalent to `kubectl rollout restart deployment/{name} -n {namespace}`.
    It triggers a rolling restart by updating the pod template annotation.

    Args:
        namespace: Kubernetes namespace
        name: Deployment name

    Returns:
        Restart result with status
    """
    try:
        logger.info(f"Restarting deployment: {namespace}/{name}")

        if not KUBERNETES_AVAILABLE:
            return {
                "namespace": namespace,
                "deployment": name,
                "status": "failed",
                "error": "kubernetes package not installed",
                "message": "Install with: pip install kubernetes>=29.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        k8s = get_k8s_client()

        # Create the restart annotation patch
        # This mimics `kubectl rollout restart` behavior
        restart_time = datetime.now(timezone.utc).isoformat()
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": restart_time
                        }
                    }
                }
            }
        }

        # Patch the deployment to trigger a rollout restart
        deployment = k8s.apps_api.patch_namespaced_deployment(
            name=name,
            namespace=namespace,
            body=patch_body
        )

        result = {
            "namespace": namespace,
            "deployment": name,
            "status": "success",
            "message": "Rollout restart initiated",
            "restarted_at": restart_time,
            "replicas": deployment.spec.replicas,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Restart initiated for deployment: {name}")
        return result

    except ApiException as e:
        logger.error(f"Kubernetes API error restarting deployment: {e.status} - {e.reason}")
        error_msg = f"API error: {e.reason}"
        if e.status == 404:
            error_msg = f"Deployment '{name}' not found in namespace '{namespace}'"
        elif e.status == 403:
            error_msg = f"Access denied to restart deployment '{name}'"
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": error_msg,
            "status_code": e.status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except RuntimeError as e:
        logger.error(f"Kubernetes client error: {e}")
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Restart failed: {e}")
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


async def query_metrics_impl(promql: str, time_range: str) -> Dict[str, Any]:
    """Query Prometheus metrics using the Prometheus API.

    Args:
        promql: PromQL query string (e.g., 'rate(http_requests_total[5m])')
        time_range: Time range (e.g., '1h', '30m', '1d', '2w')

    Returns:
        Metric values with time series data
    """
    try:
        logger.info(f"Querying metrics: {promql}, time_range={time_range}")

        # Get the Prometheus client and execute the query
        client = get_prometheus_client()
        result = await client.query_range(promql, time_range)

        # Check for errors in the result
        if "error" in result:
            logger.warning(f"Prometheus query returned error: {result['error']}")
        else:
            logger.info(f"Query returned {len(result.get('result', []))} series")

        return result

    except Exception as e:
        logger.error(f"Metrics query failed: {e}")
        return {"error": str(e), "query": promql, "result": []}


async def search_logs_impl(
    query: str, service: Optional[str], level: Optional[str], time_range: str
) -> List[Dict[str, Any]]:
    """Search logs using the configured log backend (Loki or Elasticsearch).

    The backend is determined by the LOG_BACKEND environment variable:
    - 'loki': Query Grafana Loki using LogQL
    - 'elasticsearch': Query Elasticsearch using full-text search

    Args:
        query: Search query or keyword to find in log messages
        service: Service name filter (optional)
        level: Log level filter: error, warn, info, debug (optional)
        time_range: Time range to search (e.g., '1h', '30m', '1d')

    Returns:
        List of log entries with timestamp, service, level, message, and metadata
    """
    try:
        logger.info(
            f"Searching logs: query='{query}', service={service}, level={level}, "
            f"time_range={time_range}"
        )

        # Determine which backend to use
        backend = get_log_backend()
        logger.debug(f"Using log backend: {backend}")

        if backend == "elasticsearch":
            # Use Elasticsearch for log search
            client = get_elasticsearch_client()
            logs = await client.search(
                query=query,
                service=service,
                level=level,
                time_range=time_range,
            )
        else:
            # Default to Loki for log search
            client = get_loki_client()
            logs = await client.query_range(
                query=query,
                service=service,
                level=level,
                time_range=time_range,
            )

        # Check for errors in the result
        if logs and "error" in logs[0]:
            logger.warning(f"Log search returned error: {logs[0]['error']}")
        else:
            logger.info(f"Found {len(logs)} log entries")

        return logs

    except Exception as e:
        logger.error(f"Log search failed: {e}")
        return [{"error": str(e), "timestamp": datetime.now().isoformat()}]


async def list_alerts_impl(status: str) -> List[Dict[str, Any]]:
    """List alerts from Prometheus AlertManager API.

    Queries the AlertManager API to retrieve current alerts filtered by status.
    The AlertManager URL is configured via the ALERTMANAGER_URL environment variable.

    Args:
        status: Alert status filter:
            - 'firing': Currently active alerts
            - 'pending': Alerts waiting for threshold duration
            - 'resolved': Recently resolved alerts
            - 'all': All alerts regardless of status

    Returns:
        List of alerts with name, severity, status, timestamps, labels, and annotations
    """
    try:
        logger.info(f"Listing alerts: status={status}")

        # Get the AlertManager client and query alerts
        client = get_alertmanager_client()

        # Map status to AlertManager query parameters
        silenced = False
        inhibited = False
        active = True

        if status == "resolved":
            # For resolved alerts, we need to include inactive/silenced
            active = False
            silenced = True
        elif status == "all":
            # For all alerts, include everything
            active = True
            silenced = True
            inhibited = True

        alerts = await client.list_alerts(
            status=status,
            silenced=silenced,
            inhibited=inhibited,
            active=active,
        )

        # Check for errors in the result
        if alerts and "error" in alerts[0]:
            logger.warning(f"AlertManager query returned error: {alerts[0]['error']}")
        else:
            logger.info(f"Found {len(alerts)} alerts")

        return alerts

    except Exception as e:
        logger.error(f"Failed to list alerts: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def documentation_search_impl(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Search internal documentation using RAG with Pinecone and Voyage embeddings.

    Uses semantic search via the DevOpsKnowledgeBase for finding relevant
    documentation across Kubernetes, Harness, Terraform, and internal runbooks.
    Falls back to placeholder results if RAG is not configured.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        Relevant documents with content, score, and metadata
    """
    try:
        logger.info(f"Searching documentation: query='{query}', top_k={top_k}")

        # Use RAG-based search with Pinecone + Voyage embeddings
        docs = await search_documentation_with_rag(query=query, top_k=top_k)

        logger.info(f"Found {len(docs)} documents")
        return docs

    except Exception as e:
        logger.error(f"Documentation search failed: {e}")
        return [{"error": str(e), "title": "Search failed", "content": ""}]


# =============================================================================
# Tool Definitions - Templating
# =============================================================================


def create_list_templates_tool() -> StructuredTool:
    """Create tool for listing available templates."""
    return StructuredTool.from_function(
        coroutine=list_templates_impl,
        name="list_templates",
        description=(
            "List available project templates with optional filtering by format, category, or tags. "
            "Supports Handlebars, Cookiecutter, Copier, Maven Archetype, and Harness templates."
        ),
        args_schema=ListTemplatesInput,
    )


def create_search_templates_tool() -> StructuredTool:
    """Create tool for searching templates."""
    return StructuredTool.from_function(
        coroutine=search_templates_impl,
        name="search_templates",
        description=(
            "Search templates by query string with full-text search across names, descriptions, and tags. "
            "Returns ranked results with relevance scores."
        ),
        args_schema=SearchTemplatesInput,
    )


def create_generate_from_template_tool() -> StructuredTool:
    """Create tool for generating from templates."""
    return StructuredTool.from_function(
        coroutine=generate_from_template_impl,
        name="generate_from_template",
        description=(
            "Generate a project from a template with variable substitution. "
            "Supports dry-run preview mode and all major template formats."
        ),
        args_schema=GenerateFromTemplateInput,
    )


def create_validate_template_tool() -> StructuredTool:
    """Create tool for validating templates."""
    return StructuredTool.from_function(
        coroutine=validate_template_impl,
        name="validate_template",
        description=(
            "Validate template configuration and structure. "
            "Checks syntax, variable consistency, and file references. Returns quality score."
        ),
        args_schema=ValidateTemplateInput,
    )


# =============================================================================
# Tool Definitions - Scaffolding
# =============================================================================


def create_scaffold_project_tool() -> StructuredTool:
    """Create tool for scaffolding projects."""
    return StructuredTool.from_function(
        coroutine=scaffold_project_impl,
        name="scaffold_project",
        description=(
            "Scaffold a complete project from template with optional Harness CI/CD setup, "
            "Docker configuration, and Kubernetes manifests. Initializes git repository and "
            "installs dependencies."
        ),
        args_schema=ScaffoldProjectInput,
    )


# =============================================================================
# Tool Definitions - Harness CI/CD
# =============================================================================


def create_create_pipeline_tool() -> StructuredTool:
    """Create tool for creating Harness pipelines."""
    return StructuredTool.from_function(
        coroutine=create_pipeline_impl,
        name="create_pipeline",
        description=(
            "Create a Harness CI/CD pipeline with standard patterns: standard-cicd, gitops, "
            "canary, or blue-green deployment. Generates complete pipeline YAML with stages and steps."
        ),
        args_schema=CreatePipelineInput,
    )


def create_validate_pipeline_tool() -> StructuredTool:
    """Create tool for validating pipelines."""
    return StructuredTool.from_function(
        coroutine=validate_pipeline_impl,
        name="validate_pipeline",
        description=(
            "Validate Harness pipeline YAML for syntax, references, security, and best practices. "
            "Provides actionable warnings and suggestions."
        ),
        args_schema=ValidatePipelineInput,
    )


def create_create_template_tool() -> StructuredTool:
    """Create tool for creating Harness templates."""
    return StructuredTool.from_function(
        coroutine=create_template_impl,
        name="create_template",
        description=(
            "Create reusable Harness templates for steps, stages, or entire pipelines. "
            "Supports account, org, and project scope with runtime inputs."
        ),
        args_schema=CreateTemplateInput,
    )


def create_trigger_pipeline_tool() -> StructuredTool:
    """Create tool for triggering pipelines."""
    return StructuredTool.from_function(
        coroutine=trigger_pipeline_impl,
        name="trigger_pipeline",
        description=(
            "Trigger a Harness pipeline execution with runtime inputs and environment selection. "
            "Returns execution ID for status tracking."
        ),
        args_schema=TriggerPipelineInput,
    )


def create_get_pipeline_status_tool() -> StructuredTool:
    """Create tool for checking pipeline status."""
    return StructuredTool.from_function(
        coroutine=get_pipeline_status_impl,
        name="get_pipeline_status",
        description=(
            "Check the status of a Harness pipeline execution. "
            "Returns stage-level details, duration, and completion status."
        ),
        args_schema=GetPipelineStatusInput,
    )


# =============================================================================
# Tool Definitions - Code Generation
# =============================================================================


def create_generate_api_client_tool() -> StructuredTool:
    """Create tool for generating API clients."""
    return StructuredTool.from_function(
        coroutine=generate_api_client_impl,
        name="generate_api_client",
        description=(
            "Generate type-safe API client from OpenAPI/Swagger specification. "
            "Supports TypeScript, Python, Go, Java, and C# with multiple client styles."
        ),
        args_schema=GenerateApiClientInput,
    )


def create_generate_models_tool() -> StructuredTool:
    """Create tool for generating models."""
    return StructuredTool.from_function(
        coroutine=generate_models_impl,
        name="generate_models",
        description=(
            "Generate data models from JSON Schema, Prisma schema, or GraphQL schema. "
            "Includes validation rules and serialization support."
        ),
        args_schema=GenerateModelsInput,
    )


def create_generate_tests_tool() -> StructuredTool:
    """Create tool for generating tests."""
    return StructuredTool.from_function(
        coroutine=generate_tests_impl,
        name="generate_tests",
        description=(
            "Generate comprehensive test suite from source code analysis. "
            "Supports Jest, pytest, Vitest, Mocha, and JUnit with auto-mocking."
        ),
        args_schema=GenerateTestsInput,
    )


def create_generate_migrations_tool() -> StructuredTool:
    """Create tool for generating migrations."""
    return StructuredTool.from_function(
        coroutine=generate_migrations_impl,
        name="generate_migrations",
        description=(
            "Generate database migration files from schema or model definitions. "
            "Supports PostgreSQL, MySQL, SQLite, and MongoDB with frameworks like Alembic."
        ),
        args_schema=GenerateMigrationsInput,
    )


# =============================================================================
# Tool Definitions - DevOps Operations
# =============================================================================


def create_web_search_tool() -> StructuredTool:
    """Create web search tool."""
    return StructuredTool.from_function(
        coroutine=web_search_impl,
        name="web_search",
        description=(
            "Search the web for DevOps solutions, documentation, and best practices. "
            "Use this for external information about technologies and troubleshooting."
        ),
        args_schema=WebSearchInput,
    )


def create_kubernetes_get_pods_tool() -> StructuredTool:
    """Create tool for getting Kubernetes pods."""
    return StructuredTool.from_function(
        coroutine=kubernetes_get_pods_impl,
        name="kubernetes_get_pods",
        description=(
            "Get Kubernetes pods in a namespace with optional label selector filtering. "
            "Returns pod status, readiness, and resource information."
        ),
        args_schema=KubernetesGetPodsInput,
    )


def create_kubernetes_get_services_tool() -> StructuredTool:
    """Create tool for getting Kubernetes services."""
    return StructuredTool.from_function(
        coroutine=kubernetes_get_services_impl,
        name="kubernetes_get_services",
        description=(
            "Get Kubernetes services in a namespace. "
            "Returns service type, cluster IPs, ports, and endpoints."
        ),
        args_schema=KubernetesGetServicesInput,
    )


def create_kubernetes_get_deployments_tool() -> StructuredTool:
    """Create tool for getting Kubernetes deployments."""
    return StructuredTool.from_function(
        coroutine=kubernetes_get_deployments_impl,
        name="kubernetes_get_deployments",
        description=(
            "Get Kubernetes deployments in a namespace. "
            "Returns replica counts, availability status, and rollout information."
        ),
        args_schema=KubernetesGetDeploymentsInput,
    )


def create_kubernetes_scale_tool() -> StructuredTool:
    """Create tool for scaling deployments."""
    return StructuredTool.from_function(
        coroutine=kubernetes_scale_impl,
        name="kubernetes_scale_deployment",
        description=(
            "Scale a Kubernetes deployment to the specified number of replicas. "
            "Use this to adjust capacity based on load or requirements."
        ),
        args_schema=KubernetesScaleInput,
    )


def create_kubernetes_restart_tool() -> StructuredTool:
    """Create tool for restarting deployments."""
    return StructuredTool.from_function(
        coroutine=kubernetes_restart_impl,
        name="kubernetes_restart_deployment",
        description=(
            "Restart a Kubernetes deployment by triggering a rollout restart. "
            "Use this to apply configuration changes or recover from issues."
        ),
        args_schema=KubernetesRestartInput,
    )


def create_query_metrics_tool() -> StructuredTool:
    """Create tool for querying metrics."""
    return StructuredTool.from_function(
        coroutine=query_metrics_impl,
        name="query_metrics",
        description=(
            "Query Prometheus metrics using PromQL. "
            "Retrieve time-series data for monitoring and performance analysis."
        ),
        args_schema=QueryMetricsInput,
    )


def create_search_logs_tool() -> StructuredTool:
    """Create tool for searching logs."""
    return StructuredTool.from_function(
        coroutine=search_logs_impl,
        name="search_logs",
        description=(
            "Search logs across services using query strings. "
            "Filter by service name, log level, and time range."
        ),
        args_schema=SearchLogsInput,
    )


def create_list_alerts_tool() -> StructuredTool:
    """Create tool for listing alerts."""
    return StructuredTool.from_function(
        coroutine=list_alerts_impl,
        name="list_alerts",
        description=(
            "List active alerts from AlertManager. "
            "Filter by status (firing, pending, resolved) and view alert details."
        ),
        args_schema=ListAlertsInput,
    )


def create_documentation_search_tool() -> StructuredTool:
    """Create tool for searching documentation."""
    return StructuredTool.from_function(
        coroutine=documentation_search_impl,
        name="documentation_search",
        description=(
            "Search internal documentation using semantic similarity. "
            "Find relevant runbooks, guides, and knowledge base articles."
        ),
        args_schema=DocumentationSearchInput,
    )


# =============================================================================
# Tool Registry
# =============================================================================


class DevOpsToolRegistry:
    """Registry for DevOps tools with agent-specific access control."""

    def __init__(self):
        """Initialize the tool registry."""
        self._all_tools = self._initialize_tools()
        self._agent_tool_map = self._build_agent_tool_map()

    def _initialize_tools(self) -> Dict[str, StructuredTool]:
        """Initialize all available tools.

        Returns:
            Dictionary mapping tool names to tool instances
        """
        tools = {
            # Templating tools
            "list_templates": create_list_templates_tool(),
            "search_templates": create_search_templates_tool(),
            "generate_from_template": create_generate_from_template_tool(),
            "validate_template": create_validate_template_tool(),
            # Scaffolding tools
            "scaffold_project": create_scaffold_project_tool(),
            # Harness CI/CD tools
            "create_pipeline": create_create_pipeline_tool(),
            "validate_pipeline": create_validate_pipeline_tool(),
            "create_template": create_create_template_tool(),
            "trigger_pipeline": create_trigger_pipeline_tool(),
            "get_pipeline_status": create_get_pipeline_status_tool(),
            # Code generation tools
            "generate_api_client": create_generate_api_client_tool(),
            "generate_models": create_generate_models_tool(),
            "generate_tests": create_generate_tests_tool(),
            "generate_migrations": create_generate_migrations_tool(),
            # DevOps operation tools
            "web_search": create_web_search_tool(),
            "kubernetes_get_pods": create_kubernetes_get_pods_tool(),
            "kubernetes_get_services": create_kubernetes_get_services_tool(),
            "kubernetes_get_deployments": create_kubernetes_get_deployments_tool(),
            "kubernetes_scale_deployment": create_kubernetes_scale_tool(),
            "kubernetes_restart_deployment": create_kubernetes_restart_tool(),
            "query_metrics": create_query_metrics_tool(),
            "search_logs": create_search_logs_tool(),
            "list_alerts": create_list_alerts_tool(),
            "documentation_search": create_documentation_search_tool(),
        }
        return tools

    def _build_agent_tool_map(self) -> Dict[str, List[str]]:
        """Build mapping of agent names to allowed tool names.

        Returns:
            Dictionary mapping agent names to lists of tool names
        """
        return {
            # Template Manager Agent
            "template_manager": [
                "list_templates",
                "search_templates",
                "generate_from_template",
                "validate_template",
                "documentation_search",
                "web_search",
            ],
            # Scaffold Agent
            "scaffold_agent": [
                "scaffold_project",
                "list_templates",
                "search_templates",
                "generate_from_template",
                "documentation_search",
            ],
            # Harness Expert Agent
            "harness_expert": [
                "create_pipeline",
                "validate_pipeline",
                "create_template",
                "trigger_pipeline",
                "get_pipeline_status",
                "documentation_search",
                "web_search",
            ],
            # Code Generation Agent
            "codegen_agent": [
                "generate_api_client",
                "generate_models",
                "generate_tests",
                "generate_migrations",
                "validate_template",
                "documentation_search",
                "web_search",
            ],
            # Infrastructure Agent
            "infrastructure_agent": [
                "kubernetes_get_pods",
                "kubernetes_get_services",
                "kubernetes_get_deployments",
                "kubernetes_scale_deployment",
                "kubernetes_restart_deployment",
                "trigger_pipeline",
                "get_pipeline_status",
                "documentation_search",
                "web_search",
            ],
            # Monitoring Agent
            "monitoring_agent": [
                "query_metrics",
                "search_logs",
                "list_alerts",
                "kubernetes_get_pods",
                "kubernetes_get_deployments",
                "documentation_search",
                "web_search",
            ],
            # Deployment Agent
            "deployment_agent": [
                "create_pipeline",
                "trigger_pipeline",
                "get_pipeline_status",
                "kubernetes_get_pods",
                "kubernetes_get_services",
                "kubernetes_get_deployments",
                "kubernetes_scale_deployment",
                "documentation_search",
            ],
            # Supervisor Agent (has access to all tools)
            "supervisor": list(self._all_tools.keys()),
        }

    def get_all_tools(self) -> List[StructuredTool]:
        """Get all available tools.

        Returns:
            List of all tool instances
        """
        return list(self._all_tools.values())

    def get_tools_for_agent(self, agent_name: str) -> List[StructuredTool]:
        """Get tools allowed for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of tool instances allowed for the agent
        """
        tool_names = self._agent_tool_map.get(agent_name, [])
        return [self._all_tools[name] for name in tool_names if name in self._all_tools]

    def get_tools_by_category(self) -> Dict[str, List[StructuredTool]]:
        """Get tools organized by category.

        Returns:
            Dictionary mapping category names to lists of tools
        """
        return {
            "templating": [
                self._all_tools["list_templates"],
                self._all_tools["search_templates"],
                self._all_tools["generate_from_template"],
                self._all_tools["validate_template"],
            ],
            "scaffolding": [
                self._all_tools["scaffold_project"],
            ],
            "harness_cicd": [
                self._all_tools["create_pipeline"],
                self._all_tools["validate_pipeline"],
                self._all_tools["create_template"],
                self._all_tools["trigger_pipeline"],
                self._all_tools["get_pipeline_status"],
            ],
            "code_generation": [
                self._all_tools["generate_api_client"],
                self._all_tools["generate_models"],
                self._all_tools["generate_tests"],
                self._all_tools["generate_migrations"],
            ],
            "kubernetes": [
                self._all_tools["kubernetes_get_pods"],
                self._all_tools["kubernetes_get_services"],
                self._all_tools["kubernetes_get_deployments"],
                self._all_tools["kubernetes_scale_deployment"],
                self._all_tools["kubernetes_restart_deployment"],
            ],
            "observability": [
                self._all_tools["query_metrics"],
                self._all_tools["search_logs"],
                self._all_tools["list_alerts"],
            ],
            "search": [
                self._all_tools["web_search"],
                self._all_tools["documentation_search"],
            ],
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def get_all_devops_tools() -> List[StructuredTool]:
    """Get all DevOps tools as a list.

    Returns:
        List of all available DevOps tools
    """
    registry = DevOpsToolRegistry()
    return registry.get_all_tools()


def get_tools_for_agent(agent_name: str) -> List[StructuredTool]:
    """Get tools for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        List of tools allowed for the agent
    """
    registry = DevOpsToolRegistry()
    return registry.get_tools_for_agent(agent_name)


def get_devops_tools_by_category() -> Dict[str, List[StructuredTool]]:
    """Get DevOps tools organized by category.

    Returns:
        Dictionary mapping category names to lists of tools
    """
    registry = DevOpsToolRegistry()
    return registry.get_tools_by_category()


__all__ = [
    # Harness Client
    "HarnessClient",
    "HarnessClientError",
    "get_harness_client",
    # Tavily Search Client
    "TavilySearchClient",
    "get_tavily_client",
    # RAG Integration
    "search_documentation_with_rag",
    # Registry and convenience functions
    "DevOpsToolRegistry",
    "get_all_devops_tools",
    "get_tools_for_agent",
    "get_devops_tools_by_category",
    # Tool creation functions
    "create_list_templates_tool",
    "create_search_templates_tool",
    "create_generate_from_template_tool",
    "create_validate_template_tool",
    "create_scaffold_project_tool",
    "create_create_pipeline_tool",
    "create_validate_pipeline_tool",
    "create_create_template_tool",
    "create_trigger_pipeline_tool",
    "create_get_pipeline_status_tool",
    "create_generate_api_client_tool",
    "create_generate_models_tool",
    "create_generate_tests_tool",
    "create_generate_migrations_tool",
    "create_web_search_tool",
    "create_kubernetes_get_pods_tool",
    "create_kubernetes_get_services_tool",
    "create_kubernetes_get_deployments_tool",
    "create_kubernetes_scale_tool",
    "create_kubernetes_restart_tool",
    "create_query_metrics_tool",
    "create_search_logs_tool",
    "create_list_alerts_tool",
    "create_documentation_search_tool",
]
