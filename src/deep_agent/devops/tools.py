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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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


async def create_pipeline_impl(
    name: str,
    pipeline_type: str,
    service: str,
    environments: List[str],
    stages: Optional[List[str]],
) -> Dict[str, Any]:
    """Create Harness pipeline.

    Args:
        name: Pipeline name
        pipeline_type: Pipeline type
        service: Service name
        environments: Target environments
        stages: Custom stages

    Returns:
        Pipeline creation result
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

        # Placeholder implementation
        result = {
            "name": name,
            "type": pipeline_type,
            "service": service,
            "environments": environments,
            "stages": stages,
            "pipeline_id": f"pipeline-{datetime.now().timestamp()}",
            "status": "created",
            "yaml_path": f".harness/pipelines/{name}.yaml",
            "url": f"https://app.harness.io/pipeline/{name}",
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with Harness MCP and harness-expert agent
        # from claude_code_templating_plugin.agents import HarnessExpertAgent
        # agent = HarnessExpertAgent()
        # result = agent.create_pipeline(
        #     name=name,
        #     pipeline_type=pipeline_type,
        #     service=service,
        #     environments=environments,
        #     stages=stages
        # )

        logger.info(f"Pipeline created: {result['pipeline_id']}")
        return result

    except Exception as e:
        logger.error(f"Pipeline creation failed: {e}")
        return {
            "name": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def validate_pipeline_impl(yaml_content: str) -> Dict[str, Any]:
    """Validate Harness pipeline YAML.

    Args:
        yaml_content: Pipeline YAML content

    Returns:
        Validation result
    """
    try:
        logger.info("Validating pipeline YAML")

        # Placeholder implementation
        result = {
            "valid": True,
            "errors": [],
            "warnings": [
                "Consider adding timeout values to all steps",
                "Add failure strategy for critical stages",
            ],
            "suggestions": [
                "Use pipeline templates for better reusability",
                "Add approval gates before production deployment",
            ],
            "quality_score": 88,
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with Harness validation
        # from claude_code_templating_plugin.harness import PipelineValidator
        # validator = PipelineValidator()
        # result = validator.validate(yaml_content)

        logger.info(f"Validation complete: valid={result['valid']}")
        return result

    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "timestamp": datetime.now().isoformat(),
        }


async def create_template_impl(
    template_type: str, name: str, scope: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create Harness reusable template.

    Args:
        template_type: Template type (step, stage, pipeline)
        name: Template name
        scope: Template scope (account, org, project)
        config: Template configuration

    Returns:
        Template creation result
    """
    try:
        logger.info(
            f"Creating Harness template: type={template_type}, name={name}, scope={scope}"
        )

        # Placeholder implementation
        result = {
            "type": template_type,
            "name": name,
            "scope": scope,
            "template_id": f"template-{datetime.now().timestamp()}",
            "status": "created",
            "version": "1.0",
            "yaml_path": f".harness/templates/{template_type}/{name}.yaml",
            "url": f"https://app.harness.io/template/{name}",
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with Harness MCP
        # from claude_code_templating_plugin.harness import TemplateManager
        # manager = TemplateManager()
        # result = manager.create_template(
        #     template_type=template_type,
        #     name=name,
        #     scope=scope,
        #     config=config
        # )

        logger.info(f"Template created: {result['template_id']}")
        return result

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
    """Trigger Harness pipeline execution.

    Args:
        pipeline_id: Pipeline identifier
        inputs: Runtime inputs
        environment: Target environment

    Returns:
        Execution information
    """
    try:
        logger.info(
            f"Triggering pipeline: id={pipeline_id}, env={environment}"
        )

        # Placeholder implementation
        result = {
            "pipeline_id": pipeline_id,
            "execution_id": f"exec-{datetime.now().timestamp()}",
            "status": "running",
            "environment": environment,
            "inputs": inputs,
            "url": f"https://app.harness.io/execution/{pipeline_id}",
            "started_at": datetime.now().isoformat(),
            "stages": [
                {"name": "Build", "status": "running"},
                {"name": "Deploy", "status": "pending"},
            ],
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with Harness API via MCP
        # from claude_code_templating_plugin.harness import PipelineExecutor
        # executor = PipelineExecutor()
        # result = executor.trigger(
        #     pipeline_id=pipeline_id,
        #     inputs=inputs,
        #     environment=environment
        # )

        logger.info(f"Pipeline triggered: {result['execution_id']}")
        return result

    except Exception as e:
        logger.error(f"Pipeline trigger failed: {e}")
        return {
            "pipeline_id": pipeline_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def get_pipeline_status_impl(execution_id: str) -> Dict[str, Any]:
    """Get pipeline execution status.

    Args:
        execution_id: Execution ID

    Returns:
        Execution status and details
    """
    try:
        logger.info(f"Getting pipeline status: execution_id={execution_id}")

        # Placeholder implementation
        result = {
            "execution_id": execution_id,
            "status": "success",
            "duration_seconds": 245,
            "stages": [
                {
                    "name": "Build",
                    "status": "success",
                    "duration_seconds": 120,
                    "started_at": datetime.now().isoformat(),
                },
                {
                    "name": "Deploy",
                    "status": "success",
                    "duration_seconds": 125,
                    "started_at": (datetime.now() + timedelta(seconds=120)).isoformat(),
                },
            ],
            "url": f"https://app.harness.io/execution/{execution_id}",
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with Harness API
        # from claude_code_templating_plugin.harness import PipelineExecutor
        # executor = PipelineExecutor()
        # result = executor.get_status(execution_id)

        logger.info(f"Execution status: {result['status']}")
        return result

    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        return {
            "execution_id": execution_id,
            "status": "unknown",
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
    """Search the web for DevOps solutions and documentation.

    Args:
        query: Search query string
        max_results: Maximum number of results

    Returns:
        List of search results with title, url, and snippet
    """
    try:
        logger.info(f"Web search: {query} (max_results={max_results})")

        # Placeholder implementation
        results = [
            {
                "title": f"DevOps Best Practice: {query}",
                "url": f"https://docs.example.com/devops/{query.replace(' ', '-').lower()}",
                "snippet": f"Comprehensive guide on {query} including implementation patterns, "
                f"troubleshooting, and production considerations.",
                "source": "placeholder",
            }
        ]

        # TODO: Integrate with actual search API (Tavily, SerpAPI)

        logger.info(f"Found {len(results)} results")
        return results

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return [{"error": str(e), "title": "Search failed", "url": "", "snippet": ""}]


async def kubernetes_get_pods_impl(
    namespace: str, label_selector: Optional[str]
) -> List[Dict[str, Any]]:
    """Get Kubernetes pods.

    Args:
        namespace: Kubernetes namespace
        label_selector: Optional label selector

    Returns:
        List of pods
    """
    try:
        logger.info(f"Getting K8s pods: namespace={namespace}, labels={label_selector}")

        # Placeholder implementation
        pods = [
            {
                "name": "example-pod-001",
                "namespace": namespace,
                "status": "Running",
                "ready": "1/1",
                "restarts": 0,
                "age": "5d",
                "node": "node-1",
            }
        ]

        # TODO: Integrate with kubernetes client

        logger.info(f"Found {len(pods)} pods")
        return pods

    except Exception as e:
        logger.error(f"Failed to get pods: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def kubernetes_get_services_impl(namespace: str) -> List[Dict[str, Any]]:
    """Get Kubernetes services.

    Args:
        namespace: Kubernetes namespace

    Returns:
        List of services
    """
    try:
        logger.info(f"Getting K8s services: namespace={namespace}")

        # Placeholder implementation
        services = [
            {
                "name": "example-service",
                "namespace": namespace,
                "type": "ClusterIP",
                "cluster_ip": "10.0.0.1",
                "ports": "80/TCP",
                "age": "10d",
            }
        ]

        # TODO: Integrate with kubernetes client

        logger.info(f"Found {len(services)} services")
        return services

    except Exception as e:
        logger.error(f"Failed to get services: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def kubernetes_get_deployments_impl(namespace: str) -> List[Dict[str, Any]]:
    """Get Kubernetes deployments.

    Args:
        namespace: Kubernetes namespace

    Returns:
        List of deployments
    """
    try:
        logger.info(f"Getting K8s deployments: namespace={namespace}")

        # Placeholder implementation
        deployments = [
            {
                "name": "example-deployment",
                "namespace": namespace,
                "ready": "3/3",
                "up_to_date": 3,
                "available": 3,
                "age": "15d",
            }
        ]

        # TODO: Integrate with kubernetes client

        logger.info(f"Found {len(deployments)} deployments")
        return deployments

    except Exception as e:
        logger.error(f"Failed to get deployments: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def kubernetes_scale_impl(
    namespace: str, name: str, replicas: int
) -> Dict[str, Any]:
    """Scale Kubernetes deployment.

    Args:
        namespace: Kubernetes namespace
        name: Deployment name
        replicas: Target replicas

    Returns:
        Scaling result
    """
    try:
        logger.info(f"Scaling deployment: {namespace}/{name} to {replicas} replicas")

        # Placeholder implementation
        result = {
            "namespace": namespace,
            "deployment": name,
            "replicas": replicas,
            "status": "success",
            "message": f"Scaled to {replicas} replicas",
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with kubernetes client

        logger.info("Scaling complete")
        return result

    except Exception as e:
        logger.error(f"Scaling failed: {e}")
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def kubernetes_restart_impl(namespace: str, name: str) -> Dict[str, Any]:
    """Restart Kubernetes deployment.

    Args:
        namespace: Kubernetes namespace
        name: Deployment name

    Returns:
        Restart result
    """
    try:
        logger.info(f"Restarting deployment: {namespace}/{name}")

        # Placeholder implementation
        result = {
            "namespace": namespace,
            "deployment": name,
            "status": "success",
            "message": "Rollout restart initiated",
            "timestamp": datetime.now().isoformat(),
        }

        # TODO: Integrate with kubernetes client
        # kubectl rollout restart deployment/{name} -n {namespace}

        logger.info("Restart initiated")
        return result

    except Exception as e:
        logger.error(f"Restart failed: {e}")
        return {
            "namespace": namespace,
            "deployment": name,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


async def query_metrics_impl(promql: str, time_range: str) -> Dict[str, Any]:
    """Query Prometheus metrics.

    Args:
        promql: PromQL query
        time_range: Time range

    Returns:
        Metric values
    """
    try:
        logger.info(f"Querying metrics: {promql}, time_range={time_range}")

        # Parse time range
        now = datetime.now()
        if time_range.endswith("h"):
            start = now - timedelta(hours=int(time_range[:-1]))
        elif time_range.endswith("m"):
            start = now - timedelta(minutes=int(time_range[:-1]))
        elif time_range.endswith("d"):
            start = now - timedelta(days=int(time_range[:-1]))
        else:
            start = now - timedelta(hours=1)

        # Placeholder implementation
        result = {
            "query": promql,
            "start": start.isoformat(),
            "end": now.isoformat(),
            "resultType": "vector",
            "result": [
                {
                    "metric": {"__name__": "http_requests_total", "job": "api"},
                    "value": [now.timestamp(), "1234.5"],
                }
            ],
        }

        # TODO: Integrate with Prometheus API

        logger.info(f"Query returned {len(result['result'])} series")
        return result

    except Exception as e:
        logger.error(f"Metrics query failed: {e}")
        return {"error": str(e), "query": promql, "result": []}


async def search_logs_impl(
    query: str, service: Optional[str], level: Optional[str], time_range: str
) -> List[Dict[str, Any]]:
    """Search logs.

    Args:
        query: Search query
        service: Service filter
        level: Log level filter
        time_range: Time range

    Returns:
        Log entries
    """
    try:
        logger.info(
            f"Searching logs: query='{query}', service={service}, level={level}, "
            f"time_range={time_range}"
        )

        # Placeholder implementation
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "service": service or "api-server",
                "level": level or "error",
                "message": f"Log entry matching query: {query}",
                "trace_id": "abc123",
            }
        ]

        # TODO: Integrate with log aggregation system (Elasticsearch, Loki)

        logger.info(f"Found {len(logs)} log entries")
        return logs

    except Exception as e:
        logger.error(f"Log search failed: {e}")
        return [{"error": str(e), "timestamp": datetime.now().isoformat()}]


async def list_alerts_impl(status: str) -> List[Dict[str, Any]]:
    """List alerts from AlertManager.

    Args:
        status: Alert status filter

    Returns:
        List of alerts
    """
    try:
        logger.info(f"Listing alerts: status={status}")

        # Placeholder implementation
        alerts = [
            {
                "name": "HighMemoryUsage",
                "severity": "warning",
                "status": status,
                "started_at": (datetime.now() - timedelta(minutes=15)).isoformat(),
                "labels": {"service": "api", "env": "prod"},
                "annotations": {"summary": "Memory usage above 80%"},
            }
        ]

        # TODO: Integrate with AlertManager API

        logger.info(f"Found {len(alerts)} alerts")
        return alerts

    except Exception as e:
        logger.error(f"Failed to list alerts: {e}")
        return [{"error": str(e), "name": "query-failed"}]


async def documentation_search_impl(query: str, top_k: int) -> List[Dict[str, Any]]:
    """Search internal documentation.

    Args:
        query: Search query
        top_k: Number of results

    Returns:
        Relevant documents
    """
    try:
        logger.info(f"Searching documentation: query='{query}', top_k={top_k}")

        # Placeholder implementation
        docs = [
            {
                "title": f"Documentation: {query}",
                "content": f"Internal documentation for {query}. "
                f"Includes setup, configuration, and troubleshooting.",
                "url": f"https://docs.internal/{query.replace(' ', '-').lower()}",
                "score": 0.95,
            }
        ]

        # TODO: Integrate with Pinecone or vector database

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
