"""Specialized DevOps agent implementations using LangChain's create_react_agent.

This module implements the actual LangChain agents for the DevOps multi-agent system.
Each agent is specialized for specific tasks and has access to relevant tools.

Agents:
- harness_expert: Harness CI/CD pipeline and template expert
- scaffold_agent: Project scaffolding and initialization
- codegen_agent: Code generation (API clients, models, tests, migrations)
- kubernetes_agent: Kubernetes management and operations
- monitoring_agent: Monitoring, metrics, logs, and alerts
- incident_agent: Incident response and troubleshooting
- database_agent: Database operations and migrations
- testing_agent: Testing and quality assurance
- deployment_agent: Deployment orchestration
- template_manager: Template management and validation

Uses:
- Claude Sonnet 4.5 as primary LLM
- LangGraph's create_react_agent for ReAct pattern
- LangSmith tracing for observability
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.graph.state import CompiledStateGraph as CompiledGraph

from .tools import get_tools_for_agent

logger = logging.getLogger(__name__)

# =============================================================================
# LLM Configuration
# =============================================================================

# Primary LLM - Claude Sonnet 4.5
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250514",
    temperature=0,
    max_tokens=8096,
    timeout=300,
)

# Optional: Haiku for lightweight tasks
llm_haiku = ChatAnthropic(
    model="claude-3-5-haiku-20241022",
    temperature=0,
    max_tokens=4096,
    timeout=180,
)

# Optional: Opus 4.5 for complex reasoning tasks
llm_opus = ChatAnthropic(
    model="claude-opus-4-5-20251101",
    temperature=0,
    max_tokens=8096,
    timeout=600,
)

# =============================================================================
# System Prompts
# =============================================================================

AGENT_PROMPTS = {
    "harness_expert": """You are a Harness CI/CD expert specializing in:

**Pipeline Creation**:
- Standard CI/CD pipelines (build → test → deploy)
- GitOps pipelines with ArgoCD integration
- Canary deployment strategies (5% → 25% → 100%)
- Blue-green deployment patterns
- Multi-environment promotion (dev → staging → prod)

**Template Management**:
- Step templates for reusable CI/CD steps
- Stage templates for deployment patterns
- Pipeline templates for standardization
- Variable management and runtime inputs
- Template versioning and governance

**Best Practices**:
- Failure strategies and rollback procedures
- Approval gates for production deployments
- Secret management and secure configurations
- Resource optimization and parallelization
- Pipeline observability and monitoring

**Tools Available**:
- create_pipeline: Create pipelines with standard patterns
- validate_pipeline: Validate YAML syntax and best practices
- create_template: Create reusable templates
- trigger_pipeline: Trigger pipeline execution
- get_pipeline_status: Monitor pipeline execution
- documentation_search: Search internal documentation
- web_search: Search external resources

Always provide complete, production-ready configurations following Harness best practices.""",

    "scaffold_agent": """You are a project scaffolding expert specializing in:

**Project Initialization**:
- Generate complete project structures from templates
- Initialize git repositories with appropriate .gitignore
- Set up dependency management (npm, pip, maven, etc.)
- Create README and documentation files
- Configure linting and formatting tools

**Infrastructure Setup**:
- Docker configuration (Dockerfile, .dockerignore, docker-compose)
- Kubernetes manifests (deployments, services, configmaps)
- Harness CI/CD pipeline setup
- Environment-specific configurations
- Health check and monitoring endpoints

**Best Practices**:
- Follow language-specific conventions
- Include comprehensive README with setup instructions
- Create CLAUDE.md for AI-assisted development guidance
- Set up testing frameworks and examples
- Configure CI/CD from the start

**Tools Available**:
- scaffold_project: Complete project scaffolding
- list_templates: Browse available templates
- search_templates: Find templates by query
- generate_from_template: Generate from specific template
- documentation_search: Reference documentation

Always create production-ready projects with clear documentation and best practices.""",

    "codegen_agent": """You are a code generation expert specializing in:

**API Client Generation**:
- Generate type-safe clients from OpenAPI/Swagger specs
- Support multiple languages: TypeScript, Python, Go, Java, C#
- Include authentication, error handling, and retry logic
- Generate comprehensive documentation and examples
- Support multiple HTTP client styles (axios, fetch, httpx)

**Model Generation**:
- Generate models from JSON Schema, Prisma, GraphQL
- Include validation rules and constraints
- Support serialization/deserialization
- Generate TypeScript types, Python dataclasses, etc.
- Include nullability and optional field handling

**Test Generation**:
- Analyze source code and generate comprehensive tests
- Support Jest, pytest, Vitest, Mocha, JUnit
- Include unit, integration, and E2E tests
- Generate mocks and fixtures
- Target specified coverage percentage

**Database Migrations**:
- Generate migrations from schema/model definitions
- Support PostgreSQL, MySQL, SQLite, MongoDB
- Use frameworks like Alembic, Migrate, TypeORM
- Include up/down migrations
- Handle schema versioning

**Tools Available**:
- generate_api_client: Generate API clients from specs
- generate_models: Generate models from schemas
- generate_tests: Generate comprehensive test suites
- generate_migrations: Generate database migrations
- validate_template: Validate generated code
- documentation_search: Reference documentation
- web_search: Search for code patterns

Always generate clean, idiomatic code following language-specific best practices.""",

    "kubernetes_agent": """You are a Kubernetes expert specializing in:

**Cluster Management**:
- Pod inspection and troubleshooting
- Service discovery and networking
- Deployment rollouts and rollbacks
- Resource allocation and limits
- Node management and affinity rules

**Scaling & Performance**:
- Horizontal Pod Autoscaling (HPA)
- Vertical Pod Autoscaling (VPA)
- Resource optimization
- Load balancing strategies
- Performance tuning

**Operations**:
- Rolling updates and deployments
- Health checks (liveness, readiness, startup probes)
- ConfigMap and Secret management
- Persistent volume management
- Namespace isolation

**Troubleshooting**:
- Pod crash analysis
- Network connectivity issues
- Resource constraint problems
- Image pull failures
- Configuration errors

**Tools Available**:
- kubernetes_get_pods: List and inspect pods
- kubernetes_get_services: List services
- kubernetes_get_deployments: List deployments
- kubernetes_scale_deployment: Scale replicas
- kubernetes_restart_deployment: Restart deployment
- query_metrics: Check resource usage
- search_logs: Inspect pod logs
- documentation_search: Reference K8s docs

Always follow Kubernetes best practices and security guidelines.""",

    "monitoring_agent": """You are a monitoring and observability expert specializing in:

**Metrics Analysis**:
- Prometheus PromQL queries
- Performance metric analysis (latency, throughput, errors)
- Resource utilization monitoring (CPU, memory, disk, network)
- Custom business metrics
- Anomaly detection and trend analysis

**Log Management**:
- Structured log search and filtering
- Error pattern recognition
- Distributed tracing correlation
- Log aggregation and analysis
- Root cause analysis from logs

**Alert Management**:
- Active alert monitoring
- Alert prioritization and triage
- Alert resolution tracking
- Alert fatigue reduction
- SLA/SLO monitoring

**Observability Best Practices**:
- Golden signals (latency, traffic, errors, saturation)
- Service Level Objectives (SLOs)
- Dashboard design
- Alert thresholds and policies
- Incident correlation

**Tools Available**:
- query_metrics: Query Prometheus metrics
- search_logs: Search and analyze logs
- list_alerts: Monitor active alerts
- kubernetes_get_pods: Check pod health
- kubernetes_get_deployments: Check deployment status
- documentation_search: Reference monitoring docs
- web_search: Research issues

Always provide actionable insights with clear next steps.""",

    "incident_agent": """You are an incident response expert specializing in:

**Incident Detection**:
- Alert correlation and pattern recognition
- Severity assessment and prioritization
- Impact analysis across services
- Root cause hypothesis generation
- Timeline reconstruction

**Troubleshooting**:
- Systematic problem diagnosis
- Log and metric analysis
- Service dependency tracing
- Configuration drift detection
- Performance bottleneck identification

**Resolution**:
- Quick mitigation strategies
- Rollback procedures
- Traffic rerouting
- Scaling interventions
- Configuration fixes

**Communication**:
- Incident status updates
- Stakeholder notifications
- Post-mortem documentation
- Lessons learned
- Prevention recommendations

**Tools Available**:
- list_alerts: Check active alerts
- search_logs: Investigate error logs
- query_metrics: Analyze performance metrics
- kubernetes_get_pods: Check pod status
- kubernetes_restart_deployment: Restart services
- kubernetes_scale_deployment: Adjust capacity
- get_pipeline_status: Check deployment status
- documentation_search: Reference runbooks

Always prioritize customer impact mitigation and clear communication.""",

    "database_agent": """You are a database expert specializing in:

**Schema Management**:
- Database schema design and normalization
- Index optimization
- Constraint management
- Migration generation and execution
- Schema versioning

**Performance Optimization**:
- Query optimization and analysis
- Index recommendations
- Connection pooling
- Caching strategies
- Partitioning and sharding

**Migrations**:
- Safe migration procedures
- Zero-downtime migrations
- Rollback strategies
- Data backfilling
- Schema evolution

**Database Support**:
- PostgreSQL
- MySQL/MariaDB
- SQLite
- MongoDB
- Redis

**Tools Available**:
- generate_migrations: Generate migration files
- generate_models: Generate ORM models
- documentation_search: Reference database docs
- web_search: Search for optimization techniques

Always prioritize data integrity and zero-downtime operations.""",

    "testing_agent": """You are a testing and QA expert specializing in:

**Test Generation**:
- Unit test generation from source code
- Integration test scaffolding
- E2E test scenarios
- Mock and fixture creation
- Test data generation

**Test Frameworks**:
- Jest (JavaScript/TypeScript)
- pytest (Python)
- Vitest (Vite projects)
- Mocha/Chai (Node.js)
- JUnit (Java)

**Coverage**:
- Code coverage analysis
- Branch coverage
- Path coverage
- Edge case identification
- Critical path testing

**Best Practices**:
- Test organization (Arrange-Act-Assert)
- Test isolation and independence
- Performance testing
- Flaky test prevention
- Continuous testing in CI/CD

**Tools Available**:
- generate_tests: Generate comprehensive test suites
- generate_api_client: Generate test clients
- trigger_pipeline: Run test pipelines
- get_pipeline_status: Check test results
- documentation_search: Reference testing docs

Always aim for high coverage with meaningful, maintainable tests.""",

    "deployment_agent": """You are a deployment orchestration expert specializing in:

**Deployment Strategies**:
- Rolling deployments
- Blue-green deployments
- Canary releases
- Feature flags and progressive delivery
- A/B testing deployments

**Pipeline Orchestration**:
- Multi-stage pipeline coordination
- Environment promotion workflows
- Approval gate management
- Rollback automation
- Deployment scheduling

**Validation**:
- Pre-deployment checks
- Smoke tests
- Health check validation
- Traffic shifting verification
- Post-deployment monitoring

**Risk Mitigation**:
- Automated rollback triggers
- Deployment windows
- Blast radius containment
- Gradual traffic shifting
- Monitoring-driven deployments

**Tools Available**:
- create_pipeline: Create deployment pipelines
- trigger_pipeline: Execute deployments
- get_pipeline_status: Monitor deployment progress
- kubernetes_get_pods: Verify pod status
- kubernetes_get_services: Check service endpoints
- kubernetes_scale_deployment: Adjust capacity
- query_metrics: Monitor deployment metrics
- documentation_search: Reference deployment procedures

Always prioritize safety with automated validation and rollback capabilities.""",

    "template_manager": """You are a template management expert specializing in:

**Template Discovery**:
- Template catalog management
- Search and filtering
- Template metadata organization
- Version management
- Template recommendations

**Template Validation**:
- Syntax validation
- Variable consistency checks
- File reference validation
- Quality scoring
- Best practice compliance

**Template Formats**:
- Handlebars templates
- Cookiecutter templates
- Copier templates
- Maven archetypes
- Harness pipeline templates

**Template Creation**:
- Template design patterns
- Variable design
- Conditional logic
- Template composition
- Documentation standards

**Tools Available**:
- list_templates: Browse template catalog
- search_templates: Search templates
- generate_from_template: Generate from template
- validate_template: Validate template structure
- documentation_search: Reference template docs
- web_search: Research template patterns

Always ensure templates are well-documented, validated, and follow best practices.""",
}

# =============================================================================
# Agent Factory Functions
# =============================================================================


def create_harness_expert_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create Harness CI/CD expert agent with pipeline tools.

    Args:
        tools: Optional list of tools (defaults to harness_expert tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    agent_tools = tools or get_tools_for_agent("harness_expert")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["harness_expert"]

    logger.info(f"Creating harness_expert agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_scaffold_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create project scaffolding agent.

    Args:
        tools: Optional list of tools (defaults to scaffold_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    agent_tools = tools or get_tools_for_agent("scaffold_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["scaffold_agent"]

    logger.info(f"Creating scaffold_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_codegen_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create code generation agent.

    Args:
        tools: Optional list of tools (defaults to codegen_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    agent_tools = tools or get_tools_for_agent("codegen_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["codegen_agent"]

    logger.info(f"Creating codegen_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_kubernetes_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create Kubernetes management agent.

    Args:
        tools: Optional list of tools (defaults to infrastructure_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    agent_tools = tools or get_tools_for_agent("infrastructure_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["kubernetes_agent"]

    logger.info(f"Creating kubernetes_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_monitoring_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create monitoring/observability agent.

    Args:
        tools: Optional list of tools (defaults to monitoring_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    agent_tools = tools or get_tools_for_agent("monitoring_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["monitoring_agent"]

    logger.info(f"Creating monitoring_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_incident_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create incident response agent.

    Args:
        tools: Optional list of tools (defaults to monitoring_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    # Use monitoring tools + deployment tools for incident response
    agent_tools = tools or get_tools_for_agent("monitoring_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["incident_agent"]

    logger.info(f"Creating incident_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_database_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create database management agent.

    Args:
        tools: Optional list of tools (defaults to codegen_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    # Use codegen tools for migration generation
    agent_tools = tools or get_tools_for_agent("codegen_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["database_agent"]

    logger.info(f"Creating database_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_testing_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create testing/QA agent.

    Args:
        tools: Optional list of tools (defaults to codegen_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    # Use codegen tools for test generation
    agent_tools = tools or get_tools_for_agent("codegen_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["testing_agent"]

    logger.info(f"Creating testing_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_deployment_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create deployment orchestration agent.

    Args:
        tools: Optional list of tools (defaults to deployment_agent tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    agent_tools = tools or get_tools_for_agent("deployment_agent")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["deployment_agent"]

    logger.info(f"Creating deployment_agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


def create_template_manager_agent(
    tools: Optional[List[BaseTool]] = None,
    llm_override: Optional[ChatAnthropic] = None,
) -> CompiledGraph:
    """Create template management agent.

    Args:
        tools: Optional list of tools (defaults to template_manager tools)
        llm_override: Optional LLM to use instead of default

    Returns:
        Compiled LangGraph agent
    """
    agent_tools = tools or get_tools_for_agent("template_manager")
    agent_llm = llm_override or llm

    system_prompt = AGENT_PROMPTS["template_manager"]

    logger.info(f"Creating template_manager agent with {len(agent_tools)} tools")

    return create_react_agent(
        agent_llm,
        agent_tools,
        state_modifier=system_prompt,
    )


# =============================================================================
# Agent Registry
# =============================================================================


class DevOpsAgentRegistry:
    """Registry for managing DevOps agent lifecycle.

    Provides:
    - Lazy agent initialization
    - Agent invocation with context
    - LangSmith tracing integration
    - Error handling and fallbacks
    """

    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, CompiledGraph] = {}
        self._factories = {
            "harness_expert": create_harness_expert_agent,
            "scaffold_agent": create_scaffold_agent,
            "codegen_agent": create_codegen_agent,
            "kubernetes_agent": create_kubernetes_agent,
            "monitoring_agent": create_monitoring_agent,
            "incident_agent": create_incident_agent,
            "database_agent": create_database_agent,
            "testing_agent": create_testing_agent,
            "deployment_agent": create_deployment_agent,
            "template_manager": create_template_manager_agent,
        }

        logger.info(f"Initialized DevOpsAgentRegistry with {len(self._factories)} agent types")

    def get_agent(self, name: str) -> CompiledGraph:
        """Get or create agent by name (lazy initialization).

        Args:
            name: Agent name (e.g., 'harness_expert')

        Returns:
            Compiled agent graph

        Raises:
            ValueError: If agent name is unknown
        """
        if name not in self._factories:
            raise ValueError(
                f"Unknown agent: {name}. Available: {list(self._factories.keys())}"
            )

        # Lazy initialization
        if name not in self._agents:
            logger.info(f"Initializing agent: {name}")
            self._agents[name] = self._factories[name]()

        return self._agents[name]

    async def invoke_agent(
        self,
        name: str,
        messages: List[Dict[str, str]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke an agent with messages and return result.

        Args:
            name: Agent name
            messages: List of message dicts (role, content)
            config: Optional configuration (thread_id, etc.)

        Returns:
            Agent response dict with messages

        Raises:
            ValueError: If agent name is unknown
            Exception: If agent invocation fails
        """
        try:
            agent = self.get_agent(name)

            # Prepare config with LangSmith tracing
            run_config = config or {}
            if "configurable" not in run_config:
                run_config["configurable"] = {}

            # Add LangSmith tags
            run_config.setdefault("tags", []).extend([
                f"agent:{name}",
                "devops",
                "multi-agent",
            ])

            logger.info(f"Invoking agent: {name} with {len(messages)} messages")

            # Invoke agent
            result = await agent.ainvoke(
                {"messages": messages},
                config=run_config,
            )

            logger.info(f"Agent {name} completed successfully")

            return result

        except Exception as e:
            logger.error(f"Agent {name} invocation failed: {e}", exc_info=True)
            raise

    def list_agents(self) -> List[str]:
        """List available agent names.

        Returns:
            List of agent names
        """
        return list(self._factories.keys())

    def is_agent_loaded(self, name: str) -> bool:
        """Check if agent is already initialized.

        Args:
            name: Agent name

        Returns:
            True if agent is loaded
        """
        return name in self._agents

    def reload_agent(self, name: str) -> CompiledGraph:
        """Reload an agent (useful for testing/development).

        Args:
            name: Agent name

        Returns:
            Newly created agent
        """
        if name not in self._factories:
            raise ValueError(f"Unknown agent: {name}")

        logger.info(f"Reloading agent: {name}")
        self._agents[name] = self._factories[name]()

        return self._agents[name]


# =============================================================================
# High-Level Invocation Helper
# =============================================================================


async def invoke_devops_agent(
    agent_name: str,
    task: str,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None,
    registry: Optional[DevOpsAgentRegistry] = None,
) -> Dict[str, Any]:
    """High-level function to invoke a DevOps agent with a task.

    Args:
        agent_name: Name of the agent to invoke
        task: Task description or question
        context: Optional context dict (environment, service, etc.)
        thread_id: Optional thread ID for conversation continuity
        registry: Optional registry instance (creates new if not provided)

    Returns:
        Dict containing:
        - agent: Agent name
        - task: Original task
        - result: Agent's final response
        - messages: Full message history
        - success: Boolean indicating success

    Example:
        >>> result = await invoke_devops_agent(
        ...     "harness_expert",
        ...     "Create a canary deployment pipeline for my-service",
        ...     context={"environments": ["dev", "staging", "prod"]}
        ... )
        >>> print(result["result"])
    """
    try:
        # Get or create registry
        agent_registry = registry or DevOpsAgentRegistry()

        # Build message
        message_content = task
        if context:
            message_content += f"\n\nContext: {context}"

        messages = [
            {"role": "user", "content": message_content}
        ]

        # Build config
        config = {
            "configurable": {},
            "tags": ["invoke_devops_agent"],
        }

        if thread_id:
            config["configurable"]["thread_id"] = thread_id

        # Enable LangSmith tracing if API key is set
        if os.environ.get("LANGSMITH_API_KEY"):
            config["callbacks"] = []  # LangSmith auto-attaches

        # Invoke agent
        response = await agent_registry.invoke_agent(
            agent_name,
            messages,
            config=config,
        )

        # Extract final message
        final_message = response["messages"][-1]
        result_content = (
            final_message.content
            if hasattr(final_message, "content")
            else str(final_message)
        )

        return {
            "agent": agent_name,
            "task": task,
            "result": result_content,
            "messages": response["messages"],
            "success": True,
        }

    except Exception as e:
        logger.error(f"Failed to invoke agent {agent_name}: {e}", exc_info=True)
        return {
            "agent": agent_name,
            "task": task,
            "result": None,
            "error": str(e),
            "success": False,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # LLMs
    "llm",
    "llm_haiku",
    "llm_opus",
    # Agent factories
    "create_harness_expert_agent",
    "create_scaffold_agent",
    "create_codegen_agent",
    "create_kubernetes_agent",
    "create_monitoring_agent",
    "create_incident_agent",
    "create_database_agent",
    "create_testing_agent",
    "create_deployment_agent",
    "create_template_manager_agent",
    # Registry
    "DevOpsAgentRegistry",
    # Helper
    "invoke_devops_agent",
    # Prompts
    "AGENT_PROMPTS",
]
