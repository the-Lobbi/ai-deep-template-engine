"""
DevOps Multi-Agent System Package

This package provides a comprehensive multi-agent workflow system for DevOps operations,
leveraging specialized agents for different operational domains.

Features:
    - Infrastructure provisioning and management (Harness, Terraform, Kubernetes)
    - CI/CD pipeline orchestration and deployment automation
    - Real-time monitoring, alerting, and incident response
    - Security scanning, compliance, and policy enforcement
    - Database operations and migration management
    - Automated testing and quality assurance
    - Rollback and recovery operations

Architecture:
    The system uses a coordinator-agent pattern where specialized agents handle specific
    domains (Harness, Kubernetes, monitoring, incidents, databases, testing, etc.) and
    collaborate through a shared state managed by LangGraph.

Main Components:
    State Management:
        - DevOpsAgentState: Central state for the DevOps workflow
        - DevOpsTaskType: Enum of supported task types

    Workflow:
        - create_devops_workflow: Factory to create the LangGraph workflow

    Agents:
        - DevOpsAgentRegistry: Registry of all specialized agents
        - invoke_devops_agent: Invoke specific agents by name
        - Specialized agent factories for Harness, scaffolding, codegen,
          Kubernetes, monitoring, incidents, databases, and testing

    Tools:
        - DevOpsToolRegistry: Comprehensive tooling for DevOps operations
        - Tool factories for Kubernetes, monitoring, deployment, and more
        - Tool retrieval by agent and category

    RAG (Retrieval-Augmented Generation):
        - DevOpsKnowledgeBase: Vector-based knowledge retrieval
        - VoyageEmbeddings: High-quality embeddings for semantic search
        - PineconeKnowledgeStore: Persistent vector storage
        - create_rag_retriever: Factory for RAG retrieval systems

    Secrets Management:
        - SecretsManager: Unified interface for secrets across providers
        - AzureKeyVaultProvider: Azure Key Vault integration
        - HarnessSecretsProvider: Harness Secrets Manager integration
        - EnvironmentProvider: Environment variable fallback
        - configure_agent_secrets: Load all required secrets
        - Automatic provider detection and caching

    Server:
        - FastAPI application for HTTP/REST API access
        - Asynchronous workflow execution
        - WebSocket support for real-time updates

Usage:
    >>> from deep_agent.devops import create_devops_workflow, DevOpsAgentState
    >>> workflow = create_devops_workflow()
    >>> initial_state = DevOpsAgentState(
    ...     task="Deploy application to production",
    ...     context={"app": "myapp", "env": "prod"}
    ... )
    >>> result = workflow.invoke(initial_state)
"""

from .state import DevOpsAgentState, DevOpsTaskType

from .workflow import create_devops_workflow

from .agents import (
    DevOpsAgentRegistry,
    invoke_devops_agent,
    create_harness_expert_agent,
    create_scaffold_agent,
    create_codegen_agent,
    create_kubernetes_agent,
    create_monitoring_agent,
    create_incident_agent,
    create_database_agent,
    create_testing_agent,
)

from .tools import (
    DevOpsToolRegistry,
    get_all_devops_tools,
    get_tools_for_agent,
)

# Observability clients
from .observability import (
    PrometheusClient,
    LokiClient,
    ElasticsearchClient,
    AlertManagerClient,
    get_prometheus_client,
    get_loki_client,
    get_elasticsearch_client,
    get_alertmanager_client,
    get_log_backend,
)

# Secrets management
from .secrets import (
    SecretsManager,
    AzureKeyVaultProvider,
    HarnessSecretsProvider,
    EnvironmentProvider,
    create_secrets_manager,
    configure_agent_secrets,
    SecretProvider,
    SecretNotFoundError,
    ProviderError,
    ProviderNotAvailableError,
)

# Optional RAG imports (requires pinecone, voyageai dependencies)
try:
    from .rag import (
        DevOpsKnowledgeBase,
        VoyageEmbeddings,
        PineconeKnowledgeStore,
        create_rag_retriever,
        Document,
        SearchResult,
    )
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False
    # Provide dummy classes for type hints
    DevOpsKnowledgeBase = None  # type: ignore
    VoyageEmbeddings = None  # type: ignore
    PineconeKnowledgeStore = None  # type: ignore
    create_rag_retriever = None  # type: ignore
    Document = None  # type: ignore
    SearchResult = None  # type: ignore

# Optional server imports (requires fastapi, uvicorn dependencies)
try:
    from .server import app, run_server
    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False
    # Provide dummy for type hints
    app = None  # type: ignore
    run_server = None  # type: ignore

__version__ = "0.1.0"

__all__ = [
    # State
    "DevOpsAgentState",
    "DevOpsTaskType",
    # Workflow
    "create_devops_workflow",
    # Agents
    "DevOpsAgentRegistry",
    "invoke_devops_agent",
    "create_harness_expert_agent",
    "create_scaffold_agent",
    "create_codegen_agent",
    "create_kubernetes_agent",
    "create_monitoring_agent",
    "create_incident_agent",
    "create_database_agent",
    "create_testing_agent",
    # Tools
    "DevOpsToolRegistry",
    "get_all_devops_tools",
    "get_tools_for_agent",
    # Observability clients
    "PrometheusClient",
    "LokiClient",
    "ElasticsearchClient",
    "AlertManagerClient",
    "get_prometheus_client",
    "get_loki_client",
    "get_elasticsearch_client",
    "get_alertmanager_client",
    "get_log_backend",
    # Secrets Management
    "SecretsManager",
    "AzureKeyVaultProvider",
    "HarnessSecretsProvider",
    "EnvironmentProvider",
    "SecretProvider",
    "create_secrets_manager",
    "configure_agent_secrets",
    "SecretNotFoundError",
    "ProviderError",
    "ProviderNotAvailableError",
    # RAG components
    "DevOpsKnowledgeBase",
    "VoyageEmbeddings",
    "PineconeKnowledgeStore",
    "create_rag_retriever",
    "Document",
    "SearchResult",
    # Server
    "app",
    "run_server",
]
