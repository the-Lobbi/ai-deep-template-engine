# DevOps Tools for Multi-Agent System

Comprehensive DevOps tools for the Deep Agent Harness system using LangChain's StructuredTool pattern.

## Overview

This package provides 9 production-ready DevOps tools designed for use in LangChain agents and LangGraph workflows:

1. **WebSearchTool** - Search for DevOps solutions and documentation
2. **KubernetesQueryTool** - Query Kubernetes cluster state
3. **KubernetesActionTool** - Execute Kubernetes actions
4. **MetricsQueryTool** - Query Prometheus/Grafana metrics
5. **LogSearchTool** - Search logs (ELK/Loki)
6. **AlertManagerTool** - Interact with AlertManager
7. **TerraformTool** - Execute Terraform commands
8. **HarnessPipelineTool** - Manage Harness CI/CD pipelines
9. **DocumentationSearchTool** - Search internal documentation

## Features

- **Async/Await Support**: All tools use async coroutines for non-blocking operations
- **Pydantic Schemas**: Strongly-typed input validation with Field descriptions
- **Error Handling**: Comprehensive try/except blocks with logging
- **Type Hints**: Full type annotations for IDE support
- **Structured Results**: Consistent result formats across all tools
- **Extensible**: Easy to add new tools following the same pattern

## Installation

```bash
pip install deep-agent-harness
```

## Quick Start

### Using Individual Tools

```python
import asyncio
from deep_agent.devops import (
    create_kubernetes_query_tool,
    create_harness_pipeline_tool,
    create_web_search_tool
)

async def main():
    # Create tools
    k8s_tool = create_kubernetes_query_tool()
    harness_tool = create_harness_pipeline_tool()
    search_tool = create_web_search_tool()

    # Query Kubernetes pods
    pods = await k8s_tool.ainvoke({
        "resource_type": "pods",
        "namespace": "production",
        "label_selector": "app=api"
    })

    # Trigger pipeline
    result = await harness_tool.ainvoke({
        "action": "trigger",
        "pipeline_id": "ci-pipeline",
        "inputs": {"branch": "main"}
    })

asyncio.run(main())
```

### Using All Tools

```python
from deep_agent.devops import get_all_devops_tools

# Get all tools for agent
tools = get_all_devops_tools()

# Use with LangChain agent
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({
    "input": "Check pods in production and trigger deployment"
})
```

### Using Categorized Tools

```python
from deep_agent.devops import get_devops_tools_by_category

# Get tools by category
tools_by_cat = get_devops_tools_by_category()

# Use only observability tools
monitoring_agent_tools = tools_by_cat["observability"]
# Returns: [metrics_query, log_search, alert_manager]

# Use only Kubernetes tools
k8s_agent_tools = tools_by_cat["kubernetes"]
# Returns: [kubernetes_query, kubernetes_action]
```

## Tool Documentation

### 1. WebSearchTool

Search the web for DevOps solutions, documentation, and best practices.

**Inputs:**
- `query` (str): Search query
- `max_results` (int): Maximum results (1-20, default: 5)

**Returns:**
```python
[
    {
        "title": "Result title",
        "url": "https://...",
        "snippet": "Description...",
        "source": "placeholder"
    }
]
```

**Example:**
```python
tool = create_web_search_tool()
results = await tool.ainvoke({
    "query": "Kubernetes pod CrashLoopBackOff troubleshooting",
    "max_results": 5
})
```

### 2. KubernetesQueryTool

Query Kubernetes cluster state for resources.

**Inputs:**
- `resource_type` (str): pods, services, deployments, nodes, configmaps, secrets, ingresses
- `namespace` (str): Namespace (default: "default")
- `label_selector` (str, optional): Label selector filter

**Returns:**
```python
[
    {
        "name": "pod-name",
        "namespace": "production",
        "status": "Running",
        "ready": "3/3",
        "age": "5d",
        "labels": "app=api"
    }
]
```

**Example:**
```python
tool = create_kubernetes_query_tool()
pods = await tool.ainvoke({
    "resource_type": "pods",
    "namespace": "production",
    "label_selector": "app=api-server,env=prod"
})
```

### 3. KubernetesActionTool

Execute actions on Kubernetes resources.

**Inputs:**
- `action` (str): scale, restart, delete, apply, patch
- `resource_type` (str): deployment, statefulset, daemonset, service, configmap
- `resource_name` (str): Resource name
- `namespace` (str): Namespace (default: "default")
- `params` (dict): Action parameters

**Returns:**
```python
{
    "action": "scale",
    "resource": "deployment/api-server",
    "namespace": "production",
    "status": "success",
    "message": "Successfully executed scale",
    "params": {"replicas": 5},
    "timestamp": "2026-01-16T..."
}
```

**Example:**
```python
tool = create_kubernetes_action_tool()
result = await tool.ainvoke({
    "action": "scale",
    "resource_type": "deployment",
    "resource_name": "api-server",
    "namespace": "production",
    "params": {"replicas": 5}
})
```

### 4. MetricsQueryTool

Query Prometheus/Grafana metrics using PromQL.

**Inputs:**
- `query` (str): PromQL query
- `start_time` (str, optional): ISO format or relative ("-1h", "-15m")
- `end_time` (str, optional): ISO format or "now"

**Returns:**
```python
{
    "query": "rate(http_requests_total[5m])",
    "start": "2026-01-16T...",
    "end": "2026-01-16T...",
    "resultType": "vector",
    "result": [
        {
            "metric": {"__name__": "http_requests_total", "job": "api"},
            "value": [1737072000, "1234.5"]
        }
    ]
}
```

**Example:**
```python
tool = create_metrics_query_tool()
metrics = await tool.ainvoke({
    "query": "rate(http_requests_total[5m])",
    "start_time": "-1h",
    "end_time": "now"
})
```

### 5. LogSearchTool

Search logs using ELK or Loki.

**Inputs:**
- `query` (str): Search query
- `service` (str, optional): Service name filter
- `time_range` (str): Time range (default: "1h")
- `level` (str, optional): error, warn, info, debug

**Returns:**
```python
[
    {
        "timestamp": "2026-01-16T...",
        "service": "api-server",
        "level": "error",
        "message": "Log message...",
        "source": "placeholder"
    }
]
```

**Example:**
```python
tool = create_log_search_tool()
logs = await tool.ainvoke({
    "query": "connection timeout",
    "service": "api-server",
    "time_range": "30m",
    "level": "error"
})
```

### 6. AlertManagerTool

Interact with AlertManager for alert management.

**Inputs:**
- `action` (str): list, silence, acknowledge, resolve
- `alert_name` (str, optional): Specific alert
- `silence_duration` (str): Duration for silence (default: "1h")

**Returns:**
```python
{
    "action": "list",
    "status": "success",
    "timestamp": "2026-01-16T...",
    "alerts": [
        {
            "name": "HighMemoryUsage",
            "severity": "warning",
            "status": "firing",
            "started_at": "2026-01-16T..."
        }
    ]
}
```

**Example:**
```python
tool = create_alert_manager_tool()

# List alerts
alerts = await tool.ainvoke({"action": "list"})

# Silence alert
result = await tool.ainvoke({
    "action": "silence",
    "alert_name": "HighMemoryUsage",
    "silence_duration": "2h"
})
```

### 7. TerraformTool

Execute Terraform commands for infrastructure management.

**Inputs:**
- `action` (str): plan, apply, destroy, state, validate, fmt
- `workspace` (str): Workspace name (default: "default")
- `vars` (dict): Terraform variables
- `working_dir` (str): Working directory (default: ".")

**Returns:**
```python
{
    "action": "plan",
    "workspace": "production",
    "status": "success",
    "working_dir": "/path/to/terraform",
    "timestamp": "2026-01-16T...",
    "plan": {
        "resources_to_add": 3,
        "resources_to_change": 1,
        "resources_to_destroy": 0
    }
}
```

**Example:**
```python
tool = create_terraform_tool()
result = await tool.ainvoke({
    "action": "plan",
    "workspace": "staging",
    "vars": {"region": "us-west-2", "instance_type": "t3.medium"},
    "working_dir": "./terraform"
})
```

### 8. HarnessPipelineTool

Interact with Harness CI/CD pipelines.

**Inputs:**
- `action` (str): trigger, status, list, logs, abort
- `pipeline_id` (str, optional): Pipeline ID
- `inputs` (dict): Pipeline runtime inputs
- `execution_id` (str, optional): Execution ID

**Returns:**
```python
{
    "action": "trigger",
    "status": "success",
    "timestamp": "2026-01-16T...",
    "execution_id": "exec-1737072000",
    "pipeline_id": "ci-pipeline",
    "status": "running"
}
```

**Example:**
```python
tool = create_harness_pipeline_tool()

# Trigger pipeline
result = await tool.ainvoke({
    "action": "trigger",
    "pipeline_id": "deploy-to-staging",
    "inputs": {
        "image_tag": "v1.2.3",
        "environment": "staging"
    }
})

# Check status
status = await tool.ainvoke({
    "action": "status",
    "execution_id": result["execution_id"]
})
```

### 9. DocumentationSearchTool

Search internal documentation using semantic similarity.

**Inputs:**
- `query` (str): Search query
- `top_k` (int): Number of results (1-20, default: 5)

**Returns:**
```python
[
    {
        "title": "Documentation title",
        "content": "Full documentation content...",
        "url": "https://docs.internal/...",
        "score": 0.95,
        "source": "placeholder"
    }
]
```

**Example:**
```python
tool = create_documentation_search_tool()
docs = await tool.ainvoke({
    "query": "How to configure Kubernetes ingress",
    "top_k": 3
})
```

## Tool Categories

Tools are organized into 5 categories:

- **search**: web_search, documentation_search
- **kubernetes**: kubernetes_query, kubernetes_action
- **observability**: metrics_query, log_search, alert_manager
- **infrastructure**: terraform
- **cicd**: harness_pipeline

## Integration Guide

### With LangGraph Workflow

```python
from langgraph.graph import StateGraph, END
from deep_agent.devops import get_devops_tools_by_category

def create_devops_workflow():
    # Get categorized tools
    tools = get_devops_tools_by_category()

    graph = StateGraph(DevOpsState)

    # Add nodes with specific tool categories
    graph.add_node("monitor", create_monitor_node(tools["observability"]))
    graph.add_node("deploy", create_deploy_node(tools["cicd"]))
    graph.add_node("scale", create_scale_node(tools["kubernetes"]))

    # Define edges
    graph.add_edge("monitor", "deploy")
    graph.add_edge("deploy", "scale")

    return graph.compile()
```

### With LangChain Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from deep_agent.devops import get_all_devops_tools

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = get_all_devops_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a DevOps expert assistant with access to various tools."),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use the agent
result = await executor.ainvoke({
    "input": "Check if there are any failing pods in production and restart them"
})
```

## Examples

See `examples.py` for complete working examples:

```bash
python -m deep_agent.devops.examples
```

Examples include:
1. Individual tool usage
2. All tools with agent
3. Categorized tools
4. Kubernetes troubleshooting workflow
5. CI/CD pipeline workflow

## Production Integration

### TODO: Replace Placeholder Implementations

The current implementation uses placeholders. For production:

1. **WebSearchTool**: Integrate Tavily or SerpAPI
2. **KubernetesTools**: Use `kubernetes` Python client
3. **MetricsQueryTool**: Use `prometheus-api-client`
4. **LogSearchTool**: Use Elasticsearch or Loki client
5. **AlertManagerTool**: Use AlertManager API client
6. **TerraformTool**: Use `subprocess` or `python-terraform`
7. **HarnessPipelineTool**: Use Harness API with `httpx`
8. **DocumentationSearchTool**: Integrate Pinecone or other vector DB

### Environment Variables

```bash
# Web Search
TAVILY_API_KEY=your-key

# Kubernetes
KUBECONFIG=/path/to/kubeconfig

# Prometheus
PROMETHEUS_URL=http://prometheus:9090

# Elasticsearch
ELASTICSEARCH_URL=http://elasticsearch:9200

# AlertManager
ALERTMANAGER_URL=http://alertmanager:9093

# Harness
HARNESS_API_URL=https://app.harness.io
HARNESS_API_KEY=your-key
HARNESS_ACCOUNT_ID=your-account

# Pinecone
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=your-env
```

## Architecture

```
deep_agent/devops/
├── __init__.py          # Package exports
├── tools.py             # Tool implementations
├── examples.py          # Usage examples
├── README.md            # This file
└── state.py             # State definitions (existing)
```

## Contributing

When adding new tools:

1. Create Pydantic input schema with Field descriptions
2. Implement async function with proper error handling
3. Use `StructuredTool.from_function` with coroutine parameter
4. Add to `get_all_devops_tools()` and appropriate category
5. Export from `__init__.py`
6. Add example usage to `examples.py`
7. Document in this README

## License

Apache-2.0

## See Also

- [LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Deep Agent Harness](../README.md)
