# DevOps Tools Quick Reference

## Import

```python
from deep_agent.devops import (
    get_all_devops_tools,
    get_devops_tools_by_category,
    create_web_search_tool,
    create_kubernetes_query_tool,
    create_kubernetes_action_tool,
    create_metrics_query_tool,
    create_log_search_tool,
    create_alert_manager_tool,
    create_terraform_tool,
    create_harness_pipeline_tool,
    create_documentation_search_tool,
)
```

## Quick Start

```python
import asyncio

# Get all tools
tools = get_all_devops_tools()

# Use a tool
k8s_tool = create_kubernetes_query_tool()
result = await k8s_tool.ainvoke({
    "resource_type": "pods",
    "namespace": "production"
})
```

## Tool Cheat Sheet

| Tool | Purpose | Key Inputs |
|------|---------|------------|
| `web_search` | Search web | query, max_results |
| `kubernetes_query` | Query K8s | resource_type, namespace, label_selector |
| `kubernetes_action` | Execute K8s action | action, resource_type, resource_name, namespace, params |
| `metrics_query` | Query Prometheus | query (PromQL), start_time, end_time |
| `log_search` | Search logs | query, service, time_range, level |
| `alert_manager` | Manage alerts | action, alert_name, silence_duration |
| `terraform` | Execute Terraform | action, workspace, vars, working_dir |
| `harness_pipeline` | Manage pipelines | action, pipeline_id, inputs, execution_id |
| `documentation_search` | Search docs | query, top_k |

## Common Patterns

### Pattern 1: Query and Act on K8s

```python
# Query
k8s_query = create_kubernetes_query_tool()
pods = await k8s_query.ainvoke({
    "resource_type": "pods",
    "namespace": "production",
    "label_selector": "app=api"
})

# Act
k8s_action = create_kubernetes_action_tool()
result = await k8s_action.ainvoke({
    "action": "scale",
    "resource_type": "deployment",
    "resource_name": "api-server",
    "namespace": "production",
    "params": {"replicas": 5}
})
```

### Pattern 2: Search and Deploy

```python
# Search for solution
search = create_web_search_tool()
solutions = await search.ainvoke({
    "query": "Kubernetes deployment best practices",
    "max_results": 3
})

# Deploy via Harness
harness = create_harness_pipeline_tool()
result = await harness.ainvoke({
    "action": "trigger",
    "pipeline_id": "deploy-prod",
    "inputs": {"version": "v1.2.3"}
})
```

### Pattern 3: Monitor and Alert

```python
# Query metrics
metrics = create_metrics_query_tool()
cpu_usage = await metrics.ainvoke({
    "query": "rate(cpu_usage[5m])",
    "start_time": "-1h",
    "end_time": "now"
})

# Check alerts
alerts = create_alert_manager_tool()
active = await alerts.ainvoke({
    "action": "list"
})

# Search logs
logs = create_log_search_tool()
errors = await logs.ainvoke({
    "query": "error",
    "service": "api-server",
    "time_range": "30m",
    "level": "error"
})
```

### Pattern 4: Infrastructure as Code

```python
# Plan
terraform = create_terraform_tool()
plan_result = await terraform.ainvoke({
    "action": "plan",
    "workspace": "staging",
    "vars": {"region": "us-west-2"},
    "working_dir": "./terraform"
})

# Apply
apply_result = await terraform.ainvoke({
    "action": "apply",
    "workspace": "staging",
    "vars": {"region": "us-west-2"},
    "working_dir": "./terraform"
})
```

## Tool Categories

```python
tools_by_category = get_devops_tools_by_category()

# Get only observability tools
monitoring_tools = tools_by_category["observability"]
# [metrics_query, log_search, alert_manager]

# Get only K8s tools
k8s_tools = tools_by_category["kubernetes"]
# [kubernetes_query, kubernetes_action]

# Get only search tools
search_tools = tools_by_category["search"]
# [web_search, documentation_search]

# Get infrastructure tools
infra_tools = tools_by_category["infrastructure"]
# [terraform]

# Get CI/CD tools
cicd_tools = tools_by_category["cicd"]
# [harness_pipeline]
```

## Integration with LangChain Agent

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = get_all_devops_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a DevOps expert assistant."),
    ("user", "{input}"),
    ("assistant", "{agent_scratchpad}"),
])

agent = create_openai_functions_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = await executor.ainvoke({
    "input": "Check pods in production and scale up if needed"
})
```

## Integration with LangGraph

```python
from langgraph.graph import StateGraph, END
from deep_agent.devops import get_devops_tools_by_category

tools = get_devops_tools_by_category()

graph = StateGraph(DevOpsState)

# Add nodes with specific tools
graph.add_node("monitor", monitor_node(tools["observability"]))
graph.add_node("scale", scale_node(tools["kubernetes"]))
graph.add_node("deploy", deploy_node(tools["cicd"]))

# Define workflow
graph.set_entry_point("monitor")
graph.add_edge("monitor", "scale")
graph.add_edge("scale", "deploy")
graph.add_edge("deploy", END)

workflow = graph.compile()
```

## Error Handling

All tools return structured errors:

```python
try:
    result = await tool.ainvoke(inputs)
    if "error" in result:
        # Handle error
        print(f"Error: {result['error']}")
    else:
        # Process result
        print(f"Success: {result}")
except Exception as e:
    # Handle exception
    print(f"Exception: {e}")
```

## Input Validation

Tools use Pydantic for validation:

```python
# This will validate automatically
result = await k8s_query.ainvoke({
    "resource_type": "pods",      # Valid
    "namespace": "production",     # Valid
    "label_selector": "app=api"   # Optional
})

# This will raise ValidationError
result = await k8s_query.ainvoke({
    "resource_type": "invalid",   # Invalid resource type
    "max_results": -1             # Invalid: must be >= 1
})
```

## Debugging

Enable logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deep_agent.devops")

# Now you'll see tool execution logs
result = await tool.ainvoke(inputs)
```

## Common Issues

### Issue: Tool returns placeholder data
**Solution**: Tools use placeholder implementations. See README.md for production integration.

### Issue: Async function not awaited
**Solution**: Always use `await` with tool invocations:
```python
# Wrong
result = tool.ainvoke(inputs)

# Right
result = await tool.ainvoke(inputs)
```

### Issue: Import error
**Solution**: Ensure you're importing from the correct path:
```python
from deep_agent.devops import create_kubernetes_query_tool
```

## Performance Tips

1. **Use categorized tools**: Only load tools you need
2. **Batch operations**: Group related tool calls
3. **Error handling**: Catch exceptions early
4. **Async execution**: Use `asyncio.gather()` for parallel calls

```python
import asyncio

# Parallel execution
results = await asyncio.gather(
    k8s_query.ainvoke({"resource_type": "pods", "namespace": "prod"}),
    metrics.ainvoke({"query": "cpu_usage", "start_time": "-1h"}),
    logs.ainvoke({"query": "error", "time_range": "30m"}),
)
```

## Examples

See `examples.py` for complete examples:

```bash
python -m deep_agent.devops.examples
```

## Documentation

- **Full API Reference**: See `README.md`
- **Implementation Details**: See `IMPLEMENTATION_SUMMARY.md`
- **Source Code**: See `tools.py`

## Support

- **Issues**: https://github.com/the-Lobbi/ai-deep-template-engine/issues
- **Documentation**: See README.md
- **Contact**: markus@thelobbi.io
