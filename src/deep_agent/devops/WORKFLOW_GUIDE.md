# DevOps Multi-Agent Workflow Guide

## Overview

The DevOps multi-agent workflow implements a sophisticated LangGraph-based orchestration system for DevOps operations. It uses a multi-level supervisor hierarchy to route tasks to specialized agents based on task type and context.

## Architecture

### Multi-Level Supervisor Hierarchy

```
devops_root_supervisor
    ├── infrastructure_supervisor
    │   ├── scaffold_agent
    │   ├── harness_expert
    │   └── kubernetes_agent
    ├── development_supervisor
    │   ├── codegen_agent
    │   ├── database_agent
    │   └── testing_agent
    └── operations_supervisor
        ├── monitoring_agent
        └── incident_agent
```

### Workflow Components

#### 1. Supervisors (Routing Nodes)

**devops_root_supervisor**
- Entry point for all DevOps tasks
- Routes to specialized supervisors based on task type
- Routes:
  - Infrastructure tasks → `infrastructure_supervisor`
  - Development tasks → `development_supervisor`
  - Operations tasks → `operations_supervisor`

**infrastructure_supervisor**
- Handles infrastructure provisioning and deployment tasks
- Routes:
  - Scaffolding → `scaffold_agent`
  - CI/CD pipelines → `harness_expert`
  - Kubernetes operations → `kubernetes_agent`

**development_supervisor**
- Handles code generation and development tasks
- Routes:
  - Code generation → `codegen_agent`
  - Database operations → `database_agent`
  - Testing → `testing_agent`

**operations_supervisor**
- Handles monitoring and incident response
- Routes:
  - Monitoring → `monitoring_agent`
  - Incidents → `incident_agent`

#### 2. Specialized Agent Nodes

**scaffold_agent**
- Project scaffolding and template orchestration
- Creates project structure, configuration files
- Tools: harness_pipeline, documentation_search

**harness_expert**
- CI/CD pipeline creation and management
- Harness platform expertise
- Tools: harness_pipeline, documentation_search

**kubernetes_agent**
- Kubernetes cluster management
- Deployment, scaling, debugging
- Tools: kubernetes_query, kubernetes_action, documentation_search

**codegen_agent**
- API client generation
- Model generation
- Test scaffolding
- Tools: documentation_search

**database_agent**
- Schema design
- Migration creation
- Database seeding
- Tools: documentation_search

**testing_agent**
- Test scaffolding
- Coverage analysis
- Test orchestration
- Tools: documentation_search

**monitoring_agent**
- Metrics collection and analysis
- Log aggregation
- Alert management
- Tools: metrics_query, log_search, alert_manager, documentation_search

**incident_agent**
- Incident response
- Root cause analysis
- Remediation planning
- Tools: log_search, metrics_query, kubernetes_query, web_search, documentation_search

#### 3. Reflection Node

The reflection node evaluates agent results for:
- **Quality**: Success status, output completeness
- **Risks**: Warnings, errors, failures
- **Completeness**: All required outputs present

Based on evaluation:
- Routes to `human_approval` for destructive operations
- Routes back to appropriate supervisor for retry if issues found
- Routes to `end` if all checks pass

#### 4. Human Approval Gate

Pauses workflow for human review when:
- Production deployments
- Destructive operations (delete, destroy)
- High-risk changes

## State Management

### DevOpsAgentState

```python
{
    # Core workflow
    "messages": [],              # Conversation history
    "task_type": "deploy",       # Task type enum
    "context": {},               # Task-specific context
    "current_phase": "init",     # Current workflow phase

    # Routing
    "supervisor_path": [],       # Supervisors in routing chain
    "routing_trace": [],         # Detailed routing decisions
    "next_action": "",           # Next node to execute

    # Agent results
    "subagent_results": {},      # Results from each agent
    "active_agents": [],         # Currently active agents

    # Infrastructure context
    "infrastructure_state": {},  # K8s pods, services, etc.
    "metrics": {},               # Performance metrics
    "logs": [],                  # Log entries
    "alerts": [],                # Active alerts

    # Knowledge retrieval
    "retrieved_docs": [],        # Documents from knowledge base
    "search_queries": [],        # Search queries made

    # Tool results
    "web_search_results": [],    # Web search results
    "api_responses": {},         # Custom API responses

    # Memory and reflection
    "memory_bus": None,          # Shared memory bus
    "reflection_outcomes": {},   # Quality assessments
    "remediation_tasks": [],     # Follow-up tasks
}
```

## Memory Bus Integration

The workflow uses a namespace-aware memory bus for sharing data:

### Namespaces

- **workflow**: Workflow-level data (routing, reflection)
- **agent**: Agent-specific results and context
- **org**: Organization-wide shared data

### Access Control

```python
# Workflow nodes use workflow namespace
memory_bus.set(
    "workflow",
    "routing.root_supervisor",
    data,
    access_context=AccessContext.for_workflow("supervisor_name")
)

# Agent nodes use agent namespace
memory_bus.set(
    "agent",
    "agent_name.result",
    result,
    access_context=AccessContext.for_agent("agent_name")
)
```

## Usage Examples

### Example 1: Deploy to Kubernetes

```python
from deep_agent.devops import create_devops_workflow
from langchain_core.messages import HumanMessage

workflow = create_devops_workflow()

initial_state = {
    "messages": [HumanMessage(content="Deploy api-server to production")],
    "task_type": "deploy",
    "context": {
        "action": "deploy",
        "namespace": "production",
        "service_name": "api-server",
        "image": "api-server:v1.2.3",
    },
    # ... other required state fields
}

result = await workflow.ainvoke(
    initial_state,
    {"configurable": {"thread_id": "deploy-001"}}
)
```

**Routing Flow:**
1. `devops_root_supervisor` → routes to `infrastructure_supervisor` (task_type=deploy)
2. `infrastructure_supervisor` → routes to `kubernetes_agent` (task_type=deploy)
3. `kubernetes_agent` → executes deployment using kubernetes_action tool
4. `reflection` → evaluates result quality
5. END

### Example 2: Incident Response

```python
workflow = create_devops_workflow()

initial_state = {
    "messages": [HumanMessage(content="Investigate payment service latency")],
    "task_type": "incident_response",
    "context": {
        "incident_id": "INC-12345",
        "severity": "high",
        "service": "payment-service",
    },
    # ... other required state fields
}

result = await workflow.ainvoke(
    initial_state,
    {"configurable": {"thread_id": "incident-001"}}
)
```

**Routing Flow:**
1. `devops_root_supervisor` → routes to `operations_supervisor` (task_type=incident_response)
2. `operations_supervisor` → routes to `incident_agent` (task_type=incident_response)
3. `incident_agent` → analyzes logs, metrics, traces
4. `reflection` → evaluates findings
5. END

### Example 3: Create CI/CD Pipeline

```python
workflow = create_devops_workflow()

initial_state = {
    "messages": [HumanMessage(content="Create pipeline for microservice")],
    "task_type": "pipeline",
    "context": {
        "pipeline_name": "microservice-cicd",
        "repository": "https://github.com/org/microservice",
        "environments": ["dev", "staging", "prod"],
    },
    # ... other required state fields
}

result = await workflow.ainvoke(
    initial_state,
    {"configurable": {"thread_id": "pipeline-001"}}
)
```

**Routing Flow:**
1. `devops_root_supervisor` → routes to `infrastructure_supervisor` (task_type=pipeline)
2. `infrastructure_supervisor` → routes to `harness_expert` (task_type=pipeline)
3. `harness_expert` → creates pipeline using harness_pipeline tool
4. `reflection` → evaluates pipeline configuration
5. END

## Reflection and Remediation

### Quality Evaluation

The reflection node evaluates each agent result:

```python
{
    "quality": "high" | "low",
    "completeness": "complete" | "partial",
    "risks": ["execution_failed", "missing_output", "warnings_present", "errors_detected"]
}
```

### Remediation Flow

When issues are detected:

1. **Create Remediation Tasks**
   ```python
   {
       "agent": "kubernetes-agent",
       "issue": {"quality": "low", "risks": ["execution_failed"]},
       "action": "review_and_retry",
       "timestamp": "2025-01-16T..."
   }
   ```

2. **Record Failure**
   - Added to `failure_history` for tracking
   - Used for context signals

3. **Re-route**
   - Failed agent routes back to its supervisor
   - Supervisor can try different agent or retry

### Context Signals

Updated by reflection for orchestration:

```python
{
    "needs_retry": bool,
    "needs_approval": bool,
    "risk_tags": ["execution_failed", ...],
    "recent_failure_count": int,
    "last_failure_agents": ["agent1", "agent2"]
}
```

## Checkpointing and Resume

The workflow uses MemorySaver for checkpointing:

```python
# Resume from checkpoint
result = await workflow.ainvoke(
    state,
    {"configurable": {"thread_id": "deploy-001"}}
)

# State is automatically persisted at each step
# Can resume from any checkpoint
```

## Human-in-the-Loop

### Approval Gate

Activated for:
- Production deployments
- Destructive operations (delete, destroy)
- High-risk changes

```python
# Approval required message added to state
state["approval_required"] = [
    {
        "agent": "kubernetes-agent",
        "action": "delete",
        "resource": "deployment/payment-service"
    }
]

# Workflow pauses at human_approval node
# In production, would wait for actual approval
```

## Task Type Routing Reference

| Task Type | Root Routes To | Supervisor Routes To | Agent |
|-----------|----------------|---------------------|-------|
| deploy | infrastructure_supervisor | harness_expert | CI/CD deployment |
| provision | infrastructure_supervisor | kubernetes_agent | K8s provisioning |
| scaffold | infrastructure_supervisor | scaffold_agent | Project setup |
| pipeline | infrastructure_supervisor | harness_expert | Pipeline creation |
| kubernetes | infrastructure_supervisor | kubernetes_agent | K8s operations |
| codegen | development_supervisor | codegen_agent | Code generation |
| database | development_supervisor | database_agent | Database operations |
| schema | development_supervisor | database_agent | Schema design |
| migration | development_supervisor | database_agent | Migration creation |
| test | development_supervisor | testing_agent | Test creation |
| monitor | operations_supervisor | monitoring_agent | Metrics/logs |
| metrics | operations_supervisor | monitoring_agent | Metrics analysis |
| logs | operations_supervisor | monitoring_agent | Log analysis |
| alert | operations_supervisor | monitoring_agent | Alert management |
| incident_response | operations_supervisor | incident_agent | Incident handling |
| incident | operations_supervisor | incident_agent | Incident handling |
| debug | operations_supervisor | incident_agent | Debugging |

## Integration with Templating Plugin

The workflow integrates with the existing templating plugin agents:

### Available from Plugin

1. **harness-expert**: CI/CD pipeline creation
2. **scaffold-agent**: Project scaffolding
3. **codegen-agent**: Code generation
4. **database-agent**: Database operations
5. **testing-agent**: Test creation

### TODO: Integration Points

```python
# Current: Placeholder implementation
result = {
    "agent": "harness-expert",
    "status": "success",
    "output": "Pipeline created"
}

# Future: Actual agent invocation
from deep_agent.templating import invoke_harness_expert
result = await invoke_harness_expert(context, tools)
```

## Best Practices

1. **Task Type Selection**
   - Use specific task types for optimal routing
   - General types default to appropriate supervisor

2. **Context Provision**
   - Provide rich context for agents
   - Include all relevant parameters

3. **Checkpointing**
   - Use unique thread_ids for isolation
   - Can resume long-running workflows

4. **Error Handling**
   - Reflection node catches quality issues
   - Automatic retry with supervisor re-routing

5. **Human Approval**
   - Set context flags for operations requiring approval
   - Always approve destructive operations manually

## Visualization

The workflow can be visualized using LangGraph Studio:

```bash
# Start LangGraph Studio
langgraph dev

# Open: http://localhost:8000
# Load: src/deep_agent/devops/workflow.py
```

### Graph Structure

```
START → devops_root_supervisor
           ├→ infrastructure_supervisor → [agents] → reflection
           ├→ development_supervisor → [agents] → reflection
           └→ operations_supervisor → [agents] → reflection

reflection → human_approval → END
reflection → supervisor (retry) → [agent]
reflection → END
```

## Testing

Run example workflows:

```bash
python -m deep_agent.devops.example_workflow
```

This executes all example scenarios and validates routing.

## See Also

- [Implementation Summary](./IMPLEMENTATION_SUMMARY.md)
- [Quick Reference](./QUICK_REFERENCE.md)
- [Tools Documentation](./tools.py)
- [State Schema](./state.py)
