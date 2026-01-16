# Architecture

## System Overview

The Deep Agent Harness Automation System is built on a modular, subagent-based architecture that leverages LangGraph for workflow orchestration.

## Core Components

### 1. Deep Agent Orchestrator

The main orchestration layer that:
- Analyzes incoming infrastructure tasks
- Routes tasks to appropriate subagents
- Manages workflow state and execution
- Provides MCP server interface for AI integration

**Key Classes:**
- `HarnessDeepAgent`: Main agent implementation with Harness API integration
- `AgentConfig`: Configuration management for credentials and subagent selection
- `AgentState`: LangGraph state definition for workflow management

### 2. LangGraph Workflow Engine

Provides declarative workflow orchestration with:
- **State Management**: Typed state transitions between workflow nodes
- **Conditional Routing**: Dynamic routing based on task type and context
- **Error Handling**: Graceful failure recovery and retry logic
- **Observability**: Built-in tracing and logging via LangSmith (optional)

**Workflow Nodes:**
- `analyze_task`: Parse task type and determine routing strategy
- `iac_architect_node`: Execute Terraform and IaC operations
- `container_workflow_node`: Handle Docker and container tasks
- `team_accelerator_node`: Manage repository and pipeline creation
- `general_orchestration_node`: Coordinate multi-subagent workflows

### 3. Specialized Subagents

#### iac-golden-architect
Handles all infrastructure-as-code operations:
- Terraform planning and validation
- Module scaffolding and management
- Drift detection and remediation
- Security and compliance scanning

**Skills:**
- `/tf-plan`: Generate and analyze Terraform plans
- `/tf-apply`: Apply infrastructure changes
- `/validate`: Run security and compliance checks
- `/module-scaffold`: Create new Terraform modules
- `/drift-detect`: Detect infrastructure drift

#### container-workflow
Manages container lifecycle and optimization:
- Dockerfile review and optimization
- Multi-stage build optimization
- Security vulnerability scanning
- Image size reduction
- Registry management

**Skills:**
- `/dockerfile-review`: Analyze and improve Dockerfiles
- `/build`: Build and tag container images
- `/scan`: Run security vulnerability scans
- `/optimize`: Reduce image size and layers

#### team-accelerator
Accelerates team onboarding and setup:
- Repository creation in Harness Code
- CI/CD pipeline scaffolding
- Kubernetes manifest generation
- Team documentation setup
- RBAC and access configuration

**Skills:**
- `/create-repo`: Create new Harness Code repositories
- `/scaffold-pipeline`: Generate CI/CD pipelines
- `/k8s-manifests`: Create Kubernetes deployment manifests
- `/setup-rbac`: Configure team access controls

## Workflow Execution

### Task Analysis and Routing

```
User Request
     │
     ↓
┌─────────────────┐
│  Analyze Task   │ → Determine task_type
│  (Entry Point)  │   Extract context
└─────────────────┘
     │
     ↓
┌─────────────────┐
│ Route to        │
│ Subagent        │ → Based on task_type:
└─────────────────┘   • terraform → iac_architect
     │                • docker → container_workflow
     ↓                • repository → team_accelerator
┌─────────────────┐
│ Execute         │
│ Subagent Node   │ → Delegate to specialized agent
└─────────────────┘   Run agent-specific operations
     │
     ↓
┌─────────────────┐
│ Complete        │ → Return results
│ (END)           │   Update state
└─────────────────┘
```

### State Transitions

The `AgentState` TypedDict maintains workflow context:

```python
{
    "messages": [...],                    # Conversation history
    "task_type": "terraform",             # Determines routing
    "project_identifier": "lobbiai",      # Harness project
    "context": {...},                     # Task-specific params
    "subagent_results": {...},            # Execution results
    "next_action": "iac_architect"        # Next workflow step
}
```

## Integration Points

### Harness API Integration

The agent integrates with multiple Harness APIs:

1. **Harness Code API**: Repository management
   - Endpoint: `/code/api/v1/repos/{org}/{project}`
   - Operations: Create, list, update repositories

2. **Pipeline API**: CI/CD pipeline management
   - Endpoint: `/pipeline/api/pipelines/v2`
   - Operations: Create, update, execute pipelines

3. **User API**: Authentication and authorization
   - Endpoint: `/ng/api/user/currentUser`
   - Operations: Validate credentials, get user info

### MCP Server

The Model Context Protocol server provides AI integration:

```
AI Model (Claude, GPT, etc.)
         ↓
   MCP Protocol
         ↓
Deep Agent MCP Server (port 8000)
         ↓
   HarnessDeepAgent
         ↓
   Harness APIs
```

### Kubernetes Deployment

Runs as a Kubernetes deployment with:
- **Non-root user**: Security-first container design
- **Read-only filesystem**: Immutable runtime environment
- **Resource limits**: CPU/memory constraints
- **Health checks**: Liveness and readiness probes
- **Horizontal autoscaling**: HPA based on CPU/memory

## Security Architecture

### Pod Security Standards

Enforces restricted pod security context:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

### RBAC Configuration

Minimal ServiceAccount permissions:
- **ConfigMaps/Secrets**: Read-only access
- **Pods**: List and get for monitoring
- **No cluster-wide permissions**: Scoped to namespace

### Secret Management

Credentials managed via:
1. Kubernetes Secrets for runtime config
2. HashiCorp Vault for production secrets
3. Environment variables for development

## Observability

### Logging

Structured logging at multiple levels:
- **INFO**: Task routing, subagent execution
- **DEBUG**: API request/response details
- **ERROR**: Failures and exceptions
- **WARN**: Non-fatal issues

### Tracing (Optional)

LangSmith integration for:
- Workflow execution traces
- State transition visualization
- Performance metrics
- Error analysis

### Health Checks

HTTP endpoints:
- `/health`: Overall agent health
- `/readiness`: Ready to accept requests
- `/metrics`: Prometheus metrics (future)

## Extensibility

### Adding New Subagents

1. Add subagent identifier to `AgentConfig.enabled_subagents`
2. Create workflow node function in `langgraph_integration.py`
3. Add routing logic in `route_next_step()`
4. Implement subagent-specific operations
5. Update documentation

### Custom Workflow Nodes

LangGraph allows custom node addition:
```python
workflow.add_node("custom_node", custom_node_function)
workflow.add_edge("analyze", "custom_node")
workflow.add_conditional_edges("custom_node", route_func, {...})
```

## Future Enhancements

- **Multi-cloud support**: AWS, GCP, Azure provider integration
- **GitOps workflows**: ArgoCD and Flux integration
- **Policy enforcement**: OPA/Rego policy validation
- **Cost optimization**: Cloud cost analysis and recommendations
- **Observability**: Full Prometheus/Grafana metrics
