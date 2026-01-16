# Subagents

## Overview

The Deep Agent Harness Automation System delegates work to specialized subagents, each with domain expertise and specific capabilities.

## iac-golden-architect

**Domain**: Infrastructure as Code (Terraform)

### Capabilities

#### Terraform Planning
- Generate and analyze Terraform plans
- Validate resource configurations
- Detect potential issues before apply
- Cost estimation for cloud resources

**Usage:**
```python
result = await agent.delegate_to_subagent(
    subagent="iac-golden-architect",
    task="terraform_plan",
    context={
        "working_dir": "/path/to/terraform",
        "environment": "dev",
        "auto_approve": False
    }
)
```

#### Module Scaffolding
- Create new Terraform modules with best practices
- Generate consistent module structure
- Include examples and documentation
- Set up testing framework

**Module Structure:**
```
modules/{module-name}/
├── main.tf           # Main resource definitions
├── variables.tf      # Input variables
├── outputs.tf        # Output values
├── README.md         # Module documentation
├── examples/
│   ├── complete/     # Full example
│   └── minimal/      # Minimal example
└── tests/
    └── module_test.go
```

#### Drift Detection
- Compare actual infrastructure to Terraform state
- Identify manual changes outside IaC
- Generate remediation plans
- Alert on critical drift

#### Security & Compliance
- Scan for security misconfigurations
- Enforce organizational policies
- Validate compliance standards (SOC2, HIPAA, etc.)
- Generate security reports

### Skills

| Skill | Description | Parameters |
|-------|-------------|------------|
| `/tf-plan` | Generate Terraform plan | `working_dir`, `environment`, `var_file` |
| `/tf-apply` | Apply Terraform changes | `working_dir`, `environment`, `auto_approve` |
| `/validate` | Run security/compliance checks | `working_dir`, `policies` |
| `/module-scaffold` | Create new Terraform module | `module_name`, `provider`, `resources` |
| `/drift-detect` | Detect infrastructure drift | `environment`, `resources` |

### Configuration

```yaml
subagent: iac-golden-architect
config:
  terraform_version: "1.6.0"
  backend: "s3"
  state_lock: "dynamodb"
  allowed_providers:
    - aws
    - kubernetes
    - helm
  policy_enforcement: true
  auto_approve: false
```

## container-workflow

**Domain**: Container Lifecycle Management

### Capabilities

#### Dockerfile Review
- Analyze Dockerfile best practices
- Identify security vulnerabilities
- Optimize build performance
- Recommend multi-stage builds

**Review Checks:**
- Base image security and freshness
- Non-root user configuration
- Layer caching optimization
- Secret exposure in build args
- Image size optimization
- Vulnerability scanning results

#### Image Optimization
- Reduce image size
- Minimize layer count
- Implement multi-stage builds
- Use distroless or alpine bases
- Remove unnecessary dependencies

**Optimization Techniques:**
- Multi-stage builds
- .dockerignore configuration
- Layer ordering for cache efficiency
- Dependency consolidation
- Scratch/distroless bases where applicable

#### Security Scanning
- CVE vulnerability detection
- Malware scanning
- License compliance
- Secret detection in layers
- SBOM generation

**Scan Results:**
```json
{
  "image": "deep-agent:latest",
  "vulnerabilities": {
    "critical": 0,
    "high": 2,
    "medium": 5,
    "low": 12
  },
  "recommendations": [...]
}
```

#### Registry Management
- Push images to registries
- Manage image tags and versions
- Clean up old images
- Mirror images across registries

### Skills

| Skill | Description | Parameters |
|-------|-------------|------------|
| `/dockerfile-review` | Analyze Dockerfile | `dockerfile_path`, `severity_level` |
| `/build` | Build Docker image | `context_path`, `tag`, `platform` |
| `/scan` | Security vulnerability scan | `image_name`, `severity_threshold` |
| `/optimize` | Reduce image size | `dockerfile_path`, `target_size_mb` |
| `/push` | Push to registry | `image_name`, `registry`, `credentials` |

### Configuration

```yaml
subagent: container-workflow
config:
  docker_version: "24.0.0"
  registries:
    - url: "docker.io"
      namespace: "thelobbi"
    - url: "ghcr.io"
      namespace: "the-lobbi"
  scanning:
    enabled: true
    tools: ["trivy", "grype"]
    severity_threshold: "high"
  build:
    platform: "linux/amd64"
    cache: true
    multi_stage: true
```

## team-accelerator

**Domain**: Team Onboarding and Repository Setup

### Capabilities

#### Repository Creation
- Create Harness Code repositories
- Initialize with standard structure
- Configure branch protection
- Set up webhooks and integrations

**Repository Types:**
- Microservice (API, workers)
- Helm chart library
- Terraform module
- Shared library

#### Pipeline Scaffolding
- Generate CI/CD pipelines
- Configure stages (lint, test, build, deploy)
- Set up environments (dev, staging, prod)
- Integrate with Harness CD

**Pipeline Stages:**
```yaml
stages:
  - lint:
      steps: [ruff, black, mypy]
  - test:
      steps: [pytest, coverage]
  - build:
      steps: [docker_build, push]
  - deploy:
      environments: [dev, staging, prod]
      strategy: rolling_update
```

#### Kubernetes Manifests
- Generate deployment manifests
- Create services and ingresses
- Configure autoscaling (HPA)
- Set up RBAC and security policies

**Manifest Generation:**
- Deployment with rolling updates
- Service (ClusterIP, LoadBalancer)
- Ingress with TLS
- ConfigMap and Secret references
- ServiceAccount with minimal RBAC
- NetworkPolicy for pod isolation
- HorizontalPodAutoscaler

#### Documentation Setup
- Generate README templates
- Create Confluence documentation
- Link to Jira projects
- Set up team wikis

### Skills

| Skill | Description | Parameters |
|-------|-------------|------------|
| `/create-repo` | Create Harness repository | `repo_name`, `project`, `type` |
| `/scaffold-pipeline` | Generate CI/CD pipeline | `repo_name`, `stages`, `environments` |
| `/k8s-manifests` | Create K8s manifests | `service_name`, `replicas`, `resources` |
| `/setup-rbac` | Configure team access | `team_name`, `permissions` |
| `/generate-docs` | Create documentation | `repo_name`, `template_type` |

### Configuration

```yaml
subagent: team-accelerator
config:
  harness:
    org_identifier: "default"
    default_project: "lobbiai"
  kubernetes:
    namespace: "default"
    cluster: "dev-cluster"
  repository_templates:
    microservice:
      structure: [src/, tests/, k8s/, .harness/]
      pipeline: "standard-ci-cd.yaml"
    helm_chart:
      structure: [charts/, values/, examples/]
      pipeline: "helm-publish.yaml"
  documentation:
    confluence_space: "ENG"
    jira_project: "INFRA"
```

## Subagent Communication

### Request Format

```python
{
    "subagent": "iac-golden-architect",
    "task": "terraform_plan",
    "context": {
        "working_dir": "/terraform/environments/dev",
        "environment": "dev",
        "var_file": "dev.tfvars"
    },
    "priority": "high",
    "timeout": 300
}
```

### Response Format

```python
{
    "subagent": "iac-golden-architect",
    "task": "terraform_plan",
    "status": "success",
    "output": {
        "plan": "...",
        "changes": {
            "add": 5,
            "change": 2,
            "destroy": 0
        },
        "cost_estimate": "$45.00/month"
    },
    "execution_time": 12.5,
    "warnings": [],
    "errors": []
}
```

## Adding Custom Subagents

### Step 1: Define Subagent

Create a new subagent module:
```python
# src/deep_agent/subagents/custom_agent.py

class CustomAgent:
    """Custom subagent for specialized tasks."""

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Implement custom logic
        return {"status": "success", "output": {...}}
```

### Step 2: Register in Configuration

```python
config = AgentConfig(
    enabled_subagents=[
        "iac-golden-architect",
        "container-workflow",
        "team-accelerator",
        "custom-agent"  # Add new subagent
    ]
)
```

### Step 3: Add Workflow Node

```python
# src/deep_agent/langgraph_integration.py

def custom_agent_node(state: AgentState) -> AgentState:
    """Execute custom agent tasks."""
    # Implement node logic
    return state

workflow.add_node("custom_agent", custom_agent_node)
```

### Step 4: Update Routing

```python
def route_next_step(state: AgentState) -> str:
    if state["task_type"] == "custom_task":
        return "custom_agent"
    # ... existing routing logic
```

## Best Practices

### Subagent Selection

1. **Single Responsibility**: Each subagent focuses on one domain
2. **Clear Boundaries**: No overlapping capabilities
3. **Composability**: Subagents can be combined for complex workflows
4. **Independence**: Subagents operate autonomously

### Error Handling

```python
try:
    result = await agent.delegate_to_subagent(...)
except SubagentTimeoutError:
    # Handle timeout
except SubagentExecutionError as e:
    # Handle execution failure
    logger.error(f"Subagent failed: {e}")
```

### Monitoring

Track subagent performance:
- Execution time
- Success/failure rate
- Resource usage
- Error patterns

### Testing

Test subagents independently:
```python
@pytest.mark.asyncio
async def test_iac_architect():
    agent = IacGoldenArchitect()
    result = await agent.terraform_plan(working_dir="/test")
    assert result["status"] == "success"
```
