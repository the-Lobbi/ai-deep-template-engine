"""Complete automation example using Deep Agent.

This example demonstrates end-to-end infrastructure automation:
1. Create a new repository in Harness Code
2. Generate Kubernetes manifests
3. Create CI/CD pipeline
4. Deploy to dev environment
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from deep_agent import HarnessDeepAgent, AgentConfig, create_agent_workflow


@dataclass(frozen=True)
class OrchestrationRule:
    """Rule-driven policy for orchestrated subagent execution."""

    name: str
    node: str
    task: str
    capabilities: Sequence[str]
    timeout_seconds: float = 120.0
    max_retries: int = 2
    require_approval: bool = False
    approval_env_var: str = "ORCHESTRATION_APPROVALS"

    def is_approved(self) -> bool:
        if not self.require_approval:
            return True
        approvals = {
            entry.strip()
            for entry in os.getenv(self.approval_env_var, "").split(",")
            if entry.strip()
        }
        return self.name in approvals


@dataclass
class StepTelemetry:
    """Track timing and outcomes for orchestration steps."""

    name: str
    status: str = "pending"
    attempts: int = 0
    started_at: float = 0.0
    ended_at: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration_seconds(self) -> float:
        if not self.ended_at or not self.started_at:
            return 0.0
        return self.ended_at - self.started_at


async def execute_with_retries(
    action: Callable[[], Awaitable[Dict[str, Any]]],
    rule: OrchestrationRule,
    telemetry: StepTelemetry,
) -> Dict[str, Any]:
    """Execute an async action with timeout + exponential backoff retries."""
    delay = 1.0
    for attempt in range(1, rule.max_retries + 2):
        telemetry.attempts = attempt
        try:
            return await asyncio.wait_for(action(), timeout=rule.timeout_seconds)
        except Exception as exc:
            telemetry.error = str(exc)
            if attempt > rule.max_retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2


def choose_subagent(
    agent: HarnessDeepAgent,
    rule: OrchestrationRule,
    context: Dict[str, Any],
    fallback_subagent: str,
) -> Tuple[str, List[str]]:
    """Plan subagents for a node, then choose the best available option."""
    invocations = agent.plan_subagents_for_node(
        node_name=rule.node,
        task=rule.task,
        context=context,
        capabilities=rule.capabilities,
    )
    planned = [invocation.name for invocation in invocations]
    chosen = planned[0] if planned else fallback_subagent
    return chosen, planned


async def run_rule(
    agent: HarnessDeepAgent,
    rule: OrchestrationRule,
    context: Dict[str, Any],
    fallback_subagent: str,
) -> Tuple[Dict[str, Any], StepTelemetry]:
    """Run a subagent action with rules, retries, and telemetry."""
    telemetry = StepTelemetry(name=rule.name)
    telemetry.started_at = time.monotonic()

    if not rule.is_approved():
        telemetry.status = "blocked"
        telemetry.ended_at = time.monotonic()
        return {
            "status": "blocked",
            "reason": (
                f"Approval required via {rule.approval_env_var} to run {rule.name}."
            ),
        }, telemetry

    subagent, planned = choose_subagent(agent, rule, context, fallback_subagent)
    telemetry.metadata["planned_subagents"] = planned
    telemetry.metadata["selected_subagent"] = subagent

    async def action() -> Dict[str, Any]:
        return await agent.delegate_to_subagent(
            subagent=subagent,
            task=rule.task,
            context=context,
        )

    try:
        result = await execute_with_retries(action, rule, telemetry)
        telemetry.status = "success"
        return result, telemetry
    except Exception:
        telemetry.status = "failed"
        raise
    finally:
        telemetry.ended_at = time.monotonic()


def print_telemetry_summary(telemetry: Sequence[StepTelemetry]) -> None:
    """Print a consistent orchestration timeline."""
    print("\nOrchestration Timeline")
    print("-" * 50)
    for step in telemetry:
        details = []
        if step.attempts:
            details.append(f"attempts={step.attempts}")
        if step.error:
            details.append(f"error={step.error}")
        detail_text = f" ({', '.join(details)})" if details else ""
        print(
            f"- {step.name}: {step.status} in {step.duration_seconds():.2f}s{detail_text}"
        )


async def create_microservice(
    agent: HarnessDeepAgent,
    service_name: str,
    project_identifier: str
) -> Tuple[Dict[str, Any], List[StepTelemetry]]:
    """Create a complete microservice setup.

    Args:
        agent: Configured Harness Deep Agent
        service_name: Name of the microservice
        project_identifier: Harness project identifier

    Returns:
        Results from each step
    """
    results: Dict[str, Any] = {}
    telemetry: List[StepTelemetry] = []

    # Step 1: Create repository
    print(f"Creating repository: {service_name}")
    repo_telemetry = StepTelemetry(name="create_repository", started_at=time.monotonic())
    repo_result = await agent.create_repository(
        repo_name=service_name,
        project_identifier=project_identifier,
        description=f"Microservice: {service_name}",
        default_branch="main"
    )
    repo_telemetry.status = "success"
    repo_telemetry.ended_at = time.monotonic()
    telemetry.append(repo_telemetry)
    results["repository"] = repo_result
    print(f"✓ Repository created: {repo_result.get('path')}")

    # Step 2: Plan orchestration rules for parallel work
    k8s_rule = OrchestrationRule(
        name="k8s_manifest_design",
        node="generate_k8s",
        task="iac_design",
        capabilities=("iac", "planning"),
        timeout_seconds=120.0,
        max_retries=2,
    )
    pipeline_rule = OrchestrationRule(
        name="pipeline_blueprint",
        node="pipeline_design",
        task="pipeline_create",
        capabilities=("pipelines", "repositories"),
        timeout_seconds=120.0,
        max_retries=1,
    )

    # Step 3: Execute Kubernetes + pipeline planning concurrently
    print("Generating Kubernetes manifests and pipeline blueprint")
    k8s_context = {
        "service_name": service_name,
        "replicas": 2,
        "port": 8000,
        "resources": {
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "500m", "memory": "512Mi"},
        },
        "environment": "dev",
    }
    pipeline_context = {
        "service_name": service_name,
        "project_identifier": project_identifier,
        "pipeline_style": "ci_cd",
    }
    (k8s_result, k8s_telemetry), (pipeline_blueprint, pipeline_telemetry) = (
        await asyncio.gather(
            run_rule(
                agent,
                k8s_rule,
                k8s_context,
                fallback_subagent="iac-golden-architect",
            ),
            run_rule(
                agent,
                pipeline_rule,
                pipeline_context,
                fallback_subagent="team-accelerator",
            ),
        )
    )
    telemetry.extend([k8s_telemetry, pipeline_telemetry])
    results["kubernetes"] = k8s_result
    results["pipeline_blueprint"] = pipeline_blueprint
    print("✓ Kubernetes manifests generated")
    print("✓ Pipeline blueprint created")

    # Step 4: Create CI/CD pipeline
    print("Creating CI/CD pipeline")
    pipeline_yaml = f"""
pipeline:
  name: {service_name}-pipeline
  identifier: {service_name}_pipeline
  projectIdentifier: {project_identifier}
  orgIdentifier: default
  stages:
    - stage:
        name: Build and Test
        identifier: build_test
        type: CI
        spec:
          execution:
            steps:
              - step:
                  type: Run
                  name: Lint
                  identifier: lint
                  spec:
                    command: |
                      ruff check .
                      black --check .
              - step:
                  type: Run
                  name: Test
                  identifier: test
                  spec:
                    command: pytest --cov
              - step:
                  type: BuildAndPushDockerRegistry
                  name: Build Image
                  identifier: build_image
                  spec:
                    connectorRef: docker_hub
                    repo: thelobbi/{service_name}
                    tags:
                      - <+pipeline.sequenceId>
                      - latest
    - stage:
        name: Deploy Dev
        identifier: deploy_dev
        type: Deployment
        spec:
          serviceConfig:
            serviceRef: {service_name}
          infrastructure:
            environmentRef: dev
            infrastructureDefinition:
              type: KubernetesDirect
              spec:
                connectorRef: k8s_dev_cluster
                namespace: default
          execution:
            steps:
              - step:
                  type: K8sRollingDeploy
                  name: Rolling Deploy
                  identifier: rolling_deploy
"""

    pipeline_result = await agent.create_pipeline(
        pipeline_name=f"{service_name}-pipeline",
        project_identifier=project_identifier,
        pipeline_yaml=pipeline_yaml
    )
    pipeline_create_telemetry = StepTelemetry(
        name="create_pipeline", status="success", started_at=time.monotonic()
    )
    pipeline_create_telemetry.ended_at = time.monotonic()
    telemetry.append(pipeline_create_telemetry)
    results["pipeline"] = pipeline_result
    print(f"✓ Pipeline created: {service_name}-pipeline")

    return results, telemetry


async def terraform_workflow_example(
    agent: HarnessDeepAgent,
) -> Tuple[Dict[str, Any], List[StepTelemetry]]:
    """Example Terraform workflow using iac-golden-architect.

    Args:
        agent: Configured Harness Deep Agent

    Returns:
        Terraform execution results
    """
    print("\n=== Terraform Workflow Example ===")
    telemetry: List[StepTelemetry] = []

    # Step 1: Validate Terraform configuration
    print("Validating Terraform configuration")
    validate_rule = OrchestrationRule(
        name="terraform_validate",
        node="terraform_validate",
        task="iac_review",
        capabilities=("terraform", "iac"),
        timeout_seconds=180.0,
        max_retries=1,
    )
    validate_result, validate_telemetry = await run_rule(
        agent,
        validate_rule,
        {
            "working_dir": "/terraform/environments/dev",
            "check_format": True,
            "security_scan": True,
        },
        fallback_subagent="iac-golden-architect",
    )
    telemetry.append(validate_telemetry)
    print(f"✓ Validation: {validate_result['status']}")

    # Step 2: Generate Terraform plan
    print("Generating Terraform plan")
    plan_rule = OrchestrationRule(
        name="terraform_plan",
        node="terraform_plan",
        task="terraform_plan",
        capabilities=("terraform", "planning"),
        timeout_seconds=300.0,
        max_retries=1,
    )
    plan_result, plan_telemetry = await run_rule(
        agent,
        plan_rule,
        {
            "working_dir": "/terraform/environments/dev",
            "environment": "dev",
            "var_file": "dev.tfvars",
        },
        fallback_subagent="iac-golden-architect",
    )
    telemetry.append(plan_telemetry)
    print(f"✓ Plan generated: {plan_result['status']}")

    # Step 3: Apply (with approval check)
    apply_rule = OrchestrationRule(
        name="terraform_apply",
        node="terraform_apply",
        task="terraform_apply",
        capabilities=("terraform",),
        timeout_seconds=600.0,
        max_retries=0,
        require_approval=True,
        approval_env_var="TERRAFORM_APPLY_APPROVALS",
    )
    print("Terraform plan review required before apply")
    apply_result, apply_telemetry = await run_rule(
        agent,
        apply_rule,
        {
            "working_dir": "/terraform/environments/dev",
            "environment": "dev",
            "var_file": "dev.tfvars",
        },
        fallback_subagent="iac-golden-architect",
    )
    telemetry.append(apply_telemetry)

    return {
        "validate": validate_result,
        "plan": plan_result,
        "apply": apply_result,
    }, telemetry


async def container_optimization_example(
    agent: HarnessDeepAgent,
) -> Tuple[Dict[str, Any], List[StepTelemetry]]:
    """Example container optimization workflow.

    Args:
        agent: Configured Harness Deep Agent

    Returns:
        Container optimization results
    """
    print("\n=== Container Optimization Example ===")
    telemetry: List[StepTelemetry] = []

    # Step 1: Review Dockerfile
    print("Reviewing Dockerfile")
    review_rule = OrchestrationRule(
        name="docker_review",
        node="docker_review",
        task="containerize",
        capabilities=("docker", "containers"),
        timeout_seconds=120.0,
        max_retries=1,
    )
    review_result, review_telemetry = await run_rule(
        agent,
        review_rule,
        {
            "dockerfile_path": "./Dockerfile",
            "check_security": True,
            "optimization_level": "aggressive",
        },
        fallback_subagent="container-workflow",
    )
    telemetry.append(review_telemetry)
    print(f"✓ Review complete: {review_result['status']}")

    # Step 2: Build optimized image
    print("Building Docker image")
    build_rule = OrchestrationRule(
        name="container_build",
        node="container_build",
        task="docker_build",
        capabilities=("docker", "build"),
        timeout_seconds=600.0,
        max_retries=1,
    )
    build_result, build_telemetry = await run_rule(
        agent,
        build_rule,
        {
            "context_path": ".",
            "tag": "myapp:optimized",
            "platform": "linux/amd64",
            "use_cache": True,
        },
        fallback_subagent="container-workflow",
    )
    telemetry.append(build_telemetry)
    print(f"✓ Build complete: {build_result['status']}")

    # Step 3: Security scan
    print("Scanning for vulnerabilities")
    scan_rule = OrchestrationRule(
        name="image_scan",
        node="image_scan",
        task="image_scan",
        capabilities=("docker", "containers"),
        timeout_seconds=300.0,
        max_retries=1,
    )
    scan_result, scan_telemetry = await run_rule(
        agent,
        scan_rule,
        {
            "image_name": "myapp:optimized",
            "severity_threshold": "high",
            "fail_on_critical": True,
        },
        fallback_subagent="container-workflow",
    )
    telemetry.append(scan_telemetry)
    print(f"✓ Scan complete: {scan_result['status']}")

    return {
        "review": review_result,
        "build": build_result,
        "scan": scan_result,
    }, telemetry


async def main():
    """Run complete automation examples."""
    # Configure agent from environment variables
    config = AgentConfig(
        harness_account_id=os.getenv("HARNESS_ACCOUNT_ID"),
        harness_api_url=os.getenv("HARNESS_API_URL"),
        harness_api_token=os.getenv("HARNESS_API_TOKEN"),
        org_identifier=os.getenv("HARNESS_ORG_IDENTIFIER", "default"),
        project_identifier=os.getenv("HARNESS_PROJECT_IDENTIFIER")
    )

    async with HarnessDeepAgent(config) as agent:
        # Health check
        health = await agent.health_check()
        print(f"Agent health: {health['status']}\n")

        if health["status"] != "healthy":
            print("Agent is not healthy, exiting")
            return

        # Example 1: Create microservice
        print("=== Example 1: Create Microservice ===")
        all_telemetry: List[StepTelemetry] = []
        microservice_results, microservice_telemetry = await create_microservice(
            agent=agent,
            service_name="example-api-service",
            project_identifier="lobbiai"
        )
        all_telemetry.extend(microservice_telemetry)
        print("\nMicroservice creation complete!")
        print(f"Repository: {microservice_results['repository']['path']}")

        # Example 2: Terraform workflow
        terraform_results, terraform_telemetry = await terraform_workflow_example(agent)
        all_telemetry.extend(terraform_telemetry)
        print("\nTerraform workflow complete!")

        # Example 3: Container optimization
        container_results, container_telemetry = await container_optimization_example(agent)
        all_telemetry.extend(container_telemetry)
        print("\nContainer optimization complete!")

        # Display summary
        print("\n" + "="*50)
        print("AUTOMATION SUMMARY")
        print("="*50)
        print(f"✓ Microservice created: example-api-service")
        print(f"✓ Terraform plan generated")
        print(f"✓ Container optimized and scanned")
        print("\nAll automation tasks completed successfully!")
        print_telemetry_summary(all_telemetry)


if __name__ == "__main__":
    asyncio.run(main())
