"""Complete automation example using Deep Agent.

This example demonstrates end-to-end infrastructure automation:
1. Create a new repository in Harness Code
2. Generate Kubernetes manifests
3. Create CI/CD pipeline
4. Deploy to dev environment
"""

import asyncio
import os
from typing import Dict, Any

from deep_agent import HarnessDeepAgent, AgentConfig, create_agent_workflow


async def create_microservice(
    agent: HarnessDeepAgent,
    service_name: str,
    project_identifier: str
) -> Dict[str, Any]:
    """Create a complete microservice setup.

    Args:
        agent: Configured Harness Deep Agent
        service_name: Name of the microservice
        project_identifier: Harness project identifier

    Returns:
        Results from each step
    """
    results = {}

    # Step 1: Create repository
    print(f"Creating repository: {service_name}")
    repo_result = await agent.create_repository(
        repo_name=service_name,
        project_identifier=project_identifier,
        description=f"Microservice: {service_name}",
        default_branch="main"
    )
    results["repository"] = repo_result
    print(f"✓ Repository created: {repo_result.get('path')}")

    # Step 2: Delegate Kubernetes manifest generation to team-accelerator
    print("Generating Kubernetes manifests")
    k8s_result = await agent.delegate_to_subagent(
        subagent="team-accelerator",
        task="generate_k8s_manifests",
        context={
            "service_name": service_name,
            "replicas": 2,
            "port": 8000,
            "resources": {
                "requests": {"cpu": "100m", "memory": "128Mi"},
                "limits": {"cpu": "500m", "memory": "512Mi"}
            }
        }
    )
    results["kubernetes"] = k8s_result
    print("✓ Kubernetes manifests generated")

    # Step 3: Create CI/CD pipeline
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
    results["pipeline"] = pipeline_result
    print(f"✓ Pipeline created: {service_name}-pipeline")

    return results


async def terraform_workflow_example(agent: HarnessDeepAgent) -> Dict[str, Any]:
    """Example Terraform workflow using iac-golden-architect.

    Args:
        agent: Configured Harness Deep Agent

    Returns:
        Terraform execution results
    """
    print("\n=== Terraform Workflow Example ===")

    # Step 1: Validate Terraform configuration
    print("Validating Terraform configuration")
    validate_result = await agent.delegate_to_subagent(
        subagent="iac-golden-architect",
        task="validate",
        context={
            "working_dir": "/terraform/environments/dev",
            "check_format": True,
            "security_scan": True
        }
    )
    print(f"✓ Validation: {validate_result['status']}")

    # Step 2: Generate Terraform plan
    print("Generating Terraform plan")
    plan_result = await agent.delegate_to_subagent(
        subagent="iac-golden-architect",
        task="terraform_plan",
        context={
            "working_dir": "/terraform/environments/dev",
            "environment": "dev",
            "var_file": "dev.tfvars"
        }
    )
    print(f"✓ Plan generated: {plan_result['status']}")

    # Step 3: Apply (with approval check)
    print("Terraform plan review required before apply")
    # In production, this would trigger an approval workflow

    return {
        "validate": validate_result,
        "plan": plan_result
    }


async def container_optimization_example(agent: HarnessDeepAgent) -> Dict[str, Any]:
    """Example container optimization workflow.

    Args:
        agent: Configured Harness Deep Agent

    Returns:
        Container optimization results
    """
    print("\n=== Container Optimization Example ===")

    # Step 1: Review Dockerfile
    print("Reviewing Dockerfile")
    review_result = await agent.delegate_to_subagent(
        subagent="container-workflow",
        task="dockerfile_review",
        context={
            "dockerfile_path": "./Dockerfile",
            "check_security": True,
            "optimization_level": "aggressive"
        }
    )
    print(f"✓ Review complete: {review_result['status']}")

    # Step 2: Build optimized image
    print("Building Docker image")
    build_result = await agent.delegate_to_subagent(
        subagent="container-workflow",
        task="build",
        context={
            "context_path": ".",
            "tag": "myapp:optimized",
            "platform": "linux/amd64",
            "use_cache": True
        }
    )
    print(f"✓ Build complete: {build_result['status']}")

    # Step 3: Security scan
    print("Scanning for vulnerabilities")
    scan_result = await agent.delegate_to_subagent(
        subagent="container-workflow",
        task="security_scan",
        context={
            "image_name": "myapp:optimized",
            "severity_threshold": "high",
            "fail_on_critical": True
        }
    )
    print(f"✓ Scan complete: {scan_result['status']}")

    return {
        "review": review_result,
        "build": build_result,
        "scan": scan_result
    }


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
        microservice_results = await create_microservice(
            agent=agent,
            service_name="example-api-service",
            project_identifier="lobbiai"
        )
        print("\nMicroservice creation complete!")
        print(f"Repository: {microservice_results['repository']['path']}")

        # Example 2: Terraform workflow
        terraform_results = await terraform_workflow_example(agent)
        print("\nTerraform workflow complete!")

        # Example 3: Container optimization
        container_results = await container_optimization_example(agent)
        print("\nContainer optimization complete!")

        # Display summary
        print("\n" + "="*50)
        print("AUTOMATION SUMMARY")
        print("="*50)
        print(f"✓ Microservice created: example-api-service")
        print(f"✓ Terraform plan generated")
        print(f"✓ Container optimized and scanned")
        print("\nAll automation tasks completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
