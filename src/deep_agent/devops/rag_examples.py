"""Examples demonstrating RAG integration usage for DevOps operations.

This module provides practical examples of using the RAG system for:
- Document indexing and retrieval
- Semantic search across DevOps documentation
- Hybrid search with keyword matching
- Source-specific searches (Kubernetes, Harness, Terraform)
"""

import asyncio
import logging
from typing import List

from .rag import (
    Document,
    SearchResult,
    create_rag_retriever,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Basic Setup and Search
# =============================================================================


async def example_basic_search() -> None:
    """Example: Initialize RAG system and perform basic semantic search."""
    print("\n" + "=" * 80)
    print("Example 1: Basic Setup and Search")
    print("=" * 80)

    # Create RAG retriever (loads config from environment variables)
    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Perform semantic search across all documentation
        query = "How do I scale a Kubernetes deployment?"
        print(f"\nQuery: {query}")
        print("-" * 80)

        results = await kb.search_all(query, top_k=3)

        # Display results
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (score: {result.score:.3f}):")
            print(f"ID: {result.id}")
            print(f"Source: {result.metadata.get('source', 'unknown')}")
            print(f"Content: {result.content[:200]}...")

    finally:
        # Clean up
        await embeddings.close()


# =============================================================================
# Example 2: Indexing Documents
# =============================================================================


async def example_index_documents() -> None:
    """Example: Index DevOps documentation from various sources."""
    print("\n" + "=" * 80)
    print("Example 2: Indexing Documents")
    print("=" * 80)

    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Sample Kubernetes documentation
        k8s_docs = [
            Document(
                id="k8s-deployment-scale-001",
                content="""
                To scale a Kubernetes deployment, use the kubectl scale command:

                kubectl scale deployment/<deployment-name> --replicas=<count>

                For example: kubectl scale deployment/nginx-deployment --replicas=5

                You can also use kubectl autoscale to enable automatic scaling based on CPU:
                kubectl autoscale deployment/<deployment-name> --min=2 --max=10 --cpu-percent=80

                Alternatively, edit the deployment spec directly with kubectl edit deployment/<name>
                and modify the replicas field.
                """,
                metadata={
                    "source": "kubernetes",
                    "type": "tutorial",
                    "tags": ["deployment", "scaling", "kubectl"],
                    "url": "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
                },
            ),
            Document(
                id="k8s-pod-troubleshoot-001",
                content="""
                Troubleshooting Kubernetes Pods:

                1. Check pod status: kubectl get pods -n <namespace>
                2. View pod logs: kubectl logs <pod-name> -n <namespace>
                3. Describe pod for events: kubectl describe pod <pod-name> -n <namespace>
                4. Execute commands in pod: kubectl exec -it <pod-name> -- /bin/bash
                5. Check resource usage: kubectl top pod <pod-name> -n <namespace>

                Common issues:
                - ImagePullBackOff: Cannot pull container image
                - CrashLoopBackOff: Container crashes after starting
                - Pending: Cannot schedule due to resource constraints
                - OOMKilled: Out of memory
                """,
                metadata={
                    "source": "kubernetes",
                    "type": "troubleshooting",
                    "tags": ["pods", "debugging", "troubleshooting"],
                    "url": "https://kubernetes.io/docs/tasks/debug/debug-application/",
                },
            ),
        ]

        # Sample Harness documentation
        harness_docs = [
            Document(
                id="harness-pipeline-trigger-001",
                content="""
                Triggering Harness Pipelines:

                You can trigger pipelines in several ways:

                1. Manual trigger via UI or API
                2. Git webhook trigger (on push, PR, tag)
                3. Scheduled/cron trigger
                4. Artifact trigger (new image/package)
                5. Pipeline chaining (trigger from another pipeline)

                API trigger example:
                POST /pipeline/api/pipelines/execute/{pipeline-id}
                Headers: x-api-key: <your-api-key>
                Body: {"runtimeInputs": {"environment": "production"}}

                Use input sets to pass runtime variables to pipelines.
                """,
                metadata={
                    "source": "harness",
                    "type": "guide",
                    "tags": ["pipeline", "trigger", "automation"],
                    "url": "https://docs.harness.io/article/pipeline-triggers",
                },
            ),
        ]

        # Sample Terraform documentation
        terraform_docs = [
            Document(
                id="terraform-state-management-001",
                content="""
                Terraform State Management Best Practices:

                1. Use remote state storage (S3, Azure Blob, GCS)
                2. Enable state locking to prevent concurrent modifications
                3. Never commit terraform.tfstate to version control
                4. Use workspaces for environment separation
                5. Regularly backup state files

                Remote state configuration (S3):
                terraform {
                  backend "s3" {
                    bucket         = "my-terraform-state"
                    key            = "prod/terraform.tfstate"
                    region         = "us-west-2"
                    encrypt        = true
                    dynamodb_table = "terraform-state-lock"
                  }
                }

                Commands:
                - terraform state list: List resources in state
                - terraform state show <resource>: Show resource details
                - terraform state pull: Download remote state
                """,
                metadata={
                    "source": "terraform",
                    "type": "best-practices",
                    "tags": ["state", "backend", "configuration"],
                    "url": "https://www.terraform.io/docs/language/state/",
                },
            ),
        ]

        # Sample internal runbooks
        runbook_docs = [
            Document(
                id="runbook-incident-response-001",
                content="""
                Incident Response Runbook - Production API Outage

                1. Initial Assessment (0-5 minutes)
                   - Check monitoring dashboard for affected services
                   - Verify alert severity and scope
                   - Create incident ticket in Jira
                   - Notify on-call team via Slack/PagerDuty

                2. Triage (5-15 minutes)
                   - Check recent deployments (last 2 hours)
                   - Review error logs in ELK
                   - Query metrics for anomalies
                   - Identify affected customers/regions

                3. Mitigation (15-30 minutes)
                   - If recent deployment: rollback immediately
                   - If infrastructure issue: scale resources
                   - If external dependency: enable circuit breaker
                   - Update status page with customer communication

                4. Resolution (30+ minutes)
                   - Apply permanent fix
                   - Verify metrics return to normal
                   - Conduct post-mortem analysis
                   - Update runbook with lessons learned

                Contact: oncall@company.com | Slack: #incident-response
                """,
                metadata={
                    "source": "runbooks",
                    "type": "procedure",
                    "tags": ["incident", "outage", "response", "emergency"],
                    "severity": "critical",
                },
            ),
        ]

        # Combine all documents
        all_docs = k8s_docs + harness_docs + terraform_docs + runbook_docs

        # Index documents in batch
        print(f"\nIndexing {len(all_docs)} documents...")
        await kb.store.index_documents(all_docs)
        print("✓ Documents indexed successfully")

    finally:
        await embeddings.close()


# =============================================================================
# Example 3: Source-Specific Searches
# =============================================================================


async def example_source_specific_search() -> None:
    """Example: Search specific documentation sources."""
    print("\n" + "=" * 80)
    print("Example 3: Source-Specific Searches")
    print("=" * 80)

    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Search Kubernetes documentation only
        print("\n📘 Searching Kubernetes Documentation:")
        print("-" * 80)
        k8s_results = await kb.search_k8s_docs("pod troubleshooting", top_k=2)
        for result in k8s_results:
            print(f"Score: {result.score:.3f} | {result.id}")
            print(f"Preview: {result.content[:150]}...\n")

        # Search Harness documentation only
        print("\n📗 Searching Harness Documentation:")
        print("-" * 80)
        harness_results = await kb.search_harness_docs("trigger pipeline", top_k=2)
        for result in harness_results:
            print(f"Score: {result.score:.3f} | {result.id}")
            print(f"Preview: {result.content[:150]}...\n")

        # Search internal runbooks only
        print("\n📕 Searching Internal Runbooks:")
        print("-" * 80)
        runbook_results = await kb.search_runbooks("incident response", top_k=2)
        for result in runbook_results:
            print(f"Score: {result.score:.3f} | {result.id}")
            print(f"Preview: {result.content[:150]}...\n")

    finally:
        await embeddings.close()


# =============================================================================
# Example 4: Hybrid Search
# =============================================================================


async def example_hybrid_search() -> None:
    """Example: Use hybrid search combining semantic and keyword matching."""
    print("\n" + "=" * 80)
    print("Example 4: Hybrid Search")
    print("=" * 80)

    # Create retriever with balanced hybrid search (alpha=0.5)
    kb, retriever, embeddings = create_rag_retriever(alpha=0.5)

    try:
        query = "kubectl deployment scale replicas"

        # Pure semantic search (alpha=1.0)
        print("\n🔍 Pure Semantic Search (alpha=1.0):")
        print("-" * 80)
        semantic_results = await retriever.retrieve(query, top_k=3, alpha=1.0)
        for i, result in enumerate(semantic_results, 1):
            print(f"{i}. Score: {result.score:.3f} | {result.id}")

        # Balanced hybrid search (alpha=0.5)
        print("\n🔍 Balanced Hybrid Search (alpha=0.5):")
        print("-" * 80)
        hybrid_results = await retriever.retrieve(query, top_k=3, alpha=0.5)
        for i, result in enumerate(hybrid_results, 1):
            semantic = result.metadata.get("semantic_score", 0)
            keyword = result.metadata.get("keyword_score", 0)
            print(f"{i}. Score: {result.score:.3f} (sem: {semantic:.3f}, kw: {keyword:.3f}) | {result.id}")

        # Pure keyword search (alpha=0.0)
        print("\n🔍 Pure Keyword Search (alpha=0.0):")
        print("-" * 80)
        keyword_results = await retriever.retrieve(query, top_k=3, alpha=0.0)
        for i, result in enumerate(keyword_results, 1):
            print(f"{i}. Score: {result.score:.3f} | {result.id}")

    finally:
        await embeddings.close()


# =============================================================================
# Example 5: Tag-Based Search
# =============================================================================


async def example_tag_based_search() -> None:
    """Example: Search documents by tags."""
    print("\n" + "=" * 80)
    print("Example 5: Tag-Based Search")
    print("=" * 80)

    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Search for documents with specific tags
        query = "how to troubleshoot"
        tags = ["troubleshooting", "debugging"]

        print(f"\nQuery: {query}")
        print(f"Tags: {tags}")
        print("-" * 80)

        results = await kb.search_by_tags(query, tags=tags, top_k=3)

        for i, result in enumerate(results, 1):
            result_tags = result.metadata.get("tags", [])
            print(f"\nResult {i} (score: {result.score:.3f}):")
            print(f"ID: {result.id}")
            print(f"Tags: {result_tags}")
            print(f"Content: {result.content[:200]}...")

    finally:
        await embeddings.close()


# =============================================================================
# Example 6: Real-Time Incident Query
# =============================================================================


async def example_incident_query() -> None:
    """Example: Query knowledge base during an active incident."""
    print("\n" + "=" * 80)
    print("Example 6: Real-Time Incident Query")
    print("=" * 80)

    kb, retriever, embeddings = create_rag_retriever()

    try:
        # Simulate incident scenario
        incident = {
            "service": "api-gateway",
            "error": "CrashLoopBackOff",
            "namespace": "production",
        }

        print("\n🚨 INCIDENT DETECTED:")
        print(f"Service: {incident['service']}")
        print(f"Error: {incident['error']}")
        print(f"Namespace: {incident['namespace']}")
        print("-" * 80)

        # Query runbooks for incident response
        print("\n📋 Searching for relevant runbooks...")
        runbook_results = await kb.search_runbooks(
            f"{incident['error']} incident response production",
            top_k=2,
        )

        if runbook_results:
            print(f"\n✓ Found {len(runbook_results)} relevant runbooks:\n")
            for result in runbook_results:
                print(f"Runbook: {result.id}")
                print(f"Relevance: {result.score:.3f}")
                print(f"Procedure:\n{result.content[:400]}...\n")
        else:
            print("⚠ No runbooks found, searching general documentation...")

        # Query Kubernetes troubleshooting docs
        print("\n📘 Searching Kubernetes troubleshooting documentation...")
        k8s_results = await kb.search_k8s_docs(
            f"{incident['error']} kubernetes troubleshooting",
            top_k=2,
        )

        if k8s_results:
            print(f"\n✓ Found {len(k8s_results)} relevant K8s docs:\n")
            for result in k8s_results:
                print(f"Doc: {result.id}")
                print(f"Relevance: {result.score:.3f}")
                print(f"Content:\n{result.content[:400]}...\n")

    finally:
        await embeddings.close()


# =============================================================================
# Main Runner
# =============================================================================


async def run_all_examples() -> None:
    """Run all examples sequentially."""
    examples = [
        ("Basic Search", example_basic_search),
        ("Indexing Documents", example_index_documents),
        ("Source-Specific Searches", example_source_specific_search),
        ("Hybrid Search", example_hybrid_search),
        ("Tag-Based Search", example_tag_based_search),
        ("Incident Query", example_incident_query),
    ]

    print("\n" + "=" * 80)
    print("RAG Integration Examples for DevOps Multi-Agent System")
    print("=" * 80)

    for i, (name, example_func) in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Running: {name}")
        try:
            await example_func()
        except Exception as e:
            logger.error(f"Example failed: {e}", exc_info=True)

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run all examples
    asyncio.run(run_all_examples())

    # Or run individual examples:
    # asyncio.run(example_basic_search())
    # asyncio.run(example_index_documents())
    # asyncio.run(example_source_specific_search())
    # asyncio.run(example_hybrid_search())
    # asyncio.run(example_tag_based_search())
    # asyncio.run(example_incident_query())
