"""Example client for DevOps multi-agent API server.

This module demonstrates how to interact with the DevOps API server.

Usage:
    python example_client.py
"""

import json
import time
from typing import Any, Dict

import requests


class DevOpsAPIClient:
    """Client for interacting with the DevOps multi-agent API.

    Attributes:
        base_url: Base URL of the API server
        session: Requests session for connection pooling
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def health_check(self) -> Dict[str, Any]:
        """Check server health.

        Returns:
            Health check response
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def list_agents(self) -> Dict[str, Any]:
        """List available agents.

        Returns:
            List of agents and their descriptions
        """
        response = self.session.get(f"{self.base_url}/agents")
        response.raise_for_status()
        return response.json()

    def invoke_agent(
        self,
        task: str,
        agent_name: str = None,
        context: Dict[str, Any] = None,
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """Invoke an agent with a task.

        Args:
            task: Task description
            agent_name: Specific agent name (optional, uses workflow routing if None)
            context: Additional context
            thread_id: Thread ID for conversation continuity

        Returns:
            Agent response
        """
        payload = {
            "task": task,
            "agent_name": agent_name,
            "context": context or {},
            "stream": False,
            "thread_id": thread_id,
        }

        response = self.session.post(
            f"{self.base_url}/agent/invoke",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def stream_agent(
        self,
        task: str,
        agent_name: str = None,
        context: Dict[str, Any] = None,
        thread_id: str = None,
    ):
        """Stream agent responses.

        Args:
            task: Task description
            agent_name: Specific agent name (optional)
            context: Additional context
            thread_id: Thread ID

        Yields:
            Event data dicts
        """
        payload = {
            "task": task,
            "agent_name": agent_name,
            "context": context or {},
            "thread_id": thread_id,
        }

        response = self.session.post(
            f"{self.base_url}/agent/stream",
            json=payload,
            stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    event_data = json.loads(line_str[6:])
                    yield event_data
                elif line_str.startswith("event: done"):
                    break

    def get_workflow_status(self, thread_id: str) -> Dict[str, Any]:
        """Get workflow status.

        Args:
            thread_id: Thread ID to query

        Returns:
            Workflow status
        """
        response = self.session.get(
            f"{self.base_url}/workflow/status/{thread_id}"
        )
        response.raise_for_status()
        return response.json()

    def approve_workflow(
        self,
        thread_id: str,
        approved: bool,
        comment: str = None,
    ) -> Dict[str, Any]:
        """Approve or reject a workflow.

        Args:
            thread_id: Thread ID
            approved: Whether to approve
            comment: Optional comment

        Returns:
            Approval result
        """
        payload = {
            "approved": approved,
            "comment": comment,
        }

        response = self.session.post(
            f"{self.base_url}/workflow/approve/{thread_id}",
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def example_health_check():
    """Example: Health check."""
    print("=" * 80)
    print("Example 1: Health Check")
    print("=" * 80)

    client = DevOpsAPIClient()

    health = client.health_check()
    print(f"Health Status: {health['status']}")
    print(f"LLM Status: {health['llm']}")
    print(f"Pinecone Status: {health['pinecone']}")
    print(f"Version: {health['version']}")
    print()


def example_list_agents():
    """Example: List available agents."""
    print("=" * 80)
    print("Example 2: List Available Agents")
    print("=" * 80)

    client = DevOpsAPIClient()

    agents = client.list_agents()

    print(f"Available Agents ({len(agents['agents'])}):")
    for agent in agents["agents"]:
        description = agents["descriptions"].get(agent, "No description")
        print(f"  - {agent}: {description}")
    print()


def example_invoke_specific_agent():
    """Example: Invoke a specific agent."""
    print("=" * 80)
    print("Example 3: Invoke Specific Agent")
    print("=" * 80)

    client = DevOpsAPIClient()

    task = "Create a canary deployment pipeline for my-service"
    context = {
        "service": "my-service",
        "environments": ["dev", "staging", "prod"],
    }

    print(f"Task: {task}")
    print(f"Agent: harness_expert")
    print(f"Context: {context}")
    print("\nInvoking agent...\n")

    try:
        start_time = time.time()
        result = client.invoke_agent(
            task=task,
            agent_name="harness_expert",
            context=context,
        )
        duration = time.time() - start_time

        print(f"Status: {result['status']}")
        print(f"Agent Used: {result['agent_used']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Thread ID: {result['thread_id']}")
        print(f"\nResult:\n{json.dumps(result['result'], indent=2)}")

    except requests.HTTPError as e:
        print(f"Error: {e}")
        print(f"Response: {e.response.text}")

    print()


def example_workflow_routing():
    """Example: Use workflow routing."""
    print("=" * 80)
    print("Example 4: Workflow Routing")
    print("=" * 80)

    client = DevOpsAPIClient()

    task = "Deploy my-service to Kubernetes staging environment"
    context = {
        "service": "my-service",
        "environment": "staging",
        "namespace": "staging",
    }

    print(f"Task: {task}")
    print(f"Agent: None (uses workflow routing)")
    print(f"Context: {context}")
    print("\nInvoking workflow...\n")

    try:
        result = client.invoke_agent(
            task=task,
            context=context,
        )

        print(f"Status: {result['status']}")
        print(f"Agent Used: {result['agent_used']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Thread ID: {result['thread_id']}")

        if result["status"] == "pending_approval":
            print("\nApproval Required:")
            print(json.dumps(result["result"]["approval_items"], indent=2))
        else:
            print(f"\nResult:\n{json.dumps(result['result'], indent=2)}")

    except requests.HTTPError as e:
        print(f"Error: {e}")
        print(f"Response: {e.response.text}")

    print()


def example_streaming():
    """Example: Stream agent responses."""
    print("=" * 80)
    print("Example 5: Streaming Responses")
    print("=" * 80)

    client = DevOpsAPIClient()

    task = "Monitor application metrics and identify issues"

    print(f"Task: {task}")
    print(f"Agent: monitoring_agent")
    print("\nStreaming responses...\n")

    try:
        event_count = 0
        for event in client.stream_agent(
            task=task,
            agent_name="monitoring_agent",
        ):
            event_count += 1
            event_type = event.get("event", "unknown")
            print(f"Event {event_count}: {event_type}")

            # Print selective event data
            if event_type in ["on_chat_model_stream", "on_tool_start"]:
                if "data" in event:
                    print(f"  Data: {json.dumps(event['data'])[:100]}...")

        print(f"\nTotal events received: {event_count}")

    except requests.HTTPError as e:
        print(f"Error: {e}")
        print(f"Response: {e.response.text}")

    print()


def example_workflow_status():
    """Example: Check workflow status."""
    print("=" * 80)
    print("Example 6: Workflow Status")
    print("=" * 80)

    client = DevOpsAPIClient()

    # First, invoke a workflow to get a thread ID
    task = "Create a test database schema"

    print(f"Task: {task}")
    print("\nInvoking workflow...\n")

    try:
        result = client.invoke_agent(task=task)
        thread_id = result["thread_id"]

        print(f"Thread ID: {thread_id}")
        print("\nChecking workflow status...\n")

        status = client.get_workflow_status(thread_id)

        print(f"Status: {status['status']}")
        print(f"Current Phase: {status.get('current_phase')}")
        print(f"Next Action: {status.get('next_action')}")
        print(f"Supervisor Path: {status.get('supervisor_path', [])}")

        if status.get("approval_required"):
            print(f"\nApproval Required:")
            print(json.dumps(status["approval_required"], indent=2))

    except requests.HTTPError as e:
        print(f"Error: {e}")
        if e.response.status_code == 404:
            print("Note: Thread not found in workflow state")
        else:
            print(f"Response: {e.response.text}")

    print()


def example_approval_workflow():
    """Example: Approval workflow."""
    print("=" * 80)
    print("Example 7: Approval Workflow")
    print("=" * 80)

    client = DevOpsAPIClient()

    # This is a simulated example
    # In reality, you would get a thread_id from a pending approval

    print("Note: This example simulates an approval workflow")
    print("In practice, you would:")
    print("1. Invoke a workflow that requires approval")
    print("2. Get the thread_id from the response")
    print("3. Call approve_workflow with the thread_id")
    print()

    # Example code (would fail unless there's a pending approval)
    # thread_id = "example-thread-id"
    # result = client.approve_workflow(
    #     thread_id=thread_id,
    #     approved=True,
    #     comment="Approved for staging deployment"
    # )
    # print(f"Approval Result: {result}")

    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 80)
    print("DevOps Multi-Agent API Client Examples")
    print("=" * 80)
    print()

    examples = [
        ("Health Check", example_health_check),
        ("List Agents", example_list_agents),
        ("Invoke Specific Agent", example_invoke_specific_agent),
        ("Workflow Routing", example_workflow_routing),
        ("Streaming Responses", example_streaming),
        ("Workflow Status", example_workflow_status),
        ("Approval Workflow", example_approval_workflow),
    ]

    print("Available Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    print()

    try:
        choice = input("Select example (1-7, or 'all'): ").strip().lower()

        if choice == "all":
            for name, func in examples:
                func()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            examples[idx][1]()
        else:
            print("Invalid choice. Running all examples...")
            for name, func in examples:
                func()

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
