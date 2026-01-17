#!/usr/bin/env python3
"""Quick test script for the DevOps Multi-Agent System.

Usage:
    1. Set environment variable: export ANTHROPIC_API_KEY=sk-ant-...
    2. Install: pip install -e .
    3. Run: python test_devops_agent.py
"""

import asyncio
import os
import sys

# Check for API key
if not os.getenv("ANTHROPIC_API_KEY"):
    print("❌ ANTHROPIC_API_KEY not set!")
    print("   Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
    sys.exit(1)

print("✅ ANTHROPIC_API_KEY found")


async def test_agent_import():
    """Test that agents can be imported."""
    print("\n📦 Testing imports...")

    try:
        from src.deep_agent.devops import (
            DevOpsAgentState,
            DevOpsTaskType,
            create_devops_workflow,
            DevOpsAgentRegistry,
            get_all_devops_tools,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


async def test_tools():
    """Test that tools are available."""
    print("\n🔧 Testing tools...")

    from src.deep_agent.devops.tools import get_all_devops_tools, get_tools_for_agent

    all_tools = get_all_devops_tools()
    print(f"✅ Total tools available: {len(all_tools)}")

    # List tools by agent
    agents = ["harness_expert", "kubernetes_agent", "monitoring_agent"]
    for agent in agents:
        tools = get_tools_for_agent(agent)
        print(f"   {agent}: {len(tools)} tools")

    return True


async def test_workflow():
    """Test workflow creation."""
    print("\n🔄 Testing workflow...")

    from src.deep_agent.devops.workflow import create_devops_workflow

    workflow = create_devops_workflow()
    print(f"✅ Workflow created: {type(workflow).__name__}")

    # Show nodes
    if hasattr(workflow, 'nodes'):
        print(f"   Nodes: {len(workflow.nodes)}")

    return True


async def test_agent_invocation():
    """Test invoking an agent (requires ANTHROPIC_API_KEY)."""
    print("\n🤖 Testing agent invocation...")

    from src.deep_agent.devops.agents import invoke_devops_agent

    try:
        result = await invoke_devops_agent(
            agent_name="harness_expert",
            task="What are the best practices for setting up a canary deployment pipeline?",
            context={"environments": ["dev", "staging", "prod"]}
        )

        print("✅ Agent responded successfully!")
        print(f"   Result preview: {str(result.get('result', ''))[:200]}...")
        return True

    except Exception as e:
        print(f"⚠️ Agent invocation error: {e}")
        print("   This may be expected if tools need real integrations")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("DevOps Multi-Agent System - Test Suite")
    print("=" * 60)

    results = []

    # Test imports
    results.append(await test_agent_import())

    if results[-1]:
        # Test tools
        results.append(await test_tools())

        # Test workflow
        results.append(await test_workflow())

        # Test agent (optional - needs real API key)
        print("\n" + "-" * 40)
        print("Optional: Test agent invocation?")
        print("This will call Claude API and may incur costs.")
        response = input("Run agent test? [y/N]: ").strip().lower()

        if response == 'y':
            results.append(await test_agent_invocation())

    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed - check output above")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
