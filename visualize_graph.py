#!/usr/bin/env python3
"""
Visualize the Deep Agent workflow graph.
Run this to see the graph structure without LangGraph Studio.
"""

from src.deep_agent.studio_graph import app

def visualize():
    """Generate ASCII visualization of the graph"""

    print("\n" + "="*80)
    print("Deep Agent Harness Automation - Workflow Graph")
    print("="*80 + "\n")

    print("Workflow Phases:")
    print("=" * 80)
    print()
    print("  START")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────┐")
    print("  │  Phase 1: Analyze Repository        │")
    print("  │  ─────────────────────────────────  │")
    print("  │  • Clone repository                 │")
    print("  │  • Run Repomix analysis             │")
    print("  │  • Detect languages/frameworks      │")
    print("  │  • Find configuration files         │")
    print("  └─────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────┐")
    print("  │  Phase 2: Design Harness Config     │")
    print("  │  ─────────────────────────────────  │")
    print("  │  • Determine pipeline structure     │")
    print("  │  • Design templates                 │")
    print("  │  • List required connectors         │")
    print("  │  • Define environments              │")
    print("  └─────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────┐")
    print("  │  Phase 3: Create Resources          │")
    print("  │  ─────────────────────────────────  │")
    print("  │  • Create secrets                   │")
    print("  │  • Create connectors                │")
    print("  │  • Create environments              │")
    print("  │  • Create service definitions       │")
    print("  │  • Create templates                 │")
    print("  │  • Create pipeline                  │")
    print("  └─────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  ┌─────────────────────────────────────┐")
    print("  │  Phase 4: Deploy & Verify           │")
    print("  │  ─────────────────────────────────  │")
    print("  │  • Trigger test deployment          │")
    print("  │  • Monitor execution                │")
    print("  │  • Verify deployment success        │")
    print("  │  • Report results                   │")
    print("  └─────────────────────────────────────┘")
    print("    │")
    print("    ▼")
    print("  END")
    print()
    print("="*80)
    print()

    # Print graph info
    print("Graph Information:")
    print(f"  Nodes: {len(app.get_graph().nodes)}")
    print(f"  Edges: {len(list(app.get_graph().edges))}")
    print()

    print("Nodes:")
    for node_id in app.get_graph().nodes:
        print(f"  • {node_id}")
    print()

    print("="*80)
    print("\nTo visualize this graph in LangGraph Studio:")
    print("1. Download LangGraph Studio: https://studio.langchain.com")
    print("2. Open the application")
    print("3. Click 'Open Folder' and select this project directory:")
    print(f"   /home/markus/ai-deep-template-engine")
    print("4. The graph will automatically load from langgraph.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    visualize()
