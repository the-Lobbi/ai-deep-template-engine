# LangGraph Studio Guide

## Overview

This project is fully configured for visualization and debugging in LangGraph Studio, a visual development environment for LangGraph workflows.

## Quick Start

### 1. Download LangGraph Studio

Download the desktop application from: **https://studio.langchain.com**

Available for:
- macOS (Apple Silicon & Intel)
- Linux (AppImage)
- Windows

### 2. Open Project in Studio

1. Launch LangGraph Studio
2. Click **"Open Folder"** or **File → Open Folder**
3. Navigate to: `/home/markus/ai-deep-template-engine`
4. Click **"Open"**

The application will automatically detect `langgraph.json` and load the workflow graph.

### 3. Explore the Workflow

Once loaded, you'll see:

#### Visual Graph
- **Nodes**: Workflow phases (Analyze, Design, Create, Deploy)
- **Edges**: Transitions between phases
- **Conditional routing**: Phase-based decision logic

#### Interactive Features
- **Run workflows**: Execute the graph with test inputs
- **Step through execution**: Debug phase by phase
- **Inspect state**: View agent state at each node
- **View messages**: See LLM interactions in real-time
- **Time travel debugging**: Replay and modify execution

## Graph Architecture

```
START
  │
  ├─> Analyze Repository
  │     ├─ Clone repo
  │     ├─ Run Repomix analysis
  │     ├─ Detect languages/frameworks
  │     └─ Find configuration files
  │
  ├─> Design Harness Configuration
  │     ├─ Determine pipeline structure
  │     ├─ Design templates
  │     ├─ List required connectors
  │     └─ Define environments
  │
  ├─> Create Harness Resources
  │     ├─ Create secrets
  │     ├─ Create connectors
  │     ├─ Create environments
  │     ├─ Create service definitions
  │     ├─ Create templates
  │     └─ Create pipeline
  │
  └─> Deploy & Verify
        ├─ Trigger test deployment
        ├─ Monitor execution
        ├─ Verify deployment success
        └─ Report results
        │
       END
```

## Configuration Files

### `langgraph.json`
Main configuration file that LangGraph Studio reads:

```json
{
  "dependencies": ["."],
  "graphs": {
    "studio_graph": "./src/deep_agent/studio_graph.py:app"
  },
  "env": ".env.example",
  "python_version": "3.12"
}
```

### `src/deep_agent/studio_graph.py`
Studio-optimized graph definition with:
- **State schema**: `AgentState` TypedDict
- **Phase nodes**: analyze, design, create, deploy
- **Routing logic**: Conditional phase transitions
- **Compiled graph**: Ready for execution

## Running in Studio

### Test Execution

1. In LangGraph Studio, click **"Run"**
2. Provide test input:
   ```json
   {
     "repo_url": "https://github.com/example/test-repo",
     "harness_config": {
       "org_id": "test-org",
       "project_id": "test-project"
     }
   }
   ```
3. Watch the graph execute phase by phase
4. Inspect state at each node

### Debug Mode

1. Click **"Debug"** instead of "Run"
2. Step through each node manually
3. Inspect:
   - Input state
   - Node execution
   - Output state
   - Messages exchanged

### Time Travel

1. After execution, use the timeline slider
2. Jump to any point in execution history
3. Modify state and re-run from that point
4. Compare different execution paths

## Alternative: Command-Line Visualization

If you don't have LangGraph Studio installed, run:

```bash
python visualize_graph.py
```

This provides an ASCII visualization of the workflow graph.

## Advanced Usage

### Custom Graphs

To add more graphs for visualization:

1. Create a new graph file in `src/deep_agent/`
2. Add to `langgraph.json`:
   ```json
   {
     "graphs": {
       "studio_graph": "./src/deep_agent/studio_graph.py:app",
       "custom_graph": "./src/deep_agent/custom_graph.py:app"
     }
   }
   ```

### Environment Variables

LangGraph Studio loads environment variables from `.env.example`:

```bash
HARNESS_API_KEY=your-api-key
HARNESS_ORG_ID=your-org-id
HARNESS_PROJECT_ID=your-project-id
ANTHROPIC_API_KEY=your-anthropic-key
```

Create a `.env` file (not committed to git) with actual values.

### Production Graphs

For production use, the full deep agent with subagents is in:
- `src/deep_agent/harness_deep_agent.py` - Main agent with subagents
- `src/deep_agent/langgraph_integration.py` - Hybrid workflow

The `studio_graph.py` is simplified for visualization purposes.

## Troubleshooting

### Graph Not Loading

1. Check `langgraph.json` syntax
2. Verify graph path is correct
3. Ensure dependencies are installed: `pip install -e .`
4. Check Python version matches (3.12)

### Execution Errors

1. Verify environment variables in `.env`
2. Check API credentials are valid
3. Review node function implementations
4. Use debug mode to isolate failing node

### Missing Dependencies

Install from project root:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

## Resources

- **LangGraph Studio Docs**: https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **LangSmith**: https://smith.langchain.com (for cloud tracing)

## Tips

1. **Use breakpoints**: Add conditional breakpoints to pause at specific states
2. **Export executions**: Save successful runs as test cases
3. **Share visualizations**: Export graph images for documentation
4. **Monitor performance**: Track node execution times
5. **Compare runs**: Side-by-side comparison of different inputs

## Next Steps

Once familiar with the visualization:

1. Explore the full agent implementation in `harness_deep_agent.py`
2. Review subagent definitions (repo-analyst, harness-expert, deployer)
3. Test with real repositories
4. Customize workflow for your use cases
5. Add monitoring and observability

---

**Project**: ai-deep-template-engine
**Graph File**: `src/deep_agent/studio_graph.py`
**Config**: `langgraph.json`
**Visualization**: `python visualize_graph.py`
