# 🎨 LangGraph Studio is Now Running!

## Server Information

- **Status**: ✅ Running
- **PID**: Check `langgraph.pid` file
- **Port**: 2024
- **Mode**: In-Memory Development Server

## Access Studio

### Option 1: Direct Link (Recommended)

Open this URL in your browser:

```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

### Option 2: Manual Connection

1. Navigate to: https://smith.langchain.com/studio/
2. Click **"Connect to a local server"**
3. Enter: `http://127.0.0.1:2024`
4. Click **"Connect"**

## Available Endpoints

- **Studio UI**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- **API**: http://127.0.0.1:2024
- **API Docs**: http://127.0.0.1:2024/docs
- **Health Check**: http://127.0.0.1:2024/ok

## What You'll See in Studio

Once connected, you'll have access to:

### 1. Visual Graph
The 4-phase Deep Agent workflow:
- **Analyze** → Repository analysis with Repomix
- **Design** → Harness configuration design
- **Create** → Resource creation (connectors, pipelines, etc.)
- **Deploy** → Deployment verification

### 2. Interactive Features
- ▶️ **Run workflows**: Execute with test inputs
- 🔍 **Step-through debugging**: Debug phase by phase
- 📊 **State inspection**: View agent state at each node
- 💬 **Message viewing**: See LLM interactions in real-time
- ⏮️ **Time-travel debugging**: Replay and modify execution

### 3. Test Your Graph

Try running with this test input:

```json
{
  "repo_url": "https://github.com/your-org/test-repo",
  "harness_config": {
    "org_id": "your-org-id",
    "project_id": "your-project-id"
  },
  "requirements": "Set up CI/CD with Kubernetes deployment"
}
```

## Server Management

### View Logs
```bash
tail -f langgraph.log
```

### Check Status
```bash
curl http://localhost:2024/ok
```

### Stop Server
```bash
kill $(cat langgraph.pid)
# or
pkill -f "langgraph dev"
```

### Restart Server
```bash
source venv/bin/activate
langgraph dev --no-browser --port 2024
```

## Configuration

### Environment Variables
The server is using `.env` with local-only mode:
- `LANGSMITH_TRACING=false` - Data stays local, no external tracing
- No cloud credentials required for basic visualization

### Graph Configuration
Defined in `langgraph.json`:
```json
{
  "dependencies": ["."],
  "graphs": {
    "studio_graph": "./src/deep_agent/studio_graph.py:app"
  },
  "env": ".env",
  "python_version": "3.12"
}
```

## Troubleshooting

### Port Already in Use
If port 2024 is taken, change it in the startup command:
```bash
langgraph dev --no-browser --port 8080
```

Then connect to: `http://127.0.0.1:8080`

### Graph Not Loading
1. Check the log: `tail -f langgraph.log`
2. Verify graph syntax: `python -c "from src.deep_agent.studio_graph import app; print('OK')"`
3. Restart the server

### Connection Issues
If Safari blocks localhost connections:
```bash
langgraph dev --no-browser --port 2024 --tunnel
```

This creates a secure tunnel and provides a public URL.

## Features Demonstration

### Example Workflow

1. **Start a Run**
   - Click "Run" in Studio
   - Provide input state with `repo_url` and `harness_config`
   - Watch the graph execute phase by phase

2. **Inspect State**
   - Click on any node to see its input/output state
   - View the `current_phase` and `status` fields
   - Examine messages exchanged

3. **Debug Mode**
   - Click "Debug" instead of "Run"
   - Step through each node manually
   - Modify state between nodes
   - Test different execution paths

4. **Time-Travel**
   - After execution, use the timeline slider
   - Jump to any point in history
   - Fork execution from that point
   - Compare different paths

## Next Steps

1. ✅ Open Studio in your browser using the link above
2. 📊 Explore the visual graph structure
3. ▶️ Run a test workflow with sample input
4. 🔍 Use debug mode to step through execution
5. 📝 Review the full agent implementation in `src/deep_agent/harness_deep_agent.py`

## Resources

- **LangGraph Studio Docs**: https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Project Documentation**: `LANGGRAPH_STUDIO.md`
- **API Documentation**: http://127.0.0.1:2024/docs

---

**Status**: Server is running and ready for development! 🚀
**Project**: ai-deep-template-engine
**Server PID**: Check `langgraph.pid`
