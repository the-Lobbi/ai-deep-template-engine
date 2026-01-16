# 🎨 LangGraph Studio - Full Deep Agent Visualization

## ✅ Server Status - UPDATED

- **Running on port**: 8123 (changed from 2024)
- **PID**: Check `langgraph.pid`
- **Graphs loaded**: 2 (full_graph + simple_graph)
- **Total nodes in full_graph**: **29 nodes**

## 🌐 Access Studio - NEW URL

### Full Graph Visualization (29 nodes):
```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123
```

Once opened, **select "full_graph"** from the graph dropdown to see the complete architecture!

## 🎯 What's New - Full Graph Architecture

The **full_graph** now shows the complete Deep Agent system with **29 nodes** including:

### 🎭 Main Orchestrator (6 nodes)
- `plan_task` - Initial task planning
- `route_to_subagent` - Intelligent routing to specialists
- `aggregate_results` - Result consolidation
- `human_approval_gate` - Production approval checkpoint
- `finalize_workflow` - Final validation
- Conditional routing logic

### 🔬 Repo-Analyst Subagent (7 nodes)
- `repo_analyst_entry` - Entry point
- `clone_repository` - Clone target repo
- `run_repomix` - Code pattern extraction
- `detect_languages` - Python, JS, TypeScript, etc.
- `detect_frameworks` - FastAPI, React, Django, etc.
- `find_configs` - Dockerfile, K8s, Terraform
- `repo_analyst_exit` - Exit with results

### ⚙️ Harness-Expert Subagent (10 nodes)
- `harness_expert_entry` - Entry point
- **Design Phase:**
  - `design_pipeline` - CI/CD structure
  - `design_templates` - Reusable templates
  - `list_requirements` - Connectors, secrets, envs
- **Creation Phase:**
  - `create_secrets` - Credentials
  - `create_connectors` - Git, Docker, K8s, Cloud
  - `create_environments` - Dev, staging, prod
  - `create_service` - Service definitions
  - `create_pipeline` - Complete pipeline
- `harness_expert_exit` - Exit with results

### 🚀 Deployer Subagent (5 nodes)
- `deployer_entry` - Entry point
- `trigger_execution` - Start pipeline
- `monitor_execution` - Track progress
- `verify_deployment` - Health checks
- `deployer_exit` - Exit with results

## 📊 Graph Complexity Comparison

| Graph | Nodes | Use Case |
|-------|-------|----------|
| **full_graph** | 29 | Complete architecture visualization |
| **simple_graph** | 6 | Quick overview |

## 🎮 How to Use in Studio

### 1. Open Studio
Click this link:
```
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123
```

### 2. Select Full Graph
In the Studio UI:
- Look for the **graph dropdown** (top left)
- Select **"full_graph"**
- You'll now see all 29 nodes!

### 3. Explore the Architecture
- **Zoom out** to see the full workflow
- **Click nodes** to see their implementation
- **Trace paths** through conditional edges
- **Identify subagent boundaries**

### 4. Run a Test
Click **"Run"** and provide:
```json
{
  "task": "Automate CI/CD for FastAPI microservice",
  "repo_url": "https://github.com/fastapi/fastapi",
  "harness_config": {
    "org_id": "my-org",
    "project_id": "my-project"
  }
}
```

Watch as the execution flows through:
1. **Planning** → Routing
2. **Repo-Analyst** → 7 sequential steps
3. **Harness-Expert (Design)** → 4 steps
4. **Approval Gate** → Human checkpoint
5. **Harness-Expert (Create)** → 6 resource creation steps
6. **Deployer** → 4 verification steps
7. **Finalization** → Complete

## 🔍 Key Visualization Features

### Subagent Boundaries
Each subagent has:
- **Entry node**: Clear entry point
- **Sequential workflow**: Logical step progression
- **Exit node**: Results handoff
- **Prefixed messages**: `[REPO-ANALYST]`, `[HARNESS-EXPERT]`, `[DEPLOYER]`

### Conditional Routing
The graph includes smart routing:
- **After planning**: Routes to repo-analyst
- **After analysis**: Routes to harness-expert (design)
- **After design**: Routes to approval gate
- **After approval**: Routes to harness-expert (create)
- **After creation**: Routes to deployer
- **After verification**: Finalizes

### Human-in-the-Loop
- **Approval gate** before production resource creation
- **Pause execution** for review
- **Approve or reject** via Studio UI
- **Resume after approval**

## 📈 Node Details

### Example: repo_analyst_exit Node
```python
def repo_analyst_exit(state: DeepAgentState) -> dict:
    return {
        "current_phase": "analysis_complete",
        "repo_analysis": {
            "languages": state.get("detected_languages", []),
            "frameworks": state.get("detected_frameworks", []),
            "has_docker": state.get("has_dockerfile", False),
            "has_k8s": state.get("has_kubernetes", False)
        },
        "messages": [AIMessage(content="[REPO-ANALYST] Analysis complete")]
    }
```

Click on any node in Studio to see its implementation!

## 🛠️ Server Management

### View Logs
```bash
tail -f langgraph.log
```

### Check Status
```bash
curl http://localhost:8123/ok
```

### Stop Server
```bash
kill $(cat langgraph.pid)
```

### Restart Server
```bash
source venv/bin/activate
langgraph dev --no-browser --port 8123
```

## 🎯 Next Steps

1. ✅ Open Studio at the new URL (port 8123)
2. 📊 Select **"full_graph"** from the dropdown
3. 🔍 Explore all 29 nodes
4. ▶️ Run a test execution
5. 🎮 Use debug mode to step through each node
6. 📝 Compare with `simple_graph` (6 nodes) for contrast

## 📚 Graph Files

- **Full Graph**: `src/deep_agent/studio_graph_full.py` (29 nodes)
- **Simple Graph**: `src/deep_agent/studio_graph.py` (6 nodes)
- **Configuration**: `langgraph.json`
- **Production Implementation**: `src/deep_agent/harness_deep_agent.py`

## 🎨 Visual Overview

```
                    START
                      ↓
                 plan_task
                      ↓
              route_to_subagent
                      ↓
         ┌────────────┼────────────┐
         ↓            ↓            ↓
    REPO-ANALYST  HARNESS-EXPERT  DEPLOYER
    (7 nodes)     (10 nodes)      (5 nodes)
         ↓            ↓            ↓
         └────────────┼────────────┘
                      ↓
              aggregate_results
                      ↓
           human_approval_gate
                      ↓
              finalize_workflow
                      ↓
                     END
```

---

**Much better visualization now! 29 nodes showing the complete architecture! 🚀**

**New Studio URL**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123
