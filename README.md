# Deep Agent Harness Automation System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Harness](https://img.shields.io/badge/Harness-Code%20%7C%20CI%2FCD-orange.svg)](https://harness.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Powered-green.svg)](https://github.com/langchain-ai/langgraph)

A LangGraph-powered MCP server for infrastructure orchestration with autonomous subagents.

## Overview

Deep Agent Harness Automation System orchestrates complex infrastructure tasks by delegating to specialized subagents:

- **iac-golden-architect**: Terraform planning, validation, and module management
- **container-workflow**: Docker image optimization, security scanning, and registry management
- **team-accelerator**: Repository creation, pipeline setup, and team onboarding

## Quick Start

```bash
# Clone the repository
git clone https://github.com/the-Lobbi/ai-deep-template-engine.git
cd ai-deep-template-engine

# Configure environment
cp .env.example .env
# Edit .env with your Harness credentials

# Run with Docker Compose
docker-compose up -d
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and workflow diagrams |
| [Subagents](docs/subagents.md) | Specialized subagent capabilities |
| [API Reference](docs/api.md) | MCP server API documentation |

## Environment Configuration

Required environment variables:

```bash
HARNESS_ACCOUNT_ID=your_account_id
HARNESS_API_URL=https://app.harness.io/gateway
HARNESS_API_TOKEN=your_api_token
HARNESS_ORG_IDENTIFIER=default
HARNESS_PROJECT_IDENTIFIER=your_project
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8000
```

Run the MCP server (consumes `MCP_SERVER_HOST`/`MCP_SERVER_PORT` via `AgentConfig`):

```bash
python -m deep_agent.mcp_runner
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ tests/
ruff check --fix src/ tests/

# Type checking
mypy src/
```

## Architecture

The Deep Agent uses LangGraph to orchestrate workflows across specialized subagents:

```
┌─────────────────────────────────────────────┐
│         Deep Agent Orchestrator             │
├─────────────────────────────────────────────┤
│  ┌───────────────┐  ┌──────────────────┐   │
│  │   Analyze     │→ │  Route to        │   │
│  │   Task        │  │  Subagent        │   │
│  └───────────────┘  └──────────────────┘   │
│           │                  │              │
│           ↓                  ↓              │
│  ┌─────────────────────────────────────┐   │
│  │      Specialized Subagents          │   │
│  │  • iac-golden-architect             │   │
│  │  • container-workflow               │   │
│  │  • team-accelerator                 │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Features

- **Autonomous Orchestration**: LangGraph-powered workflow management
- **Harness Integration**: Native Harness Code and Pipeline API support
- **Multi-Subagent**: Specialized agents for IaC, containers, and team acceleration
- **MCP Server**: Model Context Protocol server for AI integration
- **Production Ready**: Docker, Kubernetes, and Helm support

## CI/CD Pipeline

Harness pipeline includes:
- Lint & Test (ruff, black, pytest)
- Build Docker image
- Deploy to Kubernetes (dev, staging, prod)

## Team & Support

**Owner**: DevOps Engineering Team
**Slack**: #deep-agent-support
**On-Call**: PagerDuty rotation

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.
