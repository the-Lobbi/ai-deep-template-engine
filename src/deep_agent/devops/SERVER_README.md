# DevOps Multi-Agent API Server

Production-ready FastAPI server for the DevOps multi-agent system with specialized agents for infrastructure, development, and operations tasks.

## Features

- **REST API**: FastAPI-based REST API with OpenAPI documentation
- **Agent Invocation**: Direct agent invocation or workflow-based routing
- **Streaming**: Server-Sent Events (SSE) for real-time streaming responses
- **Health Checks**: Comprehensive health monitoring
- **Approval Workflows**: Human approval gates for destructive operations
- **LangSmith Tracing**: Built-in observability with LangSmith
- **Structured Logging**: JSON-formatted logs with structlog
- **CORS Support**: Configurable CORS middleware
- **Type Safety**: Full Pydantic models for requests/responses

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install fastapi uvicorn structlog pydantic

# Set environment variables
export ANTHROPIC_API_KEY="your-api-key"
export LANGSMITH_API_KEY="your-langsmith-key"  # Optional
export PINECONE_API_KEY="your-pinecone-key"    # Optional
```

### Running the Server

```bash
# Development mode with auto-reload
python -m deep_agent.devops.server --reload

# Production mode
python -m deep_agent.devops.server --host 0.0.0.0 --port 8000

# Custom configuration
python -m deep_agent.devops.server --host 127.0.0.1 --port 9000 --log-level debug
```

### Programmatic Usage

```python
from deep_agent.devops.server import run_server

# Run server
run_server(host="0.0.0.0", port=8000, reload=False)
```

## API Documentation

Once the server is running, access:

- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc
- **OpenAPI spec**: http://localhost:8000/openapi.json

## API Endpoints

### Health Check

**GET** `/health`

Check server health and component status.

**Response:**
```json
{
  "status": "healthy",
  "llm": "healthy",
  "pinecone": "not_configured",
  "version": "0.1.0"
}
```

### List Agents

**GET** `/agents`

List all available agents and their descriptions.

**Response:**
```json
{
  "agents": [
    "harness_expert",
    "scaffold_agent",
    "codegen_agent",
    "kubernetes_agent",
    "monitoring_agent",
    "incident_agent",
    "database_agent",
    "testing_agent",
    "deployment_agent",
    "template_manager"
  ],
  "descriptions": {
    "harness_expert": "Harness CI/CD pipeline and template expert",
    ...
  },
  "workflow_routing": true
}
```

### Invoke Agent

**POST** `/agent/invoke`

Invoke an agent or workflow with a task.

**Request:**
```json
{
  "task": "Create a canary deployment pipeline for my-service",
  "agent_name": "harness_expert",  // Optional, uses workflow routing if null
  "context": {
    "service": "my-service",
    "environments": ["dev", "staging", "prod"]
  },
  "stream": false,
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "status": "success",
  "result": {
    "message": "Pipeline created successfully",
    "pipeline_id": "pipeline-001",
    ...
  },
  "agent_used": "harness_expert",
  "execution_time": 2.5,
  "thread_id": "abc-123-def-456"
}
```

**Status values:**
- `success`: Task completed successfully
- `pending_approval`: Requires human approval
- `error`: Task failed

### Stream Agent

**POST** `/agent/stream`

Stream agent/workflow responses using Server-Sent Events (SSE).

**Request:**
```json
{
  "task": "Deploy my application to Kubernetes",
  "agent_name": null,  // Uses workflow routing
  "context": {
    "environment": "staging"
  },
  "thread_id": "optional-thread-id"
}
```

**Response:** (Server-Sent Events)
```
data: {"event": "on_chat_model_start", "data": {...}}

data: {"event": "on_chat_model_stream", "data": {...}}

data: {"event": "on_chat_model_end", "data": {...}}

event: done
data: {"status": "complete"}
```

### Get Workflow Status

**GET** `/workflow/status/{thread_id}`

Get the current status of a workflow execution.

**Response:**
```json
{
  "thread_id": "abc-123-def-456",
  "status": "in_progress",
  "current_phase": "infrastructure_routing",
  "next_action": "kubernetes_agent",
  "supervisor_path": [
    "devops_root_supervisor",
    "infrastructure_supervisor"
  ],
  "approval_required": null
}
```

### Approve Workflow

**POST** `/workflow/approve/{thread_id}`

Approve or reject a pending workflow operation.

**Request:**
```json
{
  "approved": true,
  "comment": "Approved for staging deployment"
}
```

**Response:**
```json
{
  "status": "approved",
  "thread_id": "abc-123-def-456",
  "message": "Operations approved and executed"
}
```

## Agent Descriptions

### Infrastructure Agents

- **harness_expert**: Harness CI/CD pipeline creation, templates, and deployments
- **scaffold_agent**: Project scaffolding, initialization, and setup
- **kubernetes_agent**: Kubernetes cluster management and operations

### Development Agents

- **codegen_agent**: Code generation (API clients, models, tests, migrations)
- **database_agent**: Database schema design, migrations, and operations
- **testing_agent**: Test generation and quality assurance

### Operations Agents

- **monitoring_agent**: Monitoring, metrics, logs, and observability
- **incident_agent**: Incident response, troubleshooting, and root cause analysis
- **deployment_agent**: Deployment orchestration and management
- **template_manager**: Template management and validation

## Workflow Routing

When `agent_name` is `null`, the system uses intelligent workflow routing:

1. **Root Supervisor**: Analyzes the task and routes to appropriate supervisor
   - Infrastructure tasks → `infrastructure_supervisor`
   - Development tasks → `development_supervisor`
   - Operations tasks → `operations_supervisor`

2. **Specialized Supervisors**: Route to specific agents
   - Infrastructure: `scaffold_agent`, `harness_expert`, `kubernetes_agent`
   - Development: `codegen_agent`, `database_agent`, `testing_agent`
   - Operations: `monitoring_agent`, `incident_agent`

3. **Reflection**: Evaluates results and determines next steps
   - Quality checks
   - Remediation tasks
   - Human approval gates

## Client Examples

### Python Client

```python
import requests

# Base URL
base_url = "http://localhost:8000"

# Health check
response = requests.get(f"{base_url}/health")
print(response.json())

# List agents
response = requests.get(f"{base_url}/agents")
print(response.json())

# Invoke specific agent
response = requests.post(
    f"{base_url}/agent/invoke",
    json={
        "task": "Create a CI/CD pipeline for my-service",
        "agent_name": "harness_expert",
        "context": {
            "service": "my-service",
            "environments": ["dev", "staging", "prod"]
        }
    }
)
result = response.json()
print(f"Status: {result['status']}")
print(f"Agent: {result['agent_used']}")
print(f"Result: {result['result']}")

# Use workflow routing
response = requests.post(
    f"{base_url}/agent/invoke",
    json={
        "task": "Deploy my-service to Kubernetes staging environment",
        "context": {
            "service": "my-service",
            "environment": "staging"
        }
    }
)
result = response.json()

# Check workflow status
thread_id = result['thread_id']
response = requests.get(f"{base_url}/workflow/status/{thread_id}")
print(response.json())
```

### Streaming Client

```python
import requests
import json

# Stream agent responses
response = requests.post(
    f"{base_url}/agent/stream",
    json={
        "task": "Monitor application metrics and identify issues",
        "agent_name": "monitoring_agent"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            event_data = json.loads(line_str[6:])
            print(event_data)
```

### JavaScript/TypeScript Client

```typescript
// Invoke agent
const response = await fetch('http://localhost:8000/agent/invoke', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    task: 'Create a canary deployment pipeline',
    agent_name: 'harness_expert',
    context: {
      service: 'my-service',
    },
  }),
});

const result = await response.json();
console.log(result);

// Stream responses
const eventSource = new EventSource('http://localhost:8000/agent/stream');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};

eventSource.addEventListener('done', (event) => {
  console.log('Stream completed');
  eventSource.close();
});

eventSource.onerror = (error) => {
  console.error('Stream error:', error);
  eventSource.close();
};
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# List agents
curl http://localhost:8000/agents

# Invoke agent
curl -X POST http://localhost:8000/agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Create a CI/CD pipeline",
    "agent_name": "harness_expert",
    "context": {"service": "my-service"}
  }'

# Stream responses
curl -N -X POST http://localhost:8000/agent/stream \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Monitor application metrics",
    "agent_name": "monitoring_agent"
  }'

# Get workflow status
curl http://localhost:8000/workflow/status/abc-123-def-456

# Approve workflow
curl -X POST http://localhost:8000/workflow/approve/abc-123-def-456 \
  -H "Content-Type: application/json" \
  -d '{"approved": true, "comment": "Approved"}'
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: Anthropic API key (required)
- `LANGSMITH_API_KEY`: LangSmith API key (optional, for tracing)
- `PINECONE_API_KEY`: Pinecone API key (optional, for RAG)
- `PINECONE_ENVIRONMENT`: Pinecone environment (optional)
- `PINECONE_INDEX_NAME`: Pinecone index name (optional)

### Server Options

```bash
python -m deep_agent.devops.server --help

Options:
  --host TEXT          Host to bind to (default: 0.0.0.0)
  --port INTEGER       Port to bind to (default: 8000)
  --reload            Enable auto-reload for development
  --log-level TEXT    Logging level (debug/info/warning/error)
```

## Testing

```bash
# Run all tests
pytest test_server.py

# Run with verbose output
pytest test_server.py -v

# Run specific test
pytest test_server.py -k test_health_check

# Run with coverage
pytest test_server.py --cov=server --cov-report=html
```

## Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "deep_agent.devops.server", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  devops-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LANGSMITH_API_KEY=${LANGSMITH_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    restart: unless-stopped
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: devops-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: devops-api
  template:
    metadata:
      labels:
        app: devops-api
    spec:
      containers:
      - name: devops-api
        image: devops-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: anthropic-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: devops-api
spec:
  selector:
    app: devops-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Monitoring & Observability

### LangSmith Tracing

The server automatically integrates with LangSmith when `LANGSMITH_API_KEY` is set:

- All agent invocations are traced
- Workflow execution is tracked
- Performance metrics are collected
- Error tracking and debugging

View traces at: https://smith.langchain.com/

### Structured Logging

All logs are JSON-formatted with structlog:

```json
{
  "event": "Request completed",
  "request_id": "abc-123",
  "method": "POST",
  "path": "/agent/invoke",
  "status_code": 200,
  "duration": 2.5,
  "timestamp": "2025-01-16T10:30:45.123456Z"
}
```

### Metrics

The `/health` endpoint provides component status:

- LLM connection status
- Pinecone connection status
- Overall health status

## Security Considerations

- **API Keys**: Store in environment variables, never commit to code
- **CORS**: Configure `allow_origins` appropriately for production
- **Rate Limiting**: Consider adding rate limiting middleware
- **Authentication**: Add authentication middleware for production use
- **HTTPS**: Use HTTPS in production with proper SSL certificates

## Troubleshooting

### Common Issues

1. **Server won't start**
   - Check that all dependencies are installed
   - Verify Python version (3.11+)
   - Check port availability

2. **LLM connection fails**
   - Verify `ANTHROPIC_API_KEY` is set
   - Check API key validity
   - Verify network connectivity

3. **Agent invocation fails**
   - Check server logs for detailed error messages
   - Verify task description is clear
   - Check context parameters

4. **Streaming doesn't work**
   - Ensure client supports Server-Sent Events
   - Check firewall/proxy settings
   - Verify `Cache-Control` headers

## License

MIT License

## Support

For issues and questions:
- GitHub Issues: [Repository URL]
- Documentation: [Docs URL]
- Email: support@example.com
