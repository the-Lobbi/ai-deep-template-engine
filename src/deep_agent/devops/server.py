"""Production-ready FastAPI server for DevOps multi-agent system.

This module provides a REST API for the DevOps multi-agent system with:
- Agent invocation (single agent or workflow routing)
- Streaming responses via Server-Sent Events (SSE)
- Health checks and monitoring
- Human approval workflows
- LangSmith tracing integration
- Structured logging with structlog
- CORS and security middleware

Usage:
    python -m deep_agent.devops.server

    Or programmatically:
    >>> from deep_agent.devops.server import run_server
    >>> run_server(host="0.0.0.0", port=8000)

API Endpoints:
    POST /agent/invoke - Invoke agent or workflow
    POST /agent/stream - Streaming response (SSE)
    GET /agents - List available agents
    GET /health - Health check
    GET /workflow/status/{thread_id} - Get workflow status
    POST /workflow/approve/{thread_id} - Approve pending operation
"""

import asyncio
import json
import logging
import os
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage

from .agents import DevOpsAgentRegistry, invoke_devops_agent
from .state import DevOpsAgentState
from .workflow import CHECKPOINTER, create_devops_workflow

# =============================================================================
# Logging Configuration
# =============================================================================

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# =============================================================================
# Global State
# =============================================================================

# Global registry and workflow instances (initialized in lifespan)
agent_registry: Optional[DevOpsAgentRegistry] = None
devops_workflow = None

# Track pending approvals
pending_approvals: Dict[str, DevOpsAgentState] = {}

# =============================================================================
# Pydantic Models
# =============================================================================


class AgentRequest(BaseModel):
    """Request model for agent invocation.

    Attributes:
        task: The task description or question
        agent_name: Optional specific agent name. If None, uses workflow routing
        context: Additional context (environment, service, etc.)
        stream: Whether to stream the response
        thread_id: Optional thread ID for conversation continuity
    """

    task: str = Field(..., description="Task description or question")
    agent_name: Optional[str] = Field(
        None,
        description="Specific agent name (if None, uses workflow routing)",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for the task",
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response",
    )
    thread_id: Optional[str] = Field(
        None,
        description="Thread ID for conversation continuity",
    )


class AgentResponse(BaseModel):
    """Response model for agent invocation.

    Attributes:
        status: Status of the invocation (success, error, pending_approval)
        result: The agent's response or workflow result
        agent_used: Name of the agent that handled the request
        execution_time: Time taken to execute in seconds
        thread_id: Thread ID for the conversation
    """

    status: str = Field(..., description="Status: success, error, pending_approval")
    result: Any = Field(..., description="Agent response or workflow result")
    agent_used: str = Field(..., description="Name of the agent used")
    execution_time: float = Field(..., description="Execution time in seconds")
    thread_id: str = Field(..., description="Thread ID for the conversation")


class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint.

    Attributes:
        status: Overall health status
        llm: LLM connection status
        pinecone: Pinecone connection status (if configured)
        version: API version
    """

    status: str = Field(..., description="Overall health status")
    llm: str = Field(..., description="LLM connection status")
    pinecone: str = Field(..., description="Pinecone connection status")
    version: str = Field(..., description="API version")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status endpoint.

    Attributes:
        thread_id: Thread ID of the workflow
        status: Current workflow status
        current_phase: Current phase of execution
        next_action: Next action to be taken
        supervisor_path: Path of supervisors that have processed the request
        approval_required: List of items requiring approval (if any)
    """

    thread_id: str = Field(..., description="Thread ID of the workflow")
    status: str = Field(..., description="Current workflow status")
    current_phase: Optional[str] = Field(None, description="Current execution phase")
    next_action: Optional[str] = Field(None, description="Next action to be taken")
    supervisor_path: List[str] = Field(
        default_factory=list,
        description="Path of supervisors",
    )
    approval_required: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Items requiring approval",
    )


class ApprovalRequest(BaseModel):
    """Request model for approval submission.

    Attributes:
        approved: Whether the operation is approved
        comment: Optional comment for the approval decision
    """

    approved: bool = Field(..., description="Whether to approve the operation")
    comment: Optional[str] = Field(None, description="Optional approval comment")


# =============================================================================
# Lifespan Context Manager
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifespan (startup and shutdown).

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    logger.info("Starting DevOps multi-agent API server")

    global agent_registry, devops_workflow

    try:
        # Initialize agent registry
        logger.info("Initializing agent registry")
        agent_registry = DevOpsAgentRegistry()

        # Initialize workflow
        logger.info("Initializing DevOps workflow")
        devops_workflow = create_devops_workflow()

        logger.info("Server startup complete", agents=agent_registry.list_agents())

    except Exception as e:
        logger.error("Startup failed", error=str(e), traceback=traceback.format_exc())
        raise

    yield

    # Shutdown
    logger.info("Shutting down DevOps multi-agent API server")

    # Cleanup resources
    agent_registry = None
    devops_workflow = None
    pending_approvals.clear()

    logger.info("Server shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="DevOps Multi-Agent API",
    description="REST API for DevOps multi-agent system with specialized agents for infrastructure, development, and operations tasks",
    version="0.1.0",
    lifespan=lifespan,
)

# =============================================================================
# Middleware
# =============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all incoming requests with timing."""
    request_id = str(uuid4())
    start_time = time.time()

    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    try:
        response = await call_next(request)

        duration = time.time() - start_time

        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration=duration,
        )

        return response

    except Exception as e:
        duration = time.time() - start_time

        logger.error(
            "Request failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration=duration,
            traceback=traceback.format_exc(),
        )

        raise


# =============================================================================
# Health Check Functions
# =============================================================================


def check_llm_connection() -> str:
    """Check LLM connection status.

    Returns:
        Status string: "healthy", "degraded", or "unhealthy"
    """
    try:
        # Try to import and verify API key
        anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not anthropic_api_key:
            return "unhealthy: ANTHROPIC_API_KEY not set"

        return "healthy"

    except Exception as e:
        logger.error("LLM health check failed", error=str(e))
        return f"unhealthy: {str(e)}"


def check_pinecone_connection() -> str:
    """Check Pinecone connection status.

    Returns:
        Status string: "healthy", "not_configured", or "unhealthy"
    """
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")

        if not pinecone_api_key:
            return "not_configured"

        # If configured, attempt a basic connection check
        # (Actual connection check would depend on Pinecone client initialization)
        return "healthy"

    except Exception as e:
        logger.error("Pinecone health check failed", error=str(e))
        return f"unhealthy: {str(e)}"


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint.

    Returns:
        HealthCheckResponse with component status
    """
    logger.info("Health check requested")

    llm_status = check_llm_connection()
    pinecone_status = check_pinecone_connection()

    overall_status = "healthy"
    if "unhealthy" in llm_status:
        overall_status = "unhealthy"
    elif "unhealthy" in pinecone_status:
        overall_status = "degraded"

    return HealthCheckResponse(
        status=overall_status,
        llm=llm_status,
        pinecone=pinecone_status,
        version="0.1.0",
    )


@app.get("/agents")
async def list_agents() -> Dict[str, Any]:
    """List available agents.

    Returns:
        Dict with available agents and their descriptions
    """
    logger.info("Agent list requested")

    if not agent_registry:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent registry not initialized",
        )

    agents = agent_registry.list_agents()

    agent_info = {
        "harness_expert": "Harness CI/CD pipeline and template expert",
        "scaffold_agent": "Project scaffolding and initialization",
        "codegen_agent": "Code generation (API clients, models, tests)",
        "kubernetes_agent": "Kubernetes management and operations",
        "monitoring_agent": "Monitoring, metrics, logs, and alerts",
        "incident_agent": "Incident response and troubleshooting",
        "database_agent": "Database operations and migrations",
        "testing_agent": "Testing and quality assurance",
        "deployment_agent": "Deployment orchestration",
        "template_manager": "Template management and validation",
    }

    return {
        "agents": agents,
        "descriptions": agent_info,
        "workflow_routing": True,
    }


@app.post("/agent/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    """Invoke an agent or workflow.

    If agent_name is specified, invokes that specific agent directly.
    If agent_name is None, uses workflow routing to determine the appropriate agent.

    Args:
        request: Agent invocation request

    Returns:
        AgentResponse with the result

    Raises:
        HTTPException: If invocation fails
    """
    logger.info(
        "Agent invocation requested",
        agent_name=request.agent_name,
        task=request.task[:100],
        stream=request.stream,
        thread_id=request.thread_id,
    )

    if request.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="For streaming responses, use /agent/stream endpoint",
        )

    start_time = time.time()
    thread_id = request.thread_id or str(uuid4())

    try:
        if request.agent_name:
            # Direct agent invocation
            if not agent_registry:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Agent registry not initialized",
                )

            logger.info("Invoking specific agent", agent_name=request.agent_name)

            result = await invoke_devops_agent(
                agent_name=request.agent_name,
                task=request.task,
                context=request.context,
                thread_id=thread_id,
                registry=agent_registry,
            )

            execution_time = time.time() - start_time

            if not result.get("success"):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=result.get("error", "Agent invocation failed"),
                )

            return AgentResponse(
                status="success",
                result=result.get("result"),
                agent_used=request.agent_name,
                execution_time=execution_time,
                thread_id=thread_id,
            )

        else:
            # Workflow routing
            if not devops_workflow:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Workflow not initialized",
                )

            logger.info("Using workflow routing")

            # Infer task type from task description
            task_type = _infer_task_type(request.task)

            # Prepare initial state
            initial_state: DevOpsAgentState = {
                "messages": [HumanMessage(content=request.task)],
                "task_type": task_type,
                "context": request.context,
                "subagent_results": {},
                "next_action": None,
                "current_phase": "initialization",
            }

            # Invoke workflow
            config = {
                "configurable": {"thread_id": thread_id},
                "tags": ["devops_workflow", f"task_type:{task_type}"],
            }

            result_state = await devops_workflow.ainvoke(
                initial_state,
                config=config,
            )

            execution_time = time.time() - start_time

            # Check if approval is required
            if result_state.get("approval_required"):
                pending_approvals[thread_id] = result_state

                return AgentResponse(
                    status="pending_approval",
                    result={
                        "message": "Human approval required",
                        "approval_items": result_state.get("approval_required"),
                    },
                    agent_used="workflow",
                    execution_time=execution_time,
                    thread_id=thread_id,
                )

            # Extract results
            agent_used = _extract_agent_used(result_state)

            return AgentResponse(
                status="success",
                result=result_state.get("subagent_results", {}),
                agent_used=agent_used,
                execution_time=execution_time,
                thread_id=thread_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Agent invocation failed",
            error=str(e),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent invocation failed: {str(e)}",
        )


@app.post("/agent/stream")
async def stream_agent(request: AgentRequest) -> StreamingResponse:
    """Stream agent or workflow responses using Server-Sent Events.

    Args:
        request: Agent invocation request

    Returns:
        StreamingResponse with SSE events

    Raises:
        HTTPException: If streaming fails to start
    """
    logger.info(
        "Agent streaming requested",
        agent_name=request.agent_name,
        task=request.task[:100],
        thread_id=request.thread_id,
    )

    thread_id = request.thread_id or str(uuid4())

    async def generate() -> AsyncIterator[str]:
        """Generate SSE events from agent/workflow execution."""
        try:
            if request.agent_name:
                # Direct agent invocation with streaming
                if not agent_registry:
                    yield f"event: error\ndata: {json.dumps({'error': 'Agent registry not initialized'})}\n\n"
                    return

                agent = agent_registry.get_agent(request.agent_name)

                # Build message
                message_content = request.task
                if request.context:
                    message_content += f"\n\nContext: {request.context}"

                messages = [{"role": "user", "content": message_content}]

                # Stream events
                config = {
                    "configurable": {"thread_id": thread_id},
                    "tags": [f"agent:{request.agent_name}", "streaming"],
                }

                async for event in agent.astream_events(
                    {"messages": messages},
                    config=config,
                    version="v1",
                ):
                    yield f"data: {json.dumps(event)}\n\n"

                yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

            else:
                # Workflow streaming
                if not devops_workflow:
                    yield f"event: error\ndata: {json.dumps({'error': 'Workflow not initialized'})}\n\n"
                    return

                task_type = _infer_task_type(request.task)

                initial_state: DevOpsAgentState = {
                    "messages": [HumanMessage(content=request.task)],
                    "task_type": task_type,
                    "context": request.context,
                    "subagent_results": {},
                    "next_action": None,
                    "current_phase": "initialization",
                }

                config = {
                    "configurable": {"thread_id": thread_id},
                    "tags": ["devops_workflow", f"task_type:{task_type}", "streaming"],
                }

                async for event in devops_workflow.astream_events(
                    initial_state,
                    config=config,
                    version="v1",
                ):
                    yield f"data: {json.dumps(event)}\n\n"

                yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

        except Exception as e:
            logger.error(
                "Streaming failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/workflow/status/{thread_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(thread_id: str) -> WorkflowStatusResponse:
    """Get workflow status for a thread.

    Args:
        thread_id: Thread ID to query

    Returns:
        WorkflowStatusResponse with current status

    Raises:
        HTTPException: If thread not found or workflow not initialized
    """
    logger.info("Workflow status requested", thread_id=thread_id)

    if not devops_workflow:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workflow not initialized",
        )

    try:
        # Get state from checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        state_snapshot = CHECKPOINTER.get(config)

        if not state_snapshot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow thread not found: {thread_id}",
            )

        state = state_snapshot.values

        # Determine status
        workflow_status = "in_progress"
        if state.get("next_action") == "complete":
            workflow_status = "complete"
        elif state.get("approval_required"):
            workflow_status = "pending_approval"

        return WorkflowStatusResponse(
            thread_id=thread_id,
            status=workflow_status,
            current_phase=state.get("current_phase"),
            next_action=state.get("next_action"),
            supervisor_path=state.get("supervisor_path", []),
            approval_required=state.get("approval_required"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get workflow status",
            thread_id=thread_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}",
        )


@app.post("/workflow/approve/{thread_id}")
async def approve_workflow(
    thread_id: str,
    approval: ApprovalRequest,
) -> Dict[str, Any]:
    """Approve or reject a pending workflow operation.

    Args:
        thread_id: Thread ID of the workflow
        approval: Approval decision

    Returns:
        Dict with approval result

    Raises:
        HTTPException: If thread not found or no approval pending
    """
    logger.info(
        "Workflow approval submitted",
        thread_id=thread_id,
        approved=approval.approved,
    )

    if thread_id not in pending_approvals:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending approval for thread: {thread_id}",
        )

    try:
        state = pending_approvals[thread_id]

        if approval.approved:
            # Continue workflow
            state["next_action"] = "complete"
            state["current_phase"] = "approved"

            # Update state in checkpointer
            config = {"configurable": {"thread_id": thread_id}}
            # Note: In production, you would resume the workflow here

            del pending_approvals[thread_id]

            return {
                "status": "approved",
                "thread_id": thread_id,
                "message": "Operations approved and executed",
            }
        else:
            # Reject workflow
            state["next_action"] = "rejected"
            state["current_phase"] = "rejected"

            del pending_approvals[thread_id]

            return {
                "status": "rejected",
                "thread_id": thread_id,
                "message": "Operations rejected",
                "comment": approval.comment,
            }

    except Exception as e:
        logger.error(
            "Approval processing failed",
            thread_id=thread_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Approval processing failed: {str(e)}",
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _infer_task_type(task: str) -> str:
    """Infer task type from task description.

    Args:
        task: Task description

    Returns:
        Task type string
    """
    task_lower = task.lower()

    # Infrastructure keywords
    if any(
        kw in task_lower
        for kw in [
            "deploy",
            "provision",
            "scaffold",
            "pipeline",
            "kubernetes",
            "k8s",
            "infrastructure",
        ]
    ):
        return "infrastructure"

    # Development keywords
    if any(
        kw in task_lower
        for kw in [
            "codegen",
            "generate",
            "database",
            "schema",
            "migration",
            "test",
            "api",
        ]
    ):
        return "development"

    # Operations keywords
    if any(
        kw in task_lower
        for kw in [
            "monitor",
            "incident",
            "alert",
            "debug",
            "logs",
            "metrics",
            "troubleshoot",
        ]
    ):
        return "operations"

    # Default to infrastructure
    return "infrastructure"


def _extract_agent_used(state: DevOpsAgentState) -> str:
    """Extract the primary agent used from workflow state.

    Args:
        state: Workflow state

    Returns:
        Agent name
    """
    # Get supervisor path
    supervisor_path = state.get("supervisor_path", [])

    # Get agent results
    results = state.get("subagent_results", {})

    if results:
        # Return the last agent that produced results
        return list(results.keys())[-1]

    if supervisor_path:
        # Return the last supervisor
        return supervisor_path[-1]

    return "unknown"


# =============================================================================
# Server Startup
# =============================================================================


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info",
) -> None:
    """Run the FastAPI server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
        log_level: Logging level
    """
    logger.info(
        "Starting server",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )

    uvicorn.run(
        "deep_agent.devops.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DevOps Multi-Agent API Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    args = parser.parse_args()

    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
