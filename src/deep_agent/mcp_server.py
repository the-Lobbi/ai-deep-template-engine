"""FastAPI-based MCP server for the Harness Deep Agent."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .version import __version__
from .harness_deep_agent import AgentConfig, HarnessDeepAgent

logger = logging.getLogger(__name__)


class JsonRpcRequest(BaseModel):
    """Minimal JSON-RPC 2.0 request model."""

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None


TOOL_DEFINITIONS = [
    {
        "name": "create_repository",
        "description": "Create a repository in Harness Code.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_name": {"type": "string"},
                "project_identifier": {"type": "string"},
                "description": {"type": "string"},
                "default_branch": {"type": "string", "default": "main"},
            },
            "required": ["repo_name", "project_identifier"],
        },
    },
    {
        "name": "create_pipeline",
        "description": "Create a CI/CD pipeline in Harness.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pipeline_name": {"type": "string"},
                "project_identifier": {"type": "string"},
                "pipeline_yaml": {"type": "string"},
            },
            "required": ["pipeline_name", "project_identifier", "pipeline_yaml"],
        },
    },
]


def create_mcp_app(config: AgentConfig) -> FastAPI:
    """Create a FastAPI MCP server wired to a HarnessDeepAgent instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent = HarnessDeepAgent(config)
        app.state.agent = agent
        logger.info(
            "Starting MCP server for HarnessDeepAgent on %s:%s",
            config.mcp_server_host,
            config.mcp_server_port,
        )
        yield
        await agent.client.aclose()

    app = FastAPI(title="Harness Deep Agent MCP Server", lifespan=lifespan)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/info")
    async def info() -> Dict[str, Any]:
        return {
            "name": "harness-deep-agent",
            "version": __version__,
            "tools": [tool["name"] for tool in TOOL_DEFINITIONS],
        }

    @app.post("/mcp")
    async def handle_mcp(request: Request) -> JSONResponse:
        payload = await request.json()
        rpc_request = JsonRpcRequest.model_validate(payload)
        response = await _dispatch_mcp_request(app, rpc_request)
        return JSONResponse(response.model_dump(exclude_none=True))

    return app


async def _dispatch_mcp_request(app: FastAPI, rpc_request: JsonRpcRequest) -> JsonRpcResponse:
    if rpc_request.method == "initialize":
        return JsonRpcResponse(
            id=rpc_request.id,
            result={
                "serverInfo": {
                    "name": "harness-deep-agent",
                    "version": __version__,
                },
                "capabilities": {
                    "tools": {},
                },
            },
        )

    if rpc_request.method == "tools/list":
        return JsonRpcResponse(id=rpc_request.id, result={"tools": TOOL_DEFINITIONS})

    if rpc_request.method == "tools/call":
        params = rpc_request.params or {}
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        if not tool_name:
            return JsonRpcResponse(
                id=rpc_request.id,
                error=JsonRpcError(code=-32602, message="Missing tool name."),
            )
        try:
            result = await _call_tool(app, tool_name, arguments)
        except ValueError as exc:
            return JsonRpcResponse(
                id=rpc_request.id,
                error=JsonRpcError(code=-32602, message=str(exc)),
            )
        return JsonRpcResponse(id=rpc_request.id, result=result)

    return JsonRpcResponse(
        id=rpc_request.id,
        error=JsonRpcError(code=-32601, message=f"Unknown method {rpc_request.method}."),
    )


async def _call_tool(app: FastAPI, tool_name: str, arguments: Dict[str, Any]) -> Any:
    agent: HarnessDeepAgent = app.state.agent
    if tool_name == "create_repository":
        repo_name = arguments.get("repo_name")
        project_identifier = arguments.get("project_identifier")
        if not repo_name or not project_identifier:
            raise ValueError("repo_name and project_identifier are required.")
        description = arguments.get("description", "")
        default_branch = arguments.get("default_branch", "main")
        return await agent.create_repository(
            repo_name=repo_name,
            project_identifier=project_identifier,
            description=description,
            default_branch=default_branch,
        )

    if tool_name == "create_pipeline":
        pipeline_name = arguments.get("pipeline_name")
        project_identifier = arguments.get("project_identifier")
        pipeline_yaml = arguments.get("pipeline_yaml")
        if not pipeline_name or not project_identifier or not pipeline_yaml:
            raise ValueError(
                "pipeline_name, project_identifier, and pipeline_yaml are required."
            )
        return await agent.create_pipeline(
            pipeline_name=pipeline_name,
            project_identifier=project_identifier,
            pipeline_yaml=pipeline_yaml,
        )

    raise ValueError(f"Unsupported tool: {tool_name}.")
