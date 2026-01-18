"""CLI runner for starting the Deep Agent MCP server."""

import argparse
import os
from typing import Optional

from .devops.server import run_server
from .harness_deep_agent import AgentConfig


def _get_env_var(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is not None and value.strip() == "":
        return None
    return value


def build_agent_config(args: argparse.Namespace) -> AgentConfig:
    """Build AgentConfig from CLI args and environment variables."""
    harness_account_id = args.harness_account_id or _get_env_var("HARNESS_ACCOUNT_ID")
    harness_api_url = args.harness_api_url or _get_env_var("HARNESS_API_URL")
    harness_api_token = args.harness_api_token or _get_env_var("HARNESS_API_TOKEN")
    org_identifier = args.org_identifier or _get_env_var("HARNESS_ORG_IDENTIFIER") or "default"
    project_identifier = (
        args.project_identifier or _get_env_var("HARNESS_PROJECT_IDENTIFIER")
    )
    log_level = args.log_level or _get_env_var("DEEP_AGENT_LOG_LEVEL") or "INFO"
    mcp_server_host = args.mcp_server_host or _get_env_var("MCP_SERVER_HOST") or "0.0.0.0"
    mcp_server_port = args.mcp_server_port or int(_get_env_var("MCP_SERVER_PORT") or 8000)

    if not harness_account_id:
        raise ValueError("HARNESS_ACCOUNT_ID is required (via --harness-account-id or env)")
    if not harness_api_url:
        raise ValueError("HARNESS_API_URL is required (via --harness-api-url or env)")
    if not harness_api_token:
        raise ValueError("HARNESS_API_TOKEN is required (via --harness-api-token or env)")

    return AgentConfig(
        harness_account_id=harness_account_id,
        harness_api_url=harness_api_url,
        harness_api_token=harness_api_token,
        org_identifier=org_identifier,
        project_identifier=project_identifier,
        mcp_server_host=mcp_server_host,
        mcp_server_port=mcp_server_port,
        log_level=log_level,
    )


def main() -> None:
    """Entry point for running the MCP server."""
    parser = argparse.ArgumentParser(description="Deep Agent MCP Server Runner")
    parser.add_argument("--harness-account-id", type=str, help="Harness account ID")
    parser.add_argument("--harness-api-url", type=str, help="Harness API base URL")
    parser.add_argument("--harness-api-token", type=str, help="Harness API token")
    parser.add_argument("--org-identifier", type=str, help="Harness org identifier")
    parser.add_argument("--project-identifier", type=str, help="Harness project identifier")
    parser.add_argument(
        "--mcp-server-host",
        type=str,
        help="Host for the MCP server (defaults to MCP_SERVER_HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--mcp-server-port",
        type=int,
        help="Port for the MCP server (defaults to MCP_SERVER_PORT or 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()
    config = build_agent_config(args)
    run_server(
        host=config.mcp_server_host,
        port=config.mcp_server_port,
        reload=args.reload,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()
