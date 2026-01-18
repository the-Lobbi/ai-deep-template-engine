"""Command-line entrypoint for running the MCP server."""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import uvicorn

from .harness_deep_agent import AgentConfig
from .mcp_server import create_mcp_app


def _env_or_default(env_name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(env_name)
    return value if value is not None else default


def _parse_enabled_subagents(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    stripped = raw_value.strip()
    if not stripped:
        return []
    return [name.strip() for name in stripped.split(",") if name.strip()]


def _build_config_from_args(args: argparse.Namespace) -> AgentConfig:
    harness_account_id = args.harness_account_id or _env_or_default("HARNESS_ACCOUNT_ID")
    harness_api_url = args.harness_api_url or _env_or_default("HARNESS_API_URL")
    harness_api_token = args.harness_api_token or _env_or_default("HARNESS_API_TOKEN")

    if not harness_account_id or not harness_api_url or not harness_api_token:
        raise SystemExit(
            "Missing required Harness configuration. Provide --harness-account-id, "
            "--harness-api-url, and --harness-api-token or set HARNESS_ACCOUNT_ID, "
            "HARNESS_API_URL, and HARNESS_API_TOKEN."
        )

    enabled_subagents = _parse_enabled_subagents(
        args.enabled_subagents or _env_or_default("DEEP_AGENT_ENABLED_SUBAGENTS")
    )

    return AgentConfig(
        harness_account_id=harness_account_id,
        harness_api_url=harness_api_url,
        harness_api_token=harness_api_token,
        org_identifier=args.org_identifier
        or _env_or_default("HARNESS_ORG_IDENTIFIER", "default"),
        project_identifier=args.project_identifier
        or _env_or_default("HARNESS_PROJECT_IDENTIFIER"),
        enabled_subagents=enabled_subagents,
        mcp_server_host=args.host or _env_or_default("DEEP_AGENT_MCP_HOST", "0.0.0.0"),
        mcp_server_port=int(
            args.port or _env_or_default("DEEP_AGENT_MCP_PORT", "8000")
        ),
        log_level=args.log_level or _env_or_default("DEEP_AGENT_LOG_LEVEL", "INFO"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Harness Deep Agent MCP server.")
    parser.add_argument("--harness-account-id", help="Harness account ID")
    parser.add_argument("--harness-api-url", help="Harness API base URL")
    parser.add_argument("--harness-api-token", help="Harness API token")
    parser.add_argument("--org-identifier", help="Harness org identifier")
    parser.add_argument("--project-identifier", help="Harness project identifier")
    parser.add_argument(
        "--enabled-subagents",
        help="Comma-separated list of enabled subagents. Empty string disables all.",
    )
    parser.add_argument("--host", help="MCP server host")
    parser.add_argument("--port", type=int, help="MCP server port")
    parser.add_argument("--log-level", help="Log level")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = _build_config_from_args(args)
    app = create_mcp_app(config)
    uvicorn.run(app, host=config.mcp_server_host, port=config.mcp_server_port)


if __name__ == "__main__":
    main()
