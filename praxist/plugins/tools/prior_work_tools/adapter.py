"""
MCP tool for downloading prior work snapshots.

Agents can request specific workspace snapshots to reference or build on.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from claude_agent_sdk import create_sdk_mcp_server, tool
except ImportError:
    tool = None
    create_sdk_mcp_server = None

from praxist.plugins.workflow_stages.research_loop.backend.tools.http_utils import (
    async_http_get,
    get_server_url,
    validate_safe_identifier,
    validate_safe_path,
)

logger = logging.getLogger(__name__)


def _text_result(data: Any) -> dict[str, Any]:
    text = json.dumps(data, indent=2, default=str) if not isinstance(data, str) else data
    return {"content": [{"type": "text", "text": text}]}


def _error_result(msg: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps({"error": msg})}], "is_error": True}


async def _handle_download_snapshot(args: dict[str, Any]) -> dict[str, Any]:
    """Download a workspace snapshot from a previous finding or generation."""
    snapshot_id = args["snapshot_id"]
    target_dir = args.get("target_dir", "")

    try:
        snapshot_id = validate_safe_identifier(snapshot_id, "snapshot_id")
    except ValueError as e:
        return _error_result(str(e))

    # Try server API first
    try:
        server_url = get_server_url()
        meta = await async_http_get(
            f"{server_url}/api/snapshots/{snapshot_id}",
            timeout=30,
        )
    except Exception as e:
        return _error_result(f"Could not fetch snapshot metadata: {e}")

    if "error" in meta:
        return _text_result(meta)

    # Determine target directory
    workspace = os.environ.get("WORKSPACE_DIR", "/workspace")
    if not target_dir:
        target_dir = f"{workspace}/downloaded_snapshots/{snapshot_id}"

    try:
        target_dir = validate_safe_path(target_dir, "target_dir", allowed_base=workspace)
    except ValueError as e:
        return _error_result(str(e))

    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Download from S3
    try:
        from praxist.infrastructure.s3_utils import (
            download_snapshot_from_s3,
        )

        s3_key = meta.get("snapshot_s3_key", "")
        if not s3_key:
            return _error_result("No S3 key in snapshot metadata")

        bucket = os.environ.get("S3_BUCKET", "")
        files = download_snapshot_from_s3(
            s3_key=s3_key,
            target_dir=target_dir,
            bucket_name=bucket,
        )

        return _text_result(
            {
                "status": "downloaded",
                "snapshot_id": snapshot_id,
                "target_dir": target_dir,
                "files_count": len(files) if files else 0,
                "variant_name": meta.get("variant_name", ""),
                "metrics": meta.get("metrics", {}),
            }
        )

    except ImportError:
        return _error_result("S3 utilities not available (boto3 not installed)")
    except Exception as e:
        return _error_result(f"Failed to download snapshot: {e}")


# ---------------------------------------------------------------------------
# Tool definition (new SDK API)
# ---------------------------------------------------------------------------

download_snapshot = None

if tool is not None:
    download_snapshot = tool(
        "download_snapshot",
        "Download a workspace snapshot from a previous finding or generation.",
        {
            "snapshot_id": str,
            "target_dir": str,
        },
    )(_handle_download_snapshot)


def create_prior_work_tools_server():
    """Create MCP server for prior work tools."""
    if create_sdk_mcp_server is None or tool is None:
        raise ImportError("claude_agent_sdk is required for MCP tools")
    return create_sdk_mcp_server(
        "prior-work-tools",
        tools=[download_snapshot],
    )


def create_tool_plugin() -> dict[str, object]:
    """Manifest entrypoint that exposes prior-work lookup tools."""
    return {
        "tool_server_ref": "tool_server:prior_work_tools",
        "server_name": "prior-work-tools",
        "factory": "praxist.plugins.tools.prior_work_tools.adapter:create_prior_work_tools_server",
        "tool_names": ["download_snapshot"],
        "visibility": ["peer"],
        "required_capability": "tool_server.prior_work_tools",
        "enabled_in_local_mode": False,
        "handlers": {
            "download_snapshot": "praxist.plugins.tools.prior_work_tools.adapter:_handle_download_snapshot",
        },
    }
