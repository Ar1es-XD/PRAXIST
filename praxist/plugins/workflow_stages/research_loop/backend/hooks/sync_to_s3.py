"""
PostToolUse hook — sync findings and logs to S3 after each tool use.

Fails silently to not block the agent.
"""

import contextlib
import os
from pathlib import Path


def sync():
    """Sync key files to S3."""
    peer_id = os.environ.get("PEER_ID")
    run_id = os.environ.get("RUN_ID")
    bucket = os.environ.get("S3_BUCKET")

    if not peer_id or not run_id or not bucket:
        return

    try:
        from praxist.infrastructure.s3_utils import upload_file_to_s3
    except ImportError:
        return

    prefix = os.environ.get("S3_RESULTS_PREFIX", "results/")
    s3_prefix = f"{prefix}{peer_id}/{run_id}/"
    logs_dir = Path(os.environ.get("LOGS_DIR", "logs"))

    # Sync findings
    findings = logs_dir.parent / "findings.json"
    if findings.exists():
        with contextlib.suppress(Exception):
            upload_file_to_s3(
                file_path=findings,
                s3_key=f"{s3_prefix}findings.json",
                bucket_name=bucket,
                content_type="application/json",
            )

    # Sync usage stats
    usage = logs_dir / "usage_stats.json"
    if usage.exists():
        with contextlib.suppress(Exception):
            upload_file_to_s3(
                file_path=usage,
                s3_key=f"{s3_prefix}usage_stats.json",
                bucket_name=bucket,
                content_type="application/json",
            )

    # Sync session logs
    if logs_dir.exists():
        for log_file in logs_dir.glob("session_*.log"):
            with contextlib.suppress(Exception):
                upload_file_to_s3(
                    file_path=log_file,
                    s3_key=f"{s3_prefix}logs/{log_file.name}",
                    bucket_name=bucket,
                    content_type="text/plain",
                )


if __name__ == "__main__":
    sync()
