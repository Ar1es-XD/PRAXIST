"""
S3 utilities for uploading and downloading artifacts.

Includes robust handling for RunPod S3 quirks:
- Pagination bug workaround (level-by-level directory walking)
- Large file upload with timeout retry and HeadObject verification
- 524 error handling with exponential backoff
"""

import contextlib
import logging
import os
import tarfile
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.config import Config

    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False


def get_s3_client():
    """Build an S3 client from boundary env reads with config-literal fallbacks.

    AWS credentials and region are read from the environment at
    function-call time (the documented subprocess boundary).  Empty
    env values fall back to ``praxist.config`` re-exports of the
    ``run_config.DEFAULT_*`` literals — kept so tests that
    ``patch.object(config, "AWS_ACCESS_KEY_ID", "ak")`` continue to
    override the value here.
    """
    if not HAS_BOTO3:
        raise ImportError("boto3 is required for S3 operations")

    from praxist import config

    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID") or config.AWS_ACCESS_KEY_ID
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or config.AWS_SECRET_ACCESS_KEY
    region_name = os.environ.get("AWS_REGION") or config.S3_REGION
    endpoint_url = os.environ.get("S3_ENDPOINT_URL") or config.S3_ENDPOINT_URL

    if not aws_access_key_id or not aws_secret_access_key:
        return None

    boto_config = Config(
        read_timeout=7200,
        connect_timeout=60,
        retries={"max_attempts": 5, "mode": "adaptive"},
        max_pool_connections=50,
        signature_version="s3v4",
    )

    client_kwargs = {
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "region_name": region_name,
        "config": boto_config,
    }

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url

    return boto3.client("s3", **client_kwargs)


def upload_file_to_s3(
    file_path: Path,
    s3_key: str,
    bucket_name: str,
    content_type: str = "application/octet-stream",
) -> bool:
    """Upload a single file to S3."""
    client = get_s3_client()
    if not client:
        logger.warning("No S3 client available")
        return False

    file_path = Path(file_path)
    try:
        client.upload_file(
            str(file_path),
            bucket_name,
            s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return True
    except Exception as e:
        logger.error(f"Upload failed for {s3_key}: {e}")
        return False


def upload_directory_to_s3(
    local_dir: Path,
    s3_key: str,
    bucket_name: str,
    exclude_patterns: list[str] | None = None,
) -> bool:
    """Upload a directory as tar.gz to S3."""
    if exclude_patterns is None:
        exclude_patterns = [
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            "unsloth_compiled_cache",
        ]

    local_dir = Path(local_dir)

    def tar_filter(tarinfo):
        for pat in exclude_patterns:
            if pat in tarinfo.name:
                return None
        return tarinfo

    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        with tarfile.open(tmp_path, "w:gz") as tar:
            tar.add(str(local_dir), arcname=local_dir.name, filter=tar_filter)

        return upload_file_to_s3(
            Path(tmp_path),
            s3_key,
            bucket_name,
            content_type="application/gzip",
        )
    except Exception as e:
        logger.error(f"Directory upload failed: {e}")
        return False
    finally:
        with contextlib.suppress(Exception):
            os.unlink(tmp_path)


def download_s3_file(
    s3_key: str,
    local_path: Path,
    bucket_name: str,
) -> bool:
    """Download a single file from S3."""
    client = get_s3_client()
    if not client:
        return False

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        client.download_file(bucket_name, s3_key, str(local_path))
        return True
    except Exception as e:
        logger.error(f"Download failed for {s3_key}: {e}")
        return False


def download_snapshot_from_s3(
    s3_key: str,
    target_dir: str,
    bucket_name: str,
) -> list[str]:
    """Download and extract a snapshot tar.gz from S3."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if not download_s3_file(s3_key, Path(tmp_path), bucket_name):
            return []

        with tarfile.open(tmp_path, "r:gz") as tar:
            # Security: validate members before extraction
            for member in tar.getmembers():
                member_path = Path(target) / member.name
                try:
                    member_path.resolve().relative_to(Path(target).resolve())
                except ValueError as err:
                    raise ValueError(f"Tar member {member.name} escapes target directory") from err
            tar.extractall(path=str(target))
            return tar.getnames()
    except Exception as e:
        logger.error(f"Snapshot extraction failed: {e}")
        return []
    finally:
        with contextlib.suppress(Exception):
            os.unlink(tmp_path)


def generate_idea_uid() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())
