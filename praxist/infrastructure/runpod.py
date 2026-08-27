"""
RunPod deployment utilities.

Deploys agent pods with configurable commands, environment variables,
GPU types, and datacenters.
"""

import logging
import os
import shlex
from typing import Any

logger = logging.getLogger(__name__)

RUNPOD_API_URL = "https://rest.runpod.io/v1/pods"

DEFAULT_DATACENTER_IDS = [
    "US-TX-3",
    "US-KS-2",
    "CA-MTL-1",  # North America
    "EU-SE-1",
    "EU-RO-1",  # Europe
    "JP-TY-1",  # Asia
]


class RunPodCapacityError(Exception):
    """Transient — capacity may free up, worth retrying."""

    pass


class RunPodPermanentError(Exception):
    """Permanent — fail immediately."""

    pass


def create_run_command(
    entrypoint_cmd: str,
    env_vars: dict[str, str] | None = None,
) -> str:
    """Build a bash command with environment variable exports."""
    parts = ["#!/bin/bash", "set -e"]
    if env_vars:
        for key, value in env_vars.items():
            parts.append(f"export {shlex.quote(key)}={shlex.quote(value)}")
    parts.append(entrypoint_cmd)
    return "\n".join(parts)


def deploy_pod(
    name: str,
    image: str,
    command: str,
    gpu_type: str | None = None,
    gpu_count: int = 1,
    api_key: str | None = None,
    datacenter_ids: list[str] | None = None,
    volume_size_gb: int = 100,
    env_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Deploy a pod to RunPod."""
    try:
        import requests
    except ImportError as err:
        raise ImportError("requests is required for RunPod deployment") from err

    api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        raise RunPodPermanentError("RUNPOD_API_KEY not set")
    if not gpu_type:
        raise RunPodPermanentError(
            "gpu_type must be selected explicitly from currently available provider capacity"
        )

    datacenter_ids = datacenter_ids or DEFAULT_DATACENTER_IDS

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "name": name,
        "imageName": image,
        "gpuTypeId": gpu_type,
        "gpuCount": gpu_count,
        "volumeInGb": volume_size_gb,
        "containerDiskInGb": 50,
        "startSsh": True,
        "dockerArgs": command,
        "env": env_vars or {},
        "dataCenterId": datacenter_ids,
    }

    response = requests.post(
        RUNPOD_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if response.status_code == 200:
        data = response.json()
        logger.info(f"Pod deployed: {data.get('id', 'unknown')}")
        return data

    if response.status_code == 429 or "capacity" in response.text.lower():
        raise RunPodCapacityError(f"No capacity: {response.text}")

    raise RunPodPermanentError(f"Deploy failed ({response.status_code}): {response.text}")


def get_pod_status(pod_id: str, api_key: str | None = None) -> dict[str, Any]:
    """Get pod status from RunPod."""
    import requests

    api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        f"{RUNPOD_API_URL}/{pod_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def stop_pod(pod_id: str, api_key: str | None = None) -> bool:
    """Stop a RunPod pod."""
    import requests

    api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(
        f"{RUNPOD_API_URL}/{pod_id}/stop",
        headers=headers,
        timeout=30,
    )
    return response.status_code == 200


def delete_pod(pod_id: str, api_key: str | None = None) -> bool:
    """Delete a RunPod pod permanently."""
    import requests

    api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.delete(
        f"{RUNPOD_API_URL}/{pod_id}",
        headers=headers,
        timeout=30,
    )
    return response.status_code == 200
