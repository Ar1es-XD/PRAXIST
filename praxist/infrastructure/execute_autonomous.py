"""
Entrypoint for autonomous research peers (RunPod / Docker).

Downloads prerequisites, launches the autonomous agent loop,
uploads final artifacts to S3.

Issue #75 batch 5: the seven env reads that historically lived inside
``main()`` (``PEER_ID`` / ``GENERATION_ID`` / ``MAX_RUNTIME_SECONDS`` /
``LOCAL_MODE`` / ``AGENT_MODEL`` / ``TASK_PROMPT`` / ``TASK_PROMPT_FILE``
/ ``LOGS_DIR``) are now collected into :class:`PeerInvocationConfig`. The
boundary ``main()`` builds one via ``PeerInvocationConfig.from_environ``
and hands off to :func:`run_peer`, which has no env reads of its own
so in-process callers (tests, future ``cmd_peer`` direct path) can
bypass the env round-trip.
"""

import asyncio
import json
import logging
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PeerInvocationConfig:
    """Frozen peer-invocation parameters captured at the subprocess boundary.

    ``execute_autonomous.main`` is the documented entry point for an
    autonomous peer subprocess; the orchestrator (or ``praxist peer``) sets
    a handful of env vars to parameterize the peer's run. This dataclass
    is the single landing pad for those reads — downstream code reads
    fields off the config instead of touching ``os.environ`` directly.

    Defaults mirror what ``main()`` used to apply when an env var was
    absent so a partial config (e.g. in unit tests) is ergonomic.
    """

    peer_id: str = "peer_0"
    generation_id: int = 0
    max_runtime_seconds: int = 24 * 3600
    local_mode: bool = False
    model: str = ""
    task_prompt: str = ""
    task_prompt_file: str = ""
    logs_dir: Path = Path("logs")

    @classmethod
    def from_environ(cls, env: Mapping[str, str]) -> "PeerInvocationConfig":
        """Build a :class:`PeerInvocationConfig` from an env mapping.

        Mirrors the env-reading semantics that ``main()`` historically
        applied: integer fields default on parse failure, ``LOCAL_MODE``
        accepts ``1`` / ``true`` / ``yes`` (case-insensitive), and
        ``LOGS_DIR`` falls back to a relative ``"logs"`` directory.
        """

        def _int(name: str, default: int) -> int:
            try:
                return int(env.get(name, str(default)))
            except (TypeError, ValueError):
                return default

        local_mode_raw = (env.get("LOCAL_MODE") or "false").lower()
        return cls(
            peer_id=env.get("PEER_ID", "peer_0"),
            generation_id=_int("GENERATION_ID", 0),
            max_runtime_seconds=_int("MAX_RUNTIME_SECONDS", 24 * 3600),
            local_mode=local_mode_raw in ("1", "true", "yes"),
            model=env.get("AGENT_MODEL", ""),
            task_prompt=env.get("TASK_PROMPT", ""),
            task_prompt_file=env.get("TASK_PROMPT_FILE", ""),
            logs_dir=Path(env.get("LOGS_DIR", "logs")),
        )

    def resolve_task_prompt(self) -> str | None:
        """Resolve the effective task prompt or ``None`` when missing.

        Inline ``task_prompt`` wins over ``task_prompt_file``. When both
        are empty the peer has nothing to do — callers should treat
        ``None`` as a fatal startup error.
        """
        if self.task_prompt:
            return self.task_prompt
        if self.task_prompt_file:
            path = Path(self.task_prompt_file)
            if path.exists():
                return path.read_text()
        return None


class TeeOutput:
    """Write to both a stream and a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


async def launch_autonomous_loop(
    peer_id: str,
    generation_id: int,
    task_prompt: str,
    max_runtime_seconds: int,
    local_mode: bool = False,
    model: str = "",
):
    """Launch the autonomous agent loop."""
    from praxist.core.tool_servers import (
        DEFAULT_PEER_TOOL_SERVER_REFS,
        base_peer_allowed_tools,
        build_legacy_mcp_servers,
    )
    from praxist.plugins.workflow_stages.research_loop.backend.agent import AutonomousAgentLoop

    build = build_legacy_mcp_servers(
        DEFAULT_PEER_TOOL_SERVER_REFS,
        run_dir=os.environ.get("PRAXIST_RUN_DIR") or os.environ.get("RUN_DIR"),
        local_mode=local_mode,
        multi_pi_enabled=False,
    )
    for item in build.unavailable:
        logger.warning("%s unavailable: %s", item["server_name"], item["reason"])
    mcp_servers = build.servers

    loop = AutonomousAgentLoop(
        peer_id=peer_id,
        generation_id=generation_id,
        task_prompt=task_prompt,
        max_runtime_seconds=max_runtime_seconds,
        local_mode=local_mode,
        model=model,
        mcp_servers=mcp_servers,
        allowed_tools=base_peer_allowed_tools(mcp_servers.keys()),
    )
    return await loop.run()


def upload_final_artifacts(
    peer_id: str,
    run_id: str,
    result: dict,
):
    """Upload final artifacts to S3.

    Reads ``S3_BUCKET`` / ``S3_RESULTS_PREFIX`` / ``LOGS_DIR`` env at
    function-call time (this module IS the subprocess boundary; #75
    batch 5 collected the other reads into ``PeerInvocationConfig``,
    these three stay function-local so the operational-surface
    contract test can still ``patch.object(config, …)`` to override).
    """
    from praxist import config
    from praxist.infrastructure.s3_utils import upload_file_to_s3

    s3_bucket = os.environ.get("S3_BUCKET") or config.S3_BUCKET
    s3_results_prefix = os.environ.get("S3_RESULTS_PREFIX") or config.S3_RESULTS_PREFIX
    logs_dir = os.environ.get("LOGS_DIR") or config.LOGS_DIR

    if not s3_bucket:
        logger.info("No S3 bucket configured, skipping artifact upload")
        return

    s3_prefix = f"{s3_results_prefix}{peer_id}/{run_id}/"

    # Upload result summary
    result_path = Path(logs_dir) / "run_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    upload_file_to_s3(
        result_path,
        f"{s3_prefix}run_result.json",
        s3_bucket,
        "application/json",
    )

    logger.info(f"Uploaded final artifacts to s3://{s3_bucket}/{s3_prefix}")


def run_peer(config: PeerInvocationConfig) -> None:
    """Drive one peer from an explicit :class:`PeerInvocationConfig`.

    Does no env reads of its own — used by ``main()`` (after building
    the config from env) and is available to in-process callers that
    have a config in hand. ``sys.exit`` is reserved for fatal startup
    errors that can't be repaired (missing prompt; loop exception).
    """
    task_prompt = config.resolve_task_prompt()
    if task_prompt is None:
        logger.error("No TASK_PROMPT or TASK_PROMPT_FILE provided")
        sys.exit(1)

    logs_dir = config.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "autonomous_worker.log"

    with open(log_path, "w") as log_f:
        sys.stdout = TeeOutput(sys.__stdout__, log_f)
        sys.stderr = TeeOutput(sys.__stderr__, log_f)

        try:
            result = asyncio.run(
                launch_autonomous_loop(
                    peer_id=config.peer_id,
                    generation_id=config.generation_id,
                    task_prompt=task_prompt,
                    max_runtime_seconds=config.max_runtime_seconds,
                    local_mode=config.local_mode,
                    model=config.model,
                )
            )

            if not config.local_mode:
                run_id = result.get("run_id", "unknown")
                upload_final_artifacts(config.peer_id, run_id, result)

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            sys.exit(1)
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


def main():
    """Main entrypoint for autonomous peer execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    config = PeerInvocationConfig.from_environ(os.environ)
    run_peer(config)


if __name__ == "__main__":
    main()
