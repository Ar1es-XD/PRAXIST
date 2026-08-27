"""Single-source-of-truth run configuration captured at the CLI boundary.

This module is the **schema** for the configuration discipline documented
in :doc:`/concepts/config_discipline`.  The principle:

> Core domain code does **not** read environment variables.  Env reads
> happen at the **boundaries** — CLI entry points on the way in and subprocess output
> construction (``Popen(env=…)``) on the way out.

Today this contract is partially honored: ``CredentialRef`` /
``AgentRunRequest`` / ``ModelCallSpec`` already collect boundary
strings into frozen dataclasses, but ~50 ``os.environ.get`` calls
remain in core-adjacent code (``backend/agent.py``, ``startup.py``,
``execute_autonomous.py``, …).  See ``docs/concepts/config_discipline.md``
for the migration roadmap.

:class:`RunConfig` is the destination dataclass.  CLI entrypoints build
one via :meth:`RunConfig.from_environ` (merging argparse output and
``os.environ``); downstream code reads ``cfg.run_id`` etc. instead of
``os.environ.get("PRAXIST_RUN_ID", …)``.

The dataclass is intentionally minimal in v0.1: it covers the cluster
of fields whose env-based transport caused recent bugs
(``AGENT_MODEL`` carrying an openrouter prefix into DeepSeek, the
``"/" in model`` heuristic in ``_legacy_model_provider_ref``, the
``prompt_ref`` text/metadata confusion).  Additional fields are added
in subsequent migration PRs, one env var at a time.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_AGENT_MODEL = "claude-opus-4-7"
"""Hard-coded model fallback used when no caller-supplied model is provided.

Replaces the historical ``praxist.config.AGENT_MODEL`` env-read
constant (#75 batch 7a). Production callers receive the model via
``RunConfig.from_environ`` at the CLI boundary; this constant only
serves test fixtures and the last-resort default in
``_default_model_for_provider`` for unrecognized provider refs.
"""

_DEFAULT_OUTPUT_ROOT = f"{tempfile.gettempdir()}/praxist"

DEFAULT_LOGS_DIR = f"{_DEFAULT_OUTPUT_ROOT}/logs"
"""Hard-coded logs-directory fallback used when no caller / env supplies one.

Replaces ``praxist.config.LOGS_DIR`` (#75 batch 7b). Production
callers thread the logs dir explicitly; runtime telemetry modules
(``usage_tracker``, ``tool_timing``) keep their ``os.environ.get("LOGS_DIR", …)``
boundary reads but fall back to this literal instead of an
import-time env read in ``config.py``.
"""

DEFAULT_LOCAL_FINDINGS_DIR = f"{_DEFAULT_OUTPUT_ROOT}/shared_findings"
"""Hard-coded shared-findings dir fallback (#75 batch 7b).

Replaces ``praxist.config.LOCAL_FINDINGS_DIR``.
"""

DEFAULT_FINDINGS_POLL_INTERVAL_SECONDS = 60
"""Default findings-sync poll cadence in seconds (#75 batch 7b).

Replaces ``praxist.config.FINDINGS_POLL_INTERVAL``. The
``FindingsSync`` daemon polls the local findings dir at this cadence
when running in non-local mode.
"""

DEFAULT_WORKSPACE_ROOT = str(Path(__file__).resolve().parents[2])
"""Repository root used as a last-resort workspace fallback (#75 batch 7c).

Replaces ``praxist.config.WORKSPACE_DIR``. Production callers
thread the workspace explicitly via CLI / RunConfig.workspace_root;
this literal only serves direct callers that omit the kwarg (mostly
tests). The value is the repo root the install was loaded from —
the same answer the old env-read produced for fresh installs.
"""

DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS = 5 * 24 * 3600
"""Peer safety-cap fallback in seconds (#75 batch 7c).

Replaces ``praxist.config.FULL_AUTO_MAX_RUNTIME_SECONDS``. The
real per-peer cap comes from the task descriptor or CLI; this literal
only covers direct-call paths that don't supply one.
"""

# ── S3 / AWS literal defaults (#75 batch 8a) ─────────────────────────
#
# Replace ``os.getenv`` import-time reads in ``praxist.config``
# (``S3_BUCKET`` / ``S3_ENDPOINT_URL`` / ``S3_REGION`` / ``S3_*_PREFIX``
# / ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``).  ``config.py``
# re-exports each one so legacy callers and the operational-surface
# contract test (which patches ``config.S3_BUCKET`` etc.) keep
# working; the env reads that production used to rely on at import
# time move to the function-level boundary in
# :mod:`praxist.infrastructure.s3_utils` and
# :mod:`praxist.plugins.workflow_stages.research_loop.backend.agent`
# (already-audited modules; new allowlist entries land alongside this
# change).

DEFAULT_S3_BUCKET = ""
"""S3 bucket name fallback (empty)."""

DEFAULT_S3_ENDPOINT_URL = ""
"""Custom S3 endpoint URL fallback (empty → boto3 picks AWS default)."""

DEFAULT_S3_REGION = "us-east-1"
"""AWS region fallback when ``AWS_REGION`` env is not set."""

DEFAULT_S3_IDEAS_PREFIX = "ideas/"
"""S3 key prefix for ideation artifacts."""

DEFAULT_S3_RESULTS_PREFIX = "results/"
"""S3 key prefix for run-result artifacts."""

DEFAULT_S3_FRONTIER_PREFIX = "frontier/"
"""S3 key prefix for frontier artifacts."""

DEFAULT_AWS_ACCESS_KEY_ID = ""
"""AWS access key id fallback (empty → no S3 client built)."""

DEFAULT_AWS_SECRET_ACCESS_KEY = ""
"""AWS secret access key fallback (empty → no S3 client built)."""

# ── Cohort / frontier defaults (#75 batch 8b) ────────────────────────
#
# Conservative fallback values for direct/legacy callers. Real task
# projects supply ``generation_policy`` via ``task.yaml``; CLI flags
# (``praxist start --cohort N --generations M --strategy S``) override
# per-run. These literals only matter when nothing else fires.
#
# Env names migrated from ``praxist.config``'s bare ``MAX_GENERATIONS``
# / ``COHORT_SIZE`` / ``PER_GENERATION_HOURS`` / ``PROMOTE_TOP_K`` /
# ``FRONTIER_STRATEGY`` reads. The two of these that actually have a
# Python consumer (``MAX_GENERATIONS`` / ``COHORT_SIZE``, read by
# ``research_loop.startup._apply_internal_env_overrides``) now prefer
# the ``PRAXIST_*`` env name with the bare name kept as a legacy fallback;
# the other three are pure literal defaults today (no env reader in
# ``praxist/``).

DEFAULT_MAX_GENERATIONS = 1
"""Generation count fallback for cohorts without a task-descriptor value."""

DEFAULT_COHORT_SIZE = 5
"""Peer count per generation fallback."""

DEFAULT_PER_GENERATION_HOURS = 5
"""Per-generation safety cap in hours."""

DEFAULT_PROMOTE_TOP_K = 2
"""Frontier promotions per generation (top-K)."""

DEFAULT_FRONTIER_STRATEGY = "auto"
"""Frontier exploration strategy; ``auto`` = gen0 explore, gen≥1 PI-directed."""


@dataclass(frozen=True)
class RunConfig:
    """Frozen run-level configuration assembled at the CLI boundary.

    Every field is explicit; defaults exist only so a partial config
    (e.g. a unit test that only cares about ``model``) is ergonomic.
    The CLI layer is responsible for populating every field that the
    downstream workflow actually consumes; missing values surface as
    explicit ``None`` rather than mysterious env-default fallbacks.

    Attributes:
        run_id: Stable identifier for this run.  Populated from the
            ``PRAXIST_RUN_ID`` env var or generated by the CLI entrypoint.
        run_dir: Absolute path to the run output directory.
        stage_id: Workflow stage identifier (e.g. ``research_loop``).
        role_ref: Role plugin reference for the active agent, if any.
        agent_runtime_ref: Resolved AgentRuntime plugin reference
            (e.g. ``agent_runtime:codex_sdk``).  This is the **resolved**
            ref, not the user-facing ``PRAXIST_AGENT_SYSTEM`` short name.
        agent_system: User-facing peer runtime label
            (``claude_sdk`` / ``codex_sdk``).
            Kept alongside ``agent_runtime_ref`` for replay readability.
        model_provider_ref: Resolved ModelProvider plugin reference.
        model_profile_ref: Logical model profile (``cheap_peer`` /
            ``strong_reasoner`` / …); the provider plugin maps it to
            a concrete model name.
        model: Concrete model name to send to the runtime (already
            normalized for the provider's ``api_format`` —
            see :func:`praxist.core.modeling.normalize_model_for_provider`).
        model_credential_key_id: Key id of the active model-provider
            credential, when the CLI selected one. Empty string when the
            run is credential-less (e.g. resolve-only smoke tests).
        workspace_root: Absolute path to the workspace.
        task_project_path: Absolute path to the task project root.
        budget_grant_id: Active budget grant, if any.
        budget_request_id: Active budget request, if any.
        codex_bin: Optional Codex binary override. The SDK runtime otherwise
            uses the binary bundled with ``openai-codex``.
    """

    run_id: str = ""
    run_dir: Path | None = None
    stage_id: str = ""
    role_ref: str | None = None
    agent_runtime_ref: str = ""
    agent_system: str = ""
    model_provider_ref: str = ""
    model_profile_ref: str = ""
    model: str = ""
    model_credential_key_id: str = ""
    workspace_root: Path | None = None
    task_project_path: Path | None = None
    budget_grant_id: str = ""
    budget_request_id: str = ""
    codex_bin: str = "codex"

    # Free-form passthrough for subprocess env construction.  Holds the
    # subset of env vars the CLI deliberately propagates to children
    # (provider API keys, PRAXIST_TASK_*, etc.).  Read-only after CLI build.
    subprocess_env: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_environ(
        cls,
        env: Mapping[str, str],
        *,
        overrides: Mapping[str, object] | None = None,
    ) -> RunConfig:
        """Build a :class:`RunConfig` from an env-var mapping.

        This is the **only** sanctioned bridge between ``os.environ``
        and the :class:`RunConfig` schema.  CLI entrypoints should call
        ``RunConfig.from_environ(os.environ, overrides={...args...})``
        once and pass the result downward; no other code should read
        from ``env`` to build config.

        Args:
            env: Source mapping (typically ``os.environ`` at the CLI
                entrypoint).
            overrides: Optional dict of explicit field values that win
                over the env lookup.  Mirrors the way argparse flags
                override the corresponding env var (``--model`` beats
                ``AGENT_MODEL``).  Unknown keys raise ``TypeError`` via
                :func:`dataclasses.replace`.

        Returns:
            A frozen :class:`RunConfig`.
        """
        overrides = dict(overrides or {})

        def _path(value: str) -> Path | None:
            value = value.strip()
            return Path(value).expanduser() if value else None

        cfg = cls(
            run_id=env.get("PRAXIST_RUN_ID", ""),
            run_dir=_path(env.get("PRAXIST_RUN_DIR", "")),
            stage_id=env.get("PRAXIST_STAGE_ID", ""),
            role_ref=env.get("PRAXIST_ROLE_REF") or None,
            agent_runtime_ref=env.get("PRAXIST_AGENT_RUNTIME_REF", ""),
            agent_system=env.get("PRAXIST_AGENT_SYSTEM", ""),
            model_provider_ref=env.get("PRAXIST_MODEL_PROVIDER_REF", ""),
            model_profile_ref=env.get("PRAXIST_MODEL_PROFILE_REF", ""),
            # PRAXIST_MODEL is authoritative; AGENT_MODEL remains a
            # task-launcher compatibility input.
            model=env.get("PRAXIST_MODEL") or env.get("AGENT_MODEL", ""),
            model_credential_key_id=env.get("PRAXIST_MODEL_CREDENTIAL_KEY_ID", ""),
            workspace_root=_path(env.get("PRAXIST_WORKSPACE_ROOT", "")),
            task_project_path=_path(env.get("PRAXIST_TASK_PROJECT_PATH", "")),
            budget_grant_id=env.get("PRAXIST_BUDGET_GRANT_ID", ""),
            budget_request_id=env.get("PRAXIST_BUDGET_REQUEST_ID", ""),
            codex_bin=env.get("PRAXIST_CODEX_BIN", "codex"),
        )
        if overrides:
            cfg = replace(cfg, **overrides)
        return cfg

    def with_subprocess_env(self, env: Mapping[str, str]) -> RunConfig:
        """Return a new :class:`RunConfig` with ``subprocess_env`` replaced.

        Subprocess env construction is the CLI's job: pick the subset
        of the parent env (provider API keys, propagated PRAXIST_TASK_*
        vars, …) and freeze it on the config.  Downstream code that
        spawns subprocesses reads ``cfg.subprocess_env`` instead of
        ``os.environ.copy()``.
        """
        return replace(self, subprocess_env=dict(env))


__all__ = [
    "DEFAULT_AGENT_MODEL",
    "DEFAULT_AWS_ACCESS_KEY_ID",
    "DEFAULT_AWS_SECRET_ACCESS_KEY",
    "DEFAULT_COHORT_SIZE",
    "DEFAULT_FINDINGS_POLL_INTERVAL_SECONDS",
    "DEFAULT_FRONTIER_STRATEGY",
    "DEFAULT_FULL_AUTO_MAX_RUNTIME_SECONDS",
    "DEFAULT_LOCAL_FINDINGS_DIR",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_MAX_GENERATIONS",
    "DEFAULT_PER_GENERATION_HOURS",
    "DEFAULT_PROMOTE_TOP_K",
    "DEFAULT_S3_BUCKET",
    "DEFAULT_S3_ENDPOINT_URL",
    "DEFAULT_S3_FRONTIER_PREFIX",
    "DEFAULT_S3_IDEAS_PREFIX",
    "DEFAULT_S3_REGION",
    "DEFAULT_S3_RESULTS_PREFIX",
    "DEFAULT_WORKSPACE_ROOT",
    "RunConfig",
]
