"""Prompt artifact persistence helpers for the research-loop stage."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    DERIVED_AUDIT_SNAPSHOT,
    attach_artifact_semantics,
)

logger = logging.getLogger(__name__)


def compact_artifact_ref(artifact: dict[str, Any]) -> dict[str, Any]:
    """Return the stable subset of an artifact record suitable for manifests."""
    return {
        key: artifact.get(key)
        for key in (
            "artifact_id",
            "artifact_type",
            "logical_path",
            "payload_path",
            "content_hash",
            "content_type",
            "schema_ref",
            "artifact_role",
            "artifact_status",
            "runtime_fact_source",
            "derived_from",
        )
        if key in artifact
    }


def persist_prompt_layout_artifacts(
    *,
    run_dir: Path,
    prompt_text: str,
    prompt_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    peer_id: str,
    gen_id: int,
) -> dict[str, Any]:
    """Persist PromptLayout V1 artifacts without blocking peer startup on failure."""
    try:
        from praxist.core.storage import ArtifactWriter, write_json
        from praxist.core.trajectory import TrajectoryWriter

        run_id = os.environ.get("PRAXIST_RUN_ID", run_dir.name)
        trajectory = TrajectoryWriter(run_dir, run_id)
        artifacts = ArtifactWriter(run_dir, trajectory)
        prompt_artifact = artifacts.persist_text(
            "prompt.rendered",
            f"prompts/gen_{gen_id}/{peer_id}.md",
            prompt_text,
            schema_ref="praxist.prompt.rendered.v1",
            producer={"stage_id": "research_loop", "role_ref": f"peer:{peer_id}"},
            content_type="text/markdown",
            artifact_role="audit_snapshot",
            artifact_status="committed",
            runtime_fact_source=False,
        )
        manifest_with_refs = attach_artifact_semantics(
            {
                **manifest,
                "rendered_prompt_ref": compact_artifact_ref(prompt_artifact),
                "rendered_prompt_path": str(prompt_path.relative_to(run_dir))
                if prompt_path.is_relative_to(run_dir)
                else str(prompt_path),
            },
            role=DERIVED_AUDIT_SNAPSHOT,
            stage="peer_prompt_layout",
            generation_id=gen_id,
            actor=f"peer:{peer_id}",
            canonical_sources=[
                "frontier/frontier_manifest.json",
                "gems/gems_state.json",
                "research_memory/*",
                "task_spec.yaml",
            ],
            runtime_fact_source=False,
        )
        write_json(manifest_path, manifest_with_refs)
        artifacts.persist_json(
            "prompt.layout_manifest",
            f"prompts/gen_{gen_id}/{peer_id}_layout.json",
            manifest_with_refs,
            schema_ref="praxist.prompt_layout.v1",
            producer={"stage_id": "research_loop", "role_ref": f"peer:{peer_id}"},
            source_artifact_ids=[prompt_artifact["artifact_id"]],
            artifact_role="derived_audit_snapshot",
            artifact_status="committed",
            runtime_fact_source=False,
            derived_from=[prompt_artifact["artifact_id"]],
        )
        return manifest_with_refs
    except Exception as exc:  # noqa: BLE001 - observability must not block peers.
        logger.warning("prompt layout artifact persistence failed for %s: %s", peer_id, exc)
        return manifest
