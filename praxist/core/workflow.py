"""Core workflow stage protocol helpers and optional-stage stubs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

OPTIONAL_STAGE_REFS = {
    "ideation": "workflow_stage:ideation_stub",
    "paper_writing": "workflow_stage:paper_writing_stub",
    "reviewer": "workflow_stage:reviewer_stub",
}

OPTIONAL_STAGE_CONTRACTS = {
    "ideation": {
        "stage_id": "ideation",
        "ref": "workflow_stage:ideation_stub",
        "inputs": ["seed_prompt", "constraints", "prior_artifacts"],
        "outputs": ["research_plan", "task_draft", "initial_agenda"],
        "artifact_type": "report",
        "schema_ref": "core:research_plan_stub.v1",
    },
    "paper_writing": {
        "stage_id": "paper_writing",
        "ref": "workflow_stage:paper_writing_stub",
        "inputs": ["findings", "frontier", "trajectory", "budget_ledger"],
        "outputs": ["manuscript_artifact"],
        "artifact_type": "report",
        "schema_ref": "core:manuscript_stub.v1",
    },
    "reviewer": {
        "stage_id": "reviewer",
        "ref": "workflow_stage:reviewer_stub",
        "inputs": [
            "manuscript_artifact",
            "evidence_refs",
            "artifact_index",
            "trajectory",
            "run_summary",
        ],
        "outputs": ["review_artifact", "reviewer_findings"],
        "artifact_type": "report",
        "schema_ref": "core:review_artifact.v1",
    },
}

LITERATURE_SCOUT_CONTRACT = {
    "role_slot": "literature_scout",
    "role_ref": "task_role:literature_scout",
    "role_scope": "task_project",
    "tool_ref": "tool_server:literature_lookup",
    "credential_scope": "none",
    "resource_policy": "current_environment_only",
    "evidence_schema_ref": "core:literature_evidence.v1",
    "enabled_by_default": False,
}


def disabled_optional_stages() -> list[dict[str, str | bool]]:
    """Return optional workflow stages disabled by a task descriptor."""
    return [
        {
            "stage_id": stage_id,
            "ref": stage_ref,
            "enabled": False,
            "reason": "disabled_by_default",
        }
        for stage_id, stage_ref in OPTIONAL_STAGE_REFS.items()
    ]


def disabled_optional_tools() -> list[dict[str, str | bool]]:
    """Return optional tool refs disabled by a task descriptor."""
    return [
        {
            "stage_id": "research_loop",
            "role_slot": "literature_scout",
            "role_ref": "task_role:literature_scout",
            "role_scope": "task_project",
            "tool_ref": "tool_server:literature_lookup",
            "enabled": False,
            "reason": "disabled_by_default",
        }
    ]


@dataclass(frozen=True)
class WorkflowStageSpec:
    """Declarative workflow stage contract resolved from plugin metadata."""

    stage_id: str
    ref: str
    enabled_by_default: bool
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    budget_policy_ref: str | None = None
    retry_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowStageResult:
    """Terminal result returned by a workflow stage entrypoint."""

    stage_id: str
    status: str
    success: bool
    output_artifacts: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptionalWorkflowStageContext:
    """Serializable description of an optional stage that may be disabled at startup."""

    run_dir: Path
    run_id: str
    enabled: bool = False
    mode: str = "disabled"
    reason: str = "disabled_by_default"
    source_event_ids: list[str] = field(default_factory=list)


class OptionalWorkflowStageStub:
    """Disabled-by-default contract stub for future optional workflow modules."""

    def __init__(self, stage_id: str) -> None:
        if stage_id not in OPTIONAL_STAGE_CONTRACTS:
            raise ValueError(f"Unknown optional workflow stage: {stage_id}")
        self.contract = OPTIONAL_STAGE_CONTRACTS[stage_id]
        self.stage_id = stage_id
        self.ref = str(self.contract["ref"])

    def describe(self, *, budget_policy_ref: str | None = None) -> WorkflowStageSpec:
        return WorkflowStageSpec(
            stage_id=self.stage_id,
            ref=self.ref,
            enabled_by_default=False,
            inputs=list(self.contract["inputs"]),
            outputs=list(self.contract["outputs"]),
            budget_policy_ref=budget_policy_ref,
            retry_policy={"max_attempts": 0, "retry_on": []},
        )

    async def execute(self, context: OptionalWorkflowStageContext) -> WorkflowStageResult:
        from praxist.core.storage import ArtifactWriter
        from praxist.core.trajectory import TrajectoryWriter

        trajectory = TrajectoryWriter(context.run_dir, context.run_id)
        if not context.enabled:
            trajectory.emit(
                "workflow.stage_skipped",
                scope={"stage_id": self.stage_id},
                actor={"type": "workflow_stage", "id": self.ref},
                payload={
                    "stage_id": self.stage_id,
                    "ref": self.ref,
                    "enabled": False,
                    "reason": context.reason,
                },
            )
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status="skipped",
                success=True,
                summary={"enabled": False, "reason": context.reason},
            )

        reviewer_local_modes = {
            "local",
            "artifact",
            "artifacts",
            "run_artifact",
            "claim_check",
            "review",
        }
        if self.stage_id == "reviewer" and context.mode in reviewer_local_modes:
            from praxist.plugins.workflow_stages.reviewer_stub.adapter import (
                run_local_artifact_review,
            )

            result = run_local_artifact_review(
                run_dir=context.run_dir,
                run_id=context.run_id,
                stage_ref=self.ref,
                source_event_ids=context.source_event_ids,
            )
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status=str(result["status"]),
                success=bool(result["success"]),
                output_artifacts=list(result.get("output_artifacts") or []),
                summary=dict(result.get("summary") or {}),
                error=result.get("error"),
            )

        started = trajectory.emit(
            "workflow.stage_started",
            scope={"stage_id": self.stage_id},
            actor={"type": "workflow_stage", "id": self.ref},
            payload={"stub": True, "mode": context.mode},
            parent_event_ids=context.source_event_ids,
        )
        if context.mode != "fake":
            error = f"{self.stage_id} optional workflow stage is enabled but no implementation is configured"
            trajectory.emit(
                "workflow.stage_failed",
                severity="error",
                scope={"stage_id": self.stage_id},
                actor={"type": "workflow_stage", "id": self.ref},
                payload={"error": error, "stub": True},
                parent_event_ids=[started["event_id"]],
            )
            return WorkflowStageResult(
                stage_id=self.stage_id,
                status="failed",
                success=False,
                error=error,
                summary={"enabled": True, "mode": context.mode, "stub": True},
            )

        artifacts = ArtifactWriter(context.run_dir, trajectory)
        artifact = artifacts.persist_json(
            str(self.contract["artifact_type"]),
            f"workflow/{self.stage_id}_stub.json",
            {
                "stage_id": self.stage_id,
                "ref": self.ref,
                "stub": True,
                "outputs": list(self.contract["outputs"]),
            },
            schema_ref=str(self.contract["schema_ref"]),
            producer={"stage_id": self.stage_id, "workflow_stage_ref": self.ref},
            source_event_ids=[started["event_id"]],
        )
        trajectory.emit(
            "workflow.stage_succeeded",
            scope={"stage_id": self.stage_id},
            actor={"type": "workflow_stage", "id": self.ref},
            payload={"stub": True, "artifact_id": artifact["artifact_id"]},
            artifact_refs=[artifact],
            parent_event_ids=[started["event_id"]],
        )
        return WorkflowStageResult(
            stage_id=self.stage_id,
            status="succeeded",
            success=True,
            output_artifacts=[artifact],
            summary={"enabled": True, "mode": "fake", "stub": True},
        )


def optional_workflow_stage(stage_id: str) -> OptionalWorkflowStageStub:
    """Create the disabled/enabled contract for a named optional workflow stage."""
    return OptionalWorkflowStageStub(stage_id)


def literature_scout_contract() -> dict[str, Any]:
    """Return the optional task-local literature_scout role/tool contract."""
    return dict(LITERATURE_SCOUT_CONTRACT)


def emit_disabled_optional_events(
    trajectory: Any,
    *,
    stages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    research_stage_id: str = "research_loop",
) -> None:
    """Record disabled optional stages and tools into trajectory for replay visibility."""
    stage_records = stages if stages is not None else disabled_optional_stages()
    tool_records = tools if tools is not None else disabled_optional_tools()
    for disabled in stage_records:
        trajectory.emit(
            "workflow.stage_skipped",
            scope={"stage_id": str(disabled["stage_id"])},
            actor={"type": "workflow_stage", "id": str(disabled["ref"])},
            payload={**disabled, "enabled": False},
        )
    for disabled in tool_records:
        trajectory.emit(
            "workflow.optional_disabled",
            scope={
                "stage_id": str(disabled.get("stage_id") or research_stage_id),
                "role_ref": str(disabled["role_ref"]),
            },
            actor={"type": "workflow_stage", "id": research_stage_id},
            payload={**disabled, "enabled": False},
        )
