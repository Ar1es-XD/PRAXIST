"""Typed research-topology contracts for the research-loop plugin.

These contracts intentionally live inside the ``workflow_stage:research_loop``
plugin.  They are generic within that plugin, but they are not core contracts
yet: the current production path is still the legacy generation/cohort loop,
wrapped by a compatibility topology executor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast
from uuid import uuid4

JSONDict = dict[str, Any]

TopologyScope = Literal[
    "current_generation_after_boundary",
    "next_generation",
    "future_run",
]

CommandType = Literal[
    "recommendation",
    "external_review",
    "ablation_request",
    "repair_request",
    "topology_change_request",
    "finding_injection",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass(frozen=True)
class WorkerCapabilitySet:
    """Declared permissions and output responsibilities for one worker type."""

    can_write_code: bool = False
    can_run_training: bool = False
    can_run_eval: bool = False
    can_modify_data_pipeline: bool = False
    can_submit_external_result: bool = False
    can_use_web: bool = False
    can_write_findings: bool = True
    can_emit_recommendations: bool = True

    def to_dict(self) -> JSONDict:
        return asdict(self)


DEFAULT_WORKER_CAPABILITIES: dict[str, WorkerCapabilitySet] = {
    "experiment_peer": WorkerCapabilitySet(
        can_write_code=True,
        can_run_training=True,
        can_run_eval=True,
    ),
    "repair_peer": WorkerCapabilitySet(
        can_write_code=True,
        can_run_training=True,
        can_run_eval=True,
    ),
    "falsifier_peer": WorkerCapabilitySet(
        can_write_code=True,
        can_run_training=True,
        can_run_eval=True,
    ),
    "replication_peer": WorkerCapabilitySet(
        can_write_code=True,
        can_run_training=True,
        can_run_eval=True,
    ),
    "evaluator_peer": WorkerCapabilitySet(can_run_eval=True),
    "data_engineering_peer": WorkerCapabilitySet(
        can_write_code=True,
        can_modify_data_pipeline=True,
    ),
    "literature_scout_peer": WorkerCapabilitySet(can_use_web=True),
    "writing_feedback_peer": WorkerCapabilitySet(can_emit_recommendations=True),
    "external_validity_peer": WorkerCapabilitySet(can_run_eval=True),
    "deployment_peer": WorkerCapabilitySet(
        can_write_code=True,
        can_run_eval=True,
        can_submit_external_result=True,
    ),
    "panel_role": WorkerCapabilitySet(can_write_findings=False, can_emit_recommendations=True),
    "synthesizer": WorkerCapabilitySet(can_write_findings=False, can_emit_recommendations=True),
    "external_input": WorkerCapabilitySet(can_write_findings=False, can_emit_recommendations=True),
}


@dataclass(frozen=True)
class GenerationTopologyContext:
    """Explicit context passed from GenerationLoop into topology adapters."""

    run_dir: str
    task_id: str
    generation_id: int
    cohort_size: int
    panel_topology_ref: str = ""
    metadata: JSONDict = field(default_factory=dict)
    peer_role_ref: str | None = None
    peer_role_refs: tuple[str | None, ...] = ()

    @classmethod
    def from_loop(cls, loop: Any, gen_id: int) -> GenerationTopologyContext:
        gp = loop.task_spec.generation_policy
        cohort_size = int(getattr(gp, "cohort_size", 0) or 0)
        default_role_ref = getattr(loop, "peer_role_ref", None)
        role_resolver = getattr(loop, "peer_role_ref_for", None)
        peer_role_refs: tuple[str | None, ...] = ()
        if callable(role_resolver):
            peer_role_refs = tuple(
                cast(str | None, role_resolver(gen_id, peer_index))
                for peer_index in range(cohort_size)
            )
        return cls(
            run_dir=str(getattr(loop, "run_dir", "")),
            task_id=str(getattr(loop.task_spec, "task_id", "")),
            generation_id=int(gen_id),
            cohort_size=cohort_size,
            panel_topology_ref=str(getattr(loop, "_panel_topology_ref", "")),
            metadata={"source": "GenerationLoop"},
            peer_role_ref=str(default_role_ref) if default_role_ref else None,
            peer_role_refs=peer_role_refs,
        )

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(frozen=True)
class WorkerSpec:
    """Declarative worker node in a research topology."""

    worker_id: str
    worker_type: str
    role_ref: str | None = None
    capabilities: WorkerCapabilitySet | None = None
    required_inputs: list[str] = field(default_factory=list)
    declared_outputs: list[str] = field(default_factory=list)
    metadata: JSONDict = field(default_factory=dict)

    def normalized(self) -> WorkerSpec:
        capabilities = self.capabilities or DEFAULT_WORKER_CAPABILITIES.get(
            self.worker_type,
            WorkerCapabilitySet(),
        )
        return WorkerSpec(
            worker_id=str(self.worker_id),
            worker_type=str(self.worker_type),
            role_ref=self.role_ref,
            capabilities=capabilities,
            required_inputs=list(self.required_inputs),
            declared_outputs=list(self.declared_outputs),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> JSONDict:
        normalized = self.normalized()
        data = asdict(normalized)
        data["capabilities"] = normalized.capabilities.to_dict() if normalized.capabilities else {}
        return data


@dataclass(frozen=True)
class TopologyEdge:
    """Directed artifact or event dependency between topology nodes."""

    source: str
    target: str
    relation: str = "artifact_dependency"
    metadata: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(frozen=True)
class TopologyPolicy:
    """Scheduling, visibility, and admission policy for a topology."""

    scheduling: str = "generation_cohort"
    retry: JSONDict = field(default_factory=dict)
    memory_visibility: str = "bounded_task_scoped"
    artifact_visibility: str = "task_scoped"
    promotion_policy: str = "task_defined"
    external_command_policy: str = "queue_for_boundary"
    metadata: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(frozen=True)
class ResearchTopologySpec:
    """Topology-level contract for one research-loop execution segment."""

    topology_id: str
    generation_id: int | None = None
    nodes: list[WorkerSpec] = field(default_factory=list)
    edges: list[TopologyEdge] = field(default_factory=list)
    policy: TopologyPolicy = field(default_factory=TopologyPolicy)
    metadata: JSONDict = field(default_factory=dict)

    def validate(self) -> None:
        worker_ids = [node.worker_id for node in self.nodes]
        if len(worker_ids) != len(set(worker_ids)):
            raise ValueError("ResearchTopologySpec contains duplicate worker_id values")
        known = set(worker_ids)
        for edge in self.edges:
            if edge.source not in known:
                raise ValueError(f"TopologyEdge references unknown source: {edge.source}")
            if edge.target not in known:
                raise ValueError(f"TopologyEdge references unknown target: {edge.target}")

    def to_dict(self) -> JSONDict:
        self.validate()
        return {
            "topology_id": self.topology_id,
            "generation_id": self.generation_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "policy": self.policy.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class WorkUnit:
    """Concrete unit passed from a topology executor to a worker adapter."""

    work_unit_id: str
    worker: WorkerSpec
    generation_id: int
    cohort_index: int | None = None
    phase: str = "implementation"
    payload: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "work_unit_id": self.work_unit_id,
            "worker": self.worker.to_dict(),
            "generation_id": self.generation_id,
            "cohort_index": self.cohort_index,
            "phase": self.phase,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class ResearchEvent:
    """External or internal event that can drive topology execution."""

    event_id: str
    event_type: str
    scope: str
    payload: JSONDict = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    @classmethod
    def create(
        cls, *, event_type: str, scope: str, payload: JSONDict | None = None
    ) -> ResearchEvent:
        return cls(
            event_id=_new_id("evt"),
            event_type=event_type,
            scope=scope,
            payload=dict(payload or {}),
        )

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(frozen=True)
class ResearchCommand:
    """Structured command submitted by an operator or external module."""

    command_id: str
    command_type: CommandType | str
    scope: TopologyScope | str
    payload: JSONDict = field(default_factory=dict)
    reason: str = ""
    submitted_by: str = "operator"
    created_at: str = field(default_factory=_utc_now)
    status: str = "queued"

    @classmethod
    def create(
        cls,
        *,
        command_type: CommandType | str,
        scope: TopologyScope | str = "next_generation",
        payload: JSONDict | None = None,
        reason: str = "",
        submitted_by: str = "operator",
    ) -> ResearchCommand:
        return cls(
            command_id=_new_id("cmd"),
            command_type=command_type,
            scope=scope,
            payload=dict(payload or {}),
            reason=reason,
            submitted_by=submitted_by,
        )

    def to_dict(self) -> JSONDict:
        return asdict(self)


@dataclass(frozen=True)
class TopologyChangeRequest:
    """Request to alter future research topology without mutating loop internals."""

    request_id: str
    scope: TopologyScope | str
    requested_changes: list[JSONDict]
    reason: str
    safety_constraints: list[str] = field(default_factory=list)
    accepted_by: str | None = None
    status: str = "pending"
    created_at: str = field(default_factory=_utc_now)

    @classmethod
    def create(
        cls,
        *,
        requested_changes: list[JSONDict],
        reason: str,
        scope: TopologyScope | str = "next_generation",
        safety_constraints: list[str] | None = None,
    ) -> TopologyChangeRequest:
        return cls(
            request_id=_new_id("topo"),
            scope=scope,
            requested_changes=list(requested_changes),
            reason=reason,
            safety_constraints=list(safety_constraints or []),
        )

    def to_command(self, *, submitted_by: str = "operator") -> ResearchCommand:
        return ResearchCommand.create(
            command_type="topology_change_request",
            scope=self.scope,
            payload=self.to_dict(),
            reason=self.reason,
            submitted_by=submitted_by,
        )

    def to_dict(self) -> JSONDict:
        return asdict(self)
