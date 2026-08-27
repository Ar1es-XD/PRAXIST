"""Lightweight schema helpers for DIG-Lite artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

RiskLevel = str
MechanismFamily = str
Intent = str

VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_INTENTS = {
    "exploit",
    "repair",
    "bridge",
    "ablate",
    "falsify",
    "explore",
    "anti_mainline",
    "diagnose",
    "control",
    "audit",
}


class DIGSchemaError(ValueError):
    """Raised when a DIG artifact cannot be parsed into its schema."""


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DIGSchemaError(f"{label} must be a mapping")
    return value


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        return str(value)
    except (TypeError, ValueError):
        return default


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in raw:
        text = _str(item).strip()
        if text:
            out.append(text)
    return out


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _risk(value: Any, default: str = "medium") -> str:
    text = _str(value, default).strip().lower()
    return text if text in VALID_RISK_LEVELS else default


def _mechanism_family(value: Any) -> str:
    text = _str(value, "other").strip()
    return text or "other"


def _intent(value: Any) -> str:
    text = _str(value, "explore").strip()
    return text if text in VALID_INTENTS else "explore"


@dataclass
class TaskObjective:
    """Task-level objective summary extracted during DIG baseline mapping."""

    primary_metric: str = ""
    secondary_metrics: list[str] = field(default_factory=list)
    failure_constraints: list[str] = field(default_factory=list)
    forbidden_shortcuts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> TaskObjective:
        data = _require_dict(data, "task_objective")
        return cls(
            primary_metric=_str(data.get("primary_metric")),
            secondary_metrics=_str_list(data.get("secondary_metrics")),
            failure_constraints=_str_list(data.get("failure_constraints")),
            forbidden_shortcuts=_str_list(data.get("forbidden_shortcuts")),
        )


@dataclass
class BaselineCorePathItem:
    """One baseline file or component relevant to the candidate design space."""

    file: str = ""
    role: str = ""
    key_functions_or_classes: list[str] = field(default_factory=list)
    reason_it_matters: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> BaselineCorePathItem:
        data = _require_dict(data, "baseline_core_path item")
        return cls(
            file=_str(data.get("file")),
            role=_str(data.get("role")),
            key_functions_or_classes=_str_list(data.get("key_functions_or_classes")),
            reason_it_matters=_str(data.get("reason_it_matters")),
        )


@dataclass
class InterventionSurface:
    """Allowed or forbidden surface that a DIG candidate may reference."""

    name: str = ""
    files: list[str] = field(default_factory=list)
    allowed: bool = True
    expected_impact: str = ""
    risk_level: RiskLevel = "medium"
    reason: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> InterventionSurface:
        data = _require_dict(data, "intervention surface")
        return cls(
            name=_str(data.get("name")),
            files=_str_list(data.get("files")),
            allowed=bool(data.get("allowed", True)),
            expected_impact=_str(data.get("expected_impact")),
            risk_level=_risk(data.get("risk_level")),
            reason=_str(data.get("reason")),
        )


@dataclass
class BaselineMechanismMap:
    """Structured DIG artifact describing the baseline mechanism and risks."""

    task_objective: TaskObjective
    baseline_core_path: list[BaselineCorePathItem] = field(default_factory=list)
    data_and_training_flow: dict[str, Any] = field(default_factory=dict)
    task_execution_flow: dict[str, Any] = field(default_factory=dict)
    intervention_surfaces: list[InterventionSurface] = field(default_factory=list)
    forbidden_surfaces: list[InterventionSurface] = field(default_factory=list)
    likely_bottlenecks: list[dict[str, Any]] = field(default_factory=list)
    implementation_risks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> BaselineMechanismMap:
        data = _require_dict(data, "BaselineMechanismMap")
        task_execution_flow = dict(
            data.get("task_execution_flow") or data.get("data_and_training_flow") or {}
        )
        return cls(
            task_objective=TaskObjective.from_dict(data.get("task_objective") or {}),
            baseline_core_path=[
                BaselineCorePathItem.from_dict(item)
                for item in data.get("baseline_core_path", []) or []
                if isinstance(item, dict)
            ],
            data_and_training_flow=task_execution_flow,
            task_execution_flow=task_execution_flow,
            intervention_surfaces=[
                InterventionSurface.from_dict(item)
                for item in data.get("intervention_surfaces", []) or []
                if isinstance(item, dict)
            ],
            forbidden_surfaces=[
                InterventionSurface.from_dict(item)
                for item in data.get("forbidden_surfaces", []) or []
                if isinstance(item, dict)
            ],
            likely_bottlenecks=_dict_list(data.get("likely_bottlenecks")),
            implementation_risks=_dict_list(data.get("implementation_risks")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImplementationSketch:
    """Candidate-level sketch of files and changes before code is written."""

    files_to_modify: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> ImplementationSketch:
        data = _require_dict(data, "implementation_sketch")
        return cls(
            files_to_modify=_str_list(data.get("files_to_modify")),
            changes=_str_list(data.get("changes")),
        )


@dataclass
class DiagnosticPrediction:
    """Predicted metric and internal diagnostic movement for a DIG candidate."""

    primary_metric: str = ""
    secondary_or_safety_metric: str = ""
    internal_signal: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> DiagnosticPrediction:
        data = _require_dict(data, "diagnostic_prediction")
        return cls(
            primary_metric=_str(data.get("primary_metric")),
            secondary_or_safety_metric=_str(data.get("secondary_or_safety_metric")),
            internal_signal=_str(data.get("internal_signal")),
        )


@dataclass
class CandidateRisk:
    """Implementation, gaming, bug, and compute risk labels for a candidate."""

    implementation: RiskLevel = "medium"
    metric_gaming: RiskLevel = "medium"
    silent_bug: RiskLevel = "medium"
    compute: RiskLevel = "medium"

    @classmethod
    def from_dict(cls, data: Any) -> CandidateRisk:
        data = _require_dict(data, "risk")
        return cls(
            implementation=_risk(data.get("implementation")),
            metric_gaming=_risk(data.get("metric_gaming")),
            silent_bug=_risk(data.get("silent_bug")),
            compute=_risk(data.get("compute")),
        )


@dataclass(frozen=True)
class DiversitySignature:
    """Quality-diversity coordinates for a DIG candidate."""

    mechanism_family: str = ""
    intervention_surface: str = ""
    intent: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> DiversitySignature:
        data = _require_dict(data, "diversity_signature")
        return cls(
            mechanism_family=_str(data.get("mechanism_family")),
            intervention_surface=_str(data.get("intervention_surface")),
            intent=_str(data.get("intent")),
        )

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.mechanism_family, self.intervention_surface, self.intent)


@dataclass
class Candidate:
    """One mechanism-level candidate design produced during DIG."""

    candidate_id: str
    name: str
    mechanism_family: MechanismFamily
    intervention_surface: str
    intent: Intent
    hypothesis: str
    expected_gain_path: str
    implementation_sketch: ImplementationSketch
    diagnostic_prediction: DiagnosticPrediction
    risk: CandidateRisk
    ablation_hooks: list[str] = field(default_factory=list)
    diversity_signature: DiversitySignature = field(default_factory=DiversitySignature)
    semantic_family: str = ""
    parent_lineage: str = ""
    novelty_axis: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> Candidate:
        data = _require_dict(data, "Candidate")
        candidate_id = _str(data.get("candidate_id")).strip()
        if not candidate_id:
            raise DIGSchemaError("candidate_id is required")
        mechanism_family = _mechanism_family(data.get("mechanism_family"))
        intent = _intent(data.get("intent"))
        diversity_data = data.get("diversity_signature") or {
            "mechanism_family": mechanism_family,
            "intervention_surface": _str(data.get("intervention_surface")),
            "intent": intent,
        }
        return cls(
            candidate_id=candidate_id,
            name=_str(data.get("name")),
            mechanism_family=mechanism_family,
            intervention_surface=_str(data.get("intervention_surface")),
            intent=intent,
            hypothesis=_str(data.get("hypothesis")),
            expected_gain_path=_str(data.get("expected_gain_path")),
            semantic_family=_str(data.get("semantic_family")),
            parent_lineage=_str(data.get("parent_lineage")),
            novelty_axis=_str(data.get("novelty_axis")),
            implementation_sketch=ImplementationSketch.from_dict(
                data.get("implementation_sketch") or {}
            ),
            diagnostic_prediction=DiagnosticPrediction.from_dict(
                data.get("diagnostic_prediction") or {}
            ),
            risk=CandidateRisk.from_dict(data.get("risk") or {}),
            ablation_hooks=_str_list(data.get("ablation_hooks")),
            diversity_signature=DiversitySignature.from_dict(diversity_data),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePool:
    """Collection of candidate designs generated before critique."""

    candidates: list[Candidate] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> CandidatePool:
        data = _require_dict(data, "CandidatePool")
        return cls(
            candidates=[
                Candidate.from_dict(item)
                for item in data.get("candidates", []) or []
                if isinstance(item, dict)
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [candidate.to_dict() for candidate in self.candidates]}


@dataclass
class CandidateScores:
    """Compact one-to-five critique scores for a DIG candidate."""

    mechanism_plausibility: int = 1
    implementability: int = 1
    diagnostic_clarity: int = 1
    diversity_value: int = 1
    shortcut_risk: int = 1
    silent_bug_risk: int = 1
    compute_risk: int = 1

    @staticmethod
    def _score(value: Any) -> int:
        try:
            score = int(value)
        except (TypeError, ValueError):
            score = 1
        if score < 1 or score > 5:
            raise DIGSchemaError("candidate review scores must be in [1, 5]")
        return score

    @classmethod
    def from_dict(cls, data: Any) -> CandidateScores:
        data = _require_dict(data, "scores")
        return cls(
            mechanism_plausibility=cls._score(data.get("mechanism_plausibility")),
            implementability=cls._score(data.get("implementability")),
            diagnostic_clarity=cls._score(data.get("diagnostic_clarity")),
            diversity_value=cls._score(data.get("diversity_value")),
            shortcut_risk=cls._score(data.get("shortcut_risk")),
            silent_bug_risk=cls._score(data.get("silent_bug_risk")),
            compute_risk=cls._score(data.get("compute_risk")),
        )


@dataclass
class CandidateReview:
    """Structured critique record for one candidate."""

    candidate_id: str
    scores: CandidateScores
    fatal_flaws: list[str] = field(default_factory=list)
    repair_suggestion: str = ""
    reasoned_summary: str = ""

    @property
    def quality_score(self) -> int:
        return (
            self.scores.mechanism_plausibility
            + self.scores.implementability
            + self.scores.diagnostic_clarity
            + self.scores.diversity_value
            - self.scores.shortcut_risk
            - self.scores.silent_bug_risk
            - self.scores.compute_risk
        )

    @classmethod
    def from_dict(cls, data: Any) -> CandidateReview:
        data = _require_dict(data, "CandidateReview")
        candidate_id = _str(data.get("candidate_id")).strip()
        if not candidate_id:
            raise DIGSchemaError("review candidate_id is required")
        return cls(
            candidate_id=candidate_id,
            scores=CandidateScores.from_dict(data.get("scores") or {}),
            fatal_flaws=_str_list(data.get("fatal_flaws")),
            repair_suggestion=_str(data.get("repair_suggestion")),
            reasoned_summary=_str(data.get("reasoned_summary")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quality_score"] = self.quality_score
        return data


@dataclass
class CandidateReviews:
    """Collection of candidate reviews keyed by candidate id."""

    reviews: list[CandidateReview] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> CandidateReviews:
        data = _require_dict(data, "CandidateReviews")
        return cls(
            reviews=[
                CandidateReview.from_dict(item)
                for item in data.get("reviews", []) or []
                if isinstance(item, dict)
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {"reviews": [review.to_dict() for review in self.reviews]}


@dataclass(frozen=True)
class DiversityCell:
    """Selected contract quality-diversity cell."""

    mechanism_family: str = ""
    intervention_surface: str = ""
    intent: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> DiversityCell:
        data = _require_dict(data, "diversity_cell")
        return cls(
            mechanism_family=_str(data.get("mechanism_family")),
            intervention_surface=_str(data.get("intervention_surface")),
            intent=_str(data.get("intent")),
        )

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.mechanism_family, self.intervention_surface, self.intent)


@dataclass
class ContractStep:
    """One implementation-plan step in the selected DIG contract."""

    step: int
    action: str

    @classmethod
    def from_dict(cls, data: Any) -> ContractStep:
        data = _require_dict(data, "implementation step")
        try:
            step = int(data.get("step", 0))
        except (TypeError, ValueError):
            step = 0
        return cls(step=step, action=_str(data.get("action")))


@dataclass
class ExpectedMetricSignature:
    """Predicted primary, safety, and diagnostic signals for a contract."""

    primary: str = ""
    secondary_or_safety: str = ""
    diagnostic: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> ExpectedMetricSignature:
        data = _require_dict(data, "expected_metric_signature")
        return cls(
            primary=_str(data.get("primary")),
            secondary_or_safety=_str(data.get("secondary_or_safety")),
            diagnostic=_str(data.get("diagnostic")),
        )


@dataclass
class ContractAmendmentPolicy:
    """Rules for recording meaningful implementation deviations."""

    allowed_reasons: list[str] = field(default_factory=list)
    required_artifact: str = "contract_amendment.yaml"

    @classmethod
    def from_dict(cls, data: Any) -> ContractAmendmentPolicy:
        data = _require_dict(data, "contract_amendment_policy")
        return cls(
            allowed_reasons=_str_list(data.get("allowed_reasons")),
            required_artifact=_str(data.get("required_artifact"), "contract_amendment.yaml"),
        )


@dataclass
class SelectedContract:
    """Locked DIG implementation contract injected into the peer prompt."""

    selected_candidate_id: str
    variant_name: str
    diversity_cell: DiversityCell
    mechanism_hypothesis: str
    why_selected: str
    rejected_alternatives: list[dict[str, Any]]
    files_to_modify: list[str]
    allowed_changes: list[str]
    forbidden_changes: list[str]
    implementation_plan: list[ContractStep]
    expected_metric_signature: ExpectedMetricSignature
    ablation_hooks: list[str]
    fail_fast_checks: list[str]
    contract_amendment_policy: ContractAmendmentPolicy
    semantic_family: str = ""
    parent_lineage: str = ""
    novelty_axis: str = ""
    canonical_labels: dict[str, Any] = field(default_factory=dict)
    dig_provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> SelectedContract:
        data = _require_dict(data, "SelectedContract")
        return cls(
            selected_candidate_id=_str(data.get("selected_candidate_id")),
            variant_name=_str(data.get("variant_name")),
            diversity_cell=DiversityCell.from_dict(data.get("diversity_cell") or {}),
            semantic_family=_str(data.get("semantic_family")),
            parent_lineage=_str(data.get("parent_lineage")),
            novelty_axis=_str(data.get("novelty_axis")),
            mechanism_hypothesis=_str(data.get("mechanism_hypothesis")),
            why_selected=_str(data.get("why_selected")),
            rejected_alternatives=_dict_list(data.get("rejected_alternatives")),
            files_to_modify=_str_list(data.get("files_to_modify")),
            allowed_changes=_str_list(data.get("allowed_changes")),
            forbidden_changes=_str_list(data.get("forbidden_changes")),
            implementation_plan=[
                ContractStep.from_dict(item)
                for item in data.get("implementation_plan", []) or []
                if isinstance(item, dict)
            ],
            expected_metric_signature=ExpectedMetricSignature.from_dict(
                data.get("expected_metric_signature") or {}
            ),
            ablation_hooks=_str_list(data.get("ablation_hooks")),
            fail_fast_checks=_str_list(data.get("fail_fast_checks")),
            contract_amendment_policy=ContractAmendmentPolicy.from_dict(
                data.get("contract_amendment_policy")
                or {
                    "allowed_reasons": [
                        "baseline assumption was wrong",
                        "shape or API mismatch makes original implementation impossible",
                        "contract would require touching a forbidden path",
                    ],
                    "required_artifact": "contract_amendment.yaml",
                }
            ),
            canonical_labels=dict(data.get("canonical_labels") or {}),
            dig_provenance=dict(data.get("dig_provenance") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
