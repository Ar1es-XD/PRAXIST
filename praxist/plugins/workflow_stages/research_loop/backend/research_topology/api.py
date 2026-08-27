"""Module-facing API façade for research-loop state and recommendations."""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import Any

from praxist.plugins.workflow_stages.research_loop.backend.artifact_semantics import (
    is_committed_runtime_fact_source,
)

from .schema import ResearchCommand, TopologyChangeRequest

logger = logging.getLogger(__name__)


class ResearchLoopModuleAPI:
    """Stable Python façade for external orchestration modules.

    The API writes structured requests into a dedicated queue under the run
    directory instead of asking callers to edit prompts, agenda files, or
    generation internals.  Current research-loop code treats these requests as
    queued operator intent; later topology executors can consume the same queue
    without changing the external contract.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.requests_dir = self.run_dir / "external_requests"
        self.commands_path = self.requests_dir / "research_commands.jsonl"
        self.artifact_warnings: list[dict[str, str]] = []

    def submit_recommendation(
        self,
        *,
        recommendation: str,
        scope: str = "next_generation",
        reason: str = "",
        submitted_by: str = "external_module",
        metadata: dict[str, Any] | None = None,
    ) -> ResearchCommand:
        command = ResearchCommand.create(
            command_type="recommendation",
            scope=scope,
            reason=reason,
            submitted_by=submitted_by,
            payload={
                "recommendation": recommendation,
                "metadata": dict(metadata or {}),
            },
        )
        self.append_command(command)
        return command

    def request_topology_change(
        self,
        *,
        requested_changes: list[dict[str, Any]],
        reason: str,
        scope: str = "next_generation",
        safety_constraints: list[str] | None = None,
        submitted_by: str = "external_module",
    ) -> ResearchCommand:
        request = TopologyChangeRequest.create(
            requested_changes=requested_changes,
            reason=reason,
            scope=scope,
            safety_constraints=safety_constraints,
        )
        command = request.to_command(submitted_by=submitted_by)
        self.append_command(command)
        return command

    def append_command(self, command: ResearchCommand) -> Path:
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        payload = command.to_dict()
        payload["status"] = "queued"
        with open(self.commands_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
        return self.commands_path

    def list_commands(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.commands_path.exists():
            return []
        records: list[dict[str, Any]] | deque[dict[str, Any]]
        records = deque(maxlen=limit) if limit and limit > 0 else []
        for line_number, line in enumerate(
            self.commands_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "research command queue: skipping malformed line %d of %s: %s",
                    line_number,
                    self.commands_path,
                    exc,
                )
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
        return list(records)

    def get_artifact_warnings(self) -> list[dict[str, str]]:
        """Return non-fatal artifact read warnings seen by this API instance."""

        return list(self.artifact_warnings)

    def list_findings(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        findings_dir = self.run_dir / "findings"
        shared_findings_dir = self.run_dir / "shared_findings"
        for filename in ("findings.jsonl", "findings.json"):
            records.extend(self._read_json_records(findings_dir / filename))
        records.extend(self._read_json_records(shared_findings_dir))
        return self._dedupe_records(records)

    def get_frontier_summary(self) -> list[dict[str, Any]]:
        frontier_dir = self.run_dir / "frontier"
        manifest_path = frontier_dir / "frontier_manifest.json"
        if manifest_path.exists():
            data = self._read_json_value(manifest_path)
            if isinstance(data, dict):
                if not is_committed_runtime_fact_source(data, legacy_ok=True):
                    self._record_artifact_warning(
                        path=manifest_path,
                        warning_type="non_runtime_fact_source",
                        message="frontier manifest is not committed runtime state",
                    )
                    return []
                records: list[dict[str, Any]] = []
                records.extend(self._dicts_from_value(data.get("cumulative_top")))
                lane_frontiers = data.get("lane_frontiers")
                if isinstance(lane_frontiers, dict):
                    for lane_name, lane_records in lane_frontiers.items():
                        for record in self._dicts_from_value(lane_records):
                            item = dict(record)
                            item.setdefault("frontier_lane", lane_name)
                            records.append(item)
                generations = data.get("generations")
                if isinstance(generations, dict):
                    for generation_id, generation_records in generations.items():
                        for record in self._dicts_from_value(generation_records):
                            item = dict(record)
                            item.setdefault("generation_id", generation_id)
                            records.append(item)
                if records:
                    return self._dedupe_frontier_records(records)

        summary_path = frontier_dir / "frontier.json"
        if summary_path.exists():
            data = self._read_json_value(summary_path)
            if isinstance(data, list):
                return self._dedupe_frontier_records(
                    [item for item in data if isinstance(item, dict)]
                )
            if isinstance(data, dict):
                raw = data.get("frontier") or data.get("items") or data.get("records")
                if isinstance(raw, list):
                    return self._dedupe_frontier_records(
                        [item for item in raw if isinstance(item, dict)]
                    )
        materialized = self._read_json_records(self.run_dir / "findings" / "frontier.jsonl")
        if materialized:
            return self._dedupe_frontier_records(materialized)
        return []

    def get_validation_signals(self, *, current_gen_id: int | None = None) -> list[dict[str, Any]]:
        """Return compact non-durable validation signals for planning modules.

        This is intentionally separate from ``get_frontier_summary()`` so
        modules cannot accidentally treat partial/derived/failed artifacts as
        durable frontier facts. The returned entries are suitable only for
        validation, repair, falsification, comparison, ablation, or mature
        follow-up scoring.
        """

        try:
            from praxist.plugins.workflow_stages.research_loop.backend.research_memory.evidence_pack_builder import (
                _digest_validation_candidates,
            )

            return _digest_validation_candidates(
                self.run_dir,
                current_gen_id=current_gen_id,
            )
        except Exception as exc:  # noqa: BLE001 - module API should be best-effort.
            self._record_artifact_warning(
                path=self.run_dir / "frontier" / "frontier_manifest.json",
                warning_type="validation_signal_read_failed",
                message=f"validation signals could not be read: {type(exc).__name__}",
            )
            return []

    def get_gems_summary(self) -> dict[str, Any]:
        gems_dir = self.run_dir / "gems"
        candidates = [
            gems_dir / "gems_state.json",
            gems_dir / "gems.json",
            gems_dir / "active_gems.json",
            self.run_dir / "gems.json",
        ]
        for path in candidates:
            if path.exists():
                data = self._read_json_value(path)
                if data is None:
                    continue
                if (
                    path.name == "gems_state.json"
                    and isinstance(
                        data,
                        dict,
                    )
                    and not is_committed_runtime_fact_source(data, legacy_ok=True)
                ):
                    self._record_artifact_warning(
                        path=path,
                        warning_type="non_runtime_fact_source",
                        message="Gems state is not committed runtime state",
                    )
                    return {}
                return data if isinstance(data, dict) else {"records": data}
        return {}

    def get_memory_summary(self) -> dict[str, Any]:
        memory_dir = self.run_dir / "research_memory"
        candidates = [
            memory_dir / "summary.json",
            memory_dir / "research_memory_summary.json",
            self.run_dir / "research_memory_summary.json",
        ]
        for path in candidates:
            if path.exists():
                data = self._read_json_value(path)
                if data is None:
                    continue
                return data if isinstance(data, dict) else {"records": data}
        canonical_records = self._read_json_records(
            self.run_dir / "memory" / "research_memory.jsonl"
        )
        if canonical_records:
            return {"records": canonical_records}
        return {}

    def get_run_status(self) -> dict[str, Any]:
        candidates = [
            self.run_dir / "orchestrator_status.final.json",
            self.run_dir / "orchestrator_status.json",
            self.run_dir / "run_summary.json",
            self.run_dir / "run.json",
        ]
        for path in candidates:
            if path.exists():
                data = self._read_json_value(path)
                if isinstance(data, dict):
                    return data
        return {"run_dir": str(self.run_dir), "status": "unknown"}

    def _record_artifact_warning(self, *, path: Path, warning_type: str, message: str) -> None:
        self.artifact_warnings.append(
            {
                "path": str(path),
                "warning_type": warning_type,
                "message": message,
            }
        )

    def _read_json_value(self, path: Path) -> Any | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("skipping malformed JSON file %s: %s", path, exc)
            self._record_artifact_warning(
                path=path,
                warning_type="json_decode_error",
                message=str(exc),
            )
            return None
        except OSError as exc:
            logger.warning("could not read JSON file %s: %s", path, exc)
            self._record_artifact_warning(
                path=path,
                warning_type="read_error",
                message=str(exc),
            )
            return None

    @staticmethod
    def _dicts_from_value(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            for key in ("members", "entries", "items", "records", "frontier"):
                nested = value.get(key)
                if isinstance(nested, list):
                    return [item for item in nested if isinstance(item, dict)]
            return [value]
        return []

    @classmethod
    def _dedupe_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            key = cls._record_identity(record)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @classmethod
    def _dedupe_frontier_records(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate frontier views by candidate entity before evidence id.

        Frontier manifests deliberately expose the same promoted candidate via
        cumulative, lane, and generation sections.  Different evidence ids for
        the same entity should not amplify that candidate in downstream module
        APIs or DIG context.
        """

        try:
            from praxist.plugins.workflow_stages.research_loop.backend.frontier import (
                _candidate_entity_key,
            )
        except Exception:  # pragma: no cover - import fallback for partial installs
            _candidate_entity_key = None  # type: ignore[assignment]

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                continue
            key = ""
            if _candidate_entity_key is not None:
                try:
                    entity_key = _candidate_entity_key(record)
                except Exception:
                    entity_key = ""
                if entity_key and not entity_key.startswith("object::"):
                    key = f"entity:{entity_key}"
            if not key:
                key = cls._record_identity(record)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    @staticmethod
    def _normalized_frontier_entity_key(entity_key: str) -> str:
        """Normalize common run-local variant wrappers for module API views."""

        prefix = ""
        payload = entity_key
        if "::" in entity_key:
            prefix, _, payload = entity_key.partition("::")
        payload = re.sub(r"^gen\d+_peer\d+_", "", payload)
        payload = re.sub(r"_t\d+$", "", payload)
        return f"{prefix}::{payload}" if prefix else payload

    @staticmethod
    def _record_identity(record: dict[str, Any]) -> str:
        for key in ("finding_id", "id", "record_id"):
            value = record.get(key)
            if value not in (None, ""):
                return f"id:{value}"
        for key in ("variant_name", "variant_id", "frontier_entity_key"):
            value = record.get(key)
            if value not in (None, ""):
                return f"{key}:{value}"
        try:
            return "json:" + json.dumps(record, sort_keys=True, default=str)
        except TypeError:
            return "repr:" + repr(sorted(record.items()))

    def _read_json_records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        if path.is_file():
            paths = [path]
        else:
            paths = sorted(
                child
                for child in path.iterdir()
                if child.is_file()
                and child.suffix.lower() in {".json", ".jsonl"}
                and child.name not in {"frontier.json", "frontier.jsonl"}
            )
        for item in paths:
            if item.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(
                    item.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "skipping malformed JSONL line %d of %s: %s",
                            line_number,
                            item,
                            exc,
                        )
                        self._record_artifact_warning(
                            path=item,
                            warning_type="jsonl_decode_error",
                            message=f"line {line_number}: {exc}",
                        )
                        continue
                    if isinstance(parsed, dict):
                        records.append(parsed)
            else:
                parsed = self._read_json_value(item)
                if isinstance(parsed, dict):
                    records.append(parsed)
                elif isinstance(parsed, list):
                    records.extend(entry for entry in parsed if isinstance(entry, dict))
        return records
