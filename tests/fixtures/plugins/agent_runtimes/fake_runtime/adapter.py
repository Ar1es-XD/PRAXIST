"""Executable entrypoint for the bundled deterministic AgentRuntime."""

from __future__ import annotations

from typing import Any

from praxist.core.protocol import AgentEvent, AgentRunRequest, AgentRunResult


class FakeRuntime:
    runtime_ref = "agent_runtime:fake_runtime"

    def execute_sync(self, request: AgentRunRequest) -> AgentRunResult:
        return _TranscriptRuntime(self.runtime_ref, "fake").execute_sync(request)


class _TranscriptRuntime:
    def __init__(self, runtime_ref: str, transcript_kind: str) -> None:
        self.runtime_ref = runtime_ref
        self.transcript_kind = transcript_kind

    def execute_sync(self, request: AgentRunRequest) -> AgentRunResult:
        agent_run_id = request.request_id
        credential_refs = [request.credential_ref] if request.credential_ref else []
        legacy_output = {
            "text_outputs": [f"deterministic {self.transcript_kind} response"],
            "tool_uses": [],
            "runtime_ref": self.runtime_ref,
        }
        events = [
            self._event(
                request,
                agent_run_id,
                1,
                "agent_run_started",
                {"runtime_ref": self.runtime_ref},
                credential_refs,
            ),
            self._event(
                request,
                agent_run_id,
                2,
                "assistant_text",
                {
                    "transcript_kind": self.transcript_kind,
                    "text": f"deterministic {self.transcript_kind} response",
                    "cache_hash": request.cache_policy.frozen_prefix_hash,
                },
                credential_refs,
            ),
            self._event(
                request,
                agent_run_id,
                3,
                "final_result",
                {
                    "success": True,
                    "duration": 0.0,
                    "iteration_count": 0,
                    "error": None,
                    "failover_reason": "none",
                    "legacy_output": legacy_output,
                    "result_kind": "finding",
                    "model_profile_ref": request.model_profile_ref,
                    "credential_key_id": request.credential_ref.key_id
                    if request.credential_ref
                    else None,
                },
                credential_refs,
            ),
        ]
        return AgentRunResult(
            success=True,
            events=events,
            text_output_refs=[],
            tool_uses=[],
            error=None,
            failover_reason="none",
            credential_ref=request.credential_ref,
        )

    def _event(
        self,
        request: AgentRunRequest,
        agent_run_id: str,
        event_index: int,
        event_type: str,
        payload: dict[str, object],
        credential_refs: list[Any],
    ) -> AgentEvent:
        safe_request_id = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in request.request_id
        )
        return AgentEvent(
            event_id=f"{safe_request_id}_event_{event_index:03d}",
            run_id=request.run_id,
            agent_run_id=agent_run_id,
            stage_id=request.stage_id,
            type=event_type,
            payload=payload,
            artifact_refs=[],
            credential_refs=credential_refs,
            timestamp_ms=0,
        )


def create_runtime() -> FakeRuntime:
    return FakeRuntime()
