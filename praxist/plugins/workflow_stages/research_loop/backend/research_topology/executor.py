"""Research-topology executor adapters for the current generation loop."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from .schema import GenerationTopologyContext, ResearchTopologySpec, TopologyPolicy, WorkerSpec

CohortRunner = Callable[[Any, int], Awaitable[list[dict[str, Any]]]]
logger = logging.getLogger(__name__)


def build_legacy_generation_topology(
    context_or_loop: GenerationTopologyContext | Any,
    gen_id: int | None = None,
) -> ResearchTopologySpec:
    """Materialize the current fixed cohort semantics as a topology spec.

    This is intentionally a compatibility adapter: it does not change peer
    scheduling or prompt behavior.  It gives future task-local topology plugins
    and external APIs a stable object to inspect while preserving legacy parity.
    """

    if isinstance(context_or_loop, GenerationTopologyContext):
        context = context_or_loop
    else:
        if gen_id is None:
            raise TypeError("gen_id is required when building topology from a legacy loop")
        context = GenerationTopologyContext.from_loop(context_or_loop, gen_id)
    generation_id = int(context.generation_id)
    cohort_size = int(context.cohort_size)
    nodes: list[WorkerSpec] = []
    for peer_index in range(cohort_size):
        peer_id = f"gen{generation_id}_peer{peer_index}"
        role_ref = (
            context.peer_role_refs[peer_index]
            if peer_index < len(context.peer_role_refs)
            else context.peer_role_ref
        )
        nodes.append(
            WorkerSpec(
                worker_id=peer_id,
                worker_type="experiment_peer",
                role_ref=role_ref,
                required_inputs=[
                    "task_prompt",
                    "frontier_summary",
                    "research_memory_summary",
                    "gems_context",
                ],
                declared_outputs=[
                    "variant_artifact",
                    "result_artifact",
                    "shared_finding",
                    "peer_memory_update",
                ],
                metadata={
                    "generation_id": generation_id,
                    "cohort_index": peer_index,
                    "legacy_peer_id": peer_id,
                },
            )
        )

    return ResearchTopologySpec(
        topology_id=f"legacy_generation_cohort_v1_gen_{generation_id}",
        generation_id=generation_id,
        nodes=nodes,
        policy=TopologyPolicy(
            scheduling="legacy_parallel_generation_cohort",
            retry={"source": "generation_policy_and_runtime"},
            memory_visibility="bounded_task_scoped",
            artifact_visibility="run_dir_task_scoped",
            promotion_policy="generation_boundary_task_defined",
            external_command_policy="queue_for_generation_boundary",
            metadata={
                "compatibility_adapter": True,
                "cohort_size": cohort_size,
                "panel_topology_ref": context.panel_topology_ref,
            },
        ),
        metadata={
            "source": "GenerationLoop compatibility adapter",
            "preserves_legacy_behavior": True,
            "task_id": context.task_id,
        },
    )


class LegacyResearchTopologyExecutor:
    """Execute the existing cohort runner behind a topology contract boundary."""

    def __init__(
        self,
        cohort_runner: CohortRunner | Callable[[], CohortRunner],
        *,
        resolve_each_call: bool = False,
    ) -> None:
        self._cohort_runner = cohort_runner
        self._resolve_each_call = resolve_each_call

    async def execute_generation(self, loop: Any, gen_id: int) -> list[dict[str, Any]]:
        context = GenerationTopologyContext.from_loop(loop, gen_id)
        topology = build_legacy_generation_topology(context)
        try:
            self.persist_topology(Path(context.run_dir) / f"gen_{gen_id}", topology)
        except Exception as exc:  # noqa: BLE001 - sidecar loss must not stop research.
            logger.warning(
                "could not persist research topology sidecar for gen %s: %s", gen_id, exc
            )
        if self._resolve_each_call:
            runner_factory = cast(Callable[[], CohortRunner], self._cohort_runner)
            runner = runner_factory()
        else:
            runner = cast(CohortRunner, self._cohort_runner)
        return await runner(loop, gen_id)

    @staticmethod
    def persist_topology(gen_dir: Path, topology: ResearchTopologySpec) -> Path:
        gen_dir.mkdir(parents=True, exist_ok=True)
        path = gen_dir / "research_topology.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(topology.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)
        return path
