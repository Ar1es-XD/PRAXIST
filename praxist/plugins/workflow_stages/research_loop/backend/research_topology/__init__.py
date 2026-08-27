"""Research-loop topology contracts and compatibility execution adapters."""

from .api import ResearchLoopModuleAPI
from .executor import LegacyResearchTopologyExecutor, build_legacy_generation_topology
from .schema import (
    GenerationTopologyContext,
    ResearchCommand,
    ResearchEvent,
    ResearchTopologySpec,
    TopologyChangeRequest,
    TopologyEdge,
    TopologyPolicy,
    WorkerCapabilitySet,
    WorkerSpec,
    WorkUnit,
)

__all__ = [
    "LegacyResearchTopologyExecutor",
    "GenerationTopologyContext",
    "ResearchCommand",
    "ResearchEvent",
    "ResearchLoopModuleAPI",
    "ResearchTopologySpec",
    "TopologyChangeRequest",
    "TopologyEdge",
    "TopologyPolicy",
    "WorkerCapabilitySet",
    "WorkerSpec",
    "WorkUnit",
    "build_legacy_generation_topology",
]
