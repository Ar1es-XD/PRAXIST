"""Compatibility facade for Multi-PI panel execution.

Gate C moves the topology-specific implementation out of this public module so
callers keep importing ``run_panel`` while the executable topology can evolve
behind a plugin boundary.
"""

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.legacy_two_round_executor import (
    PanelMode,
    PanelResult,
    run_panel,
)

__all__ = ["PanelMode", "PanelResult", "run_panel"]
