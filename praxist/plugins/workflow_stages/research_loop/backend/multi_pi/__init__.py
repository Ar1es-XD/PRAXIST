"""Multi-PI panel and heterogeneous-KB synthesis support.

See ``docs/concepts/architecture.md`` for the active workflow boundary.
"""

from praxist.plugins.workflow_stages.research_loop.backend.multi_pi.panel_runner import (
    PanelMode,
    PanelResult,
    run_panel,
)

__all__ = ["PanelMode", "PanelResult", "run_panel"]
