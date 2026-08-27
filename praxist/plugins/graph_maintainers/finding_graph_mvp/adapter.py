"""Executable finding graph maintainer plugin."""

from __future__ import annotations

from praxist.plugins.graph_maintainers.finding_graph_mvp.engine import (
    FindingGraphBuilder,
    FindingGraphMaintainer,
    write_graph_health,
)


class FindingGraphMaintainerPlugin:
    """Plugin façade for finding graph maintenance and session guidance helpers."""

    graph_ref = "graph_maintainer:finding_graph_mvp"
    builder_class = FindingGraphBuilder
    maintainer_class = FindingGraphMaintainer

    def write_health(self, graph_dir):
        return write_graph_health(graph_dir)


def create_graph_maintainer() -> FindingGraphMaintainerPlugin:
    """Manifest entrypoint that constructs the finding graph maintainer plugin."""
    return FindingGraphMaintainerPlugin()
