"""Built-in V2 product-usage protocol and client components for Praxist."""

from praxist import __version__

from .lifecycle import PeerStatusSummary, RunTelemetryContext
from .protocol import (
    CONSENT_NOTICE_VERSION,
    SCHEMA_VERSION,
    GenerationFinishedEvent,
    RunFinishedEvent,
    RunReconciledEvent,
    RunStartedEvent,
    UsageBatch,
    UsageEvent,
)

__all__ = [
    "__version__",
    "CONSENT_NOTICE_VERSION",
    "SCHEMA_VERSION",
    "GenerationFinishedEvent",
    "PeerStatusSummary",
    "RunFinishedEvent",
    "RunReconciledEvent",
    "RunStartedEvent",
    "RunTelemetryContext",
    "UsageBatch",
    "UsageEvent",
]
