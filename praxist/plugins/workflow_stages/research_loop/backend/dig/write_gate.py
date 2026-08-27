"""In-process write/command gate model for DIG-Lite.

The production research-loop integration enforces DIG by running the planner
before peer implementation starts and by giving that planner a read-only tool
allowlist. This object captures the same contract for tests and for future
fine-grained runtime adapters that can call ``can_write`` / ``can_run_command``
directly.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from .config import DIGLiteConfig
from .schema import SelectedContract
from .validator import DIGValidationContext, validate_selected_contract


class PeerPhase(StrEnum):
    """DIG peer lifecycle phases used by the in-process gate model."""

    DIG = "DIG"
    IMPLEMENTATION = "IMPLEMENTATION"


class DIGWriteGate:
    """Validate write and command permissions before and after DIG unlock."""

    def __init__(self, dig_dir: str | Path, config: DIGLiteConfig):
        self.phase = PeerPhase.DIG
        self.dig_dir = Path(dig_dir).resolve()
        self.config = config

    def can_write(self, path: str | Path) -> bool:
        path_obj = Path(path)
        normalized = str(path_obj).replace("\\", "/")
        if self.phase == PeerPhase.IMPLEMENTATION:
            return True
        if self.config.write_gate.block_variant_dir_before_unlock and "/variants/" in normalized:
            return False
        if self.config.write_gate.block_results_dir_before_unlock and "/results/" in normalized:
            return False
        if "assets/harness/baseline/" in normalized:
            return False
        if not self.config.write_gate.allow_writes_only_under_dig_dir_before_unlock:
            return True
        try:
            path_obj.resolve().relative_to(self.dig_dir)
        except ValueError:
            return False
        return True

    def can_create_directory(self, path: str | Path) -> bool:
        return self.can_write(path)

    def can_run_command(self, command: str) -> bool:
        return not (
            self.phase == PeerPhase.DIG and self.config.write_gate.block_shell_before_unlock
        )

    def unlock(self, contract: SelectedContract, ctx: DIGValidationContext) -> None:
        validate_selected_contract(contract, ctx, self.config)
        self.phase = PeerPhase.IMPLEMENTATION
