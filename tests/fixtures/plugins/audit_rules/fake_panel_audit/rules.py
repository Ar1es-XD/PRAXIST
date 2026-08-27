"""Executable fake panel audit plugin."""

from __future__ import annotations

from typing import Any


class FakePanelAudit:
    audit_ref = "audit_rule:fake_panel_audit"

    def audit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"passed": True, "violations": [], "checked_fields": sorted(payload)}


def create_audit_rule() -> FakePanelAudit:
    return FakePanelAudit()
