"""Research memory layer: structured ledgers + evidence cards + context firewall.

See ``docs/concepts/architecture.md`` for the active state boundary.

Public surface:
- card_builder: build EvidenceCard from a finding row
- evidence_pack_builder: assemble shared_evidence_core + private packs
- context_firewall: enforce per-mode token/card budgets
- ledgers/*: append-only YAML-backed ledger stores
- source_resolver: resolve evidence_card.source_ref -> raw file content
"""
