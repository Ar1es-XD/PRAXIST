# Peer-Local Structured Memory For Long-Context Continuity

This document describes the Praxist mechanism that improves peer continuity
across multiple autonomous sessions without carrying raw transcripts forward.

## Design Goal

Each peer may span multiple runtime sessions inside one generation. A later
session should understand what the same peer already tried, what evidence it
created, what sibling peers shared, and where the current hypothesis stands. It
should not need a single unbounded chat transcript to do that.

The mechanism preserves the useful parts of a long continuous context:

- current hypothesis and open questions;
- experiment ledger and abandoned branches;
- prior session handoff;
- relevant new shared findings;
- Deep Innovation Gate (DIG) selected-contract state when present;
- anti-anchoring prompts that force reconsideration before repeating work.

It deliberately does not preserve raw message transcripts in the prompt.

## Session Boundary

The research-loop backend updates memory around each peer session:

```text
AutonomousAgentLoop._run_session()
  -> build session_id
  -> compose base task prompt + peer-local memory block
  -> execute runtime session
  -> record structured session result
```

Task-local content remains unchanged; memory supplies execution continuity and
audit artifacts only.

## Artifact Layout

For each peer:

```text
runs/<run_id>/gen_<N>/peers/<peer_id>/memory/
  peer_state.yaml
  experiment_ledger.jsonl
  session_handoff.md
  seen_shared_findings.json
  memory_prompt.md
```

`peer_state.yaml` is the compact state card:

- current peer identity;
- current hypothesis;
- open questions;
- known dead ends;
- active variant;
- last session status;
- recent result artifacts discovered for the peer.

`experiment_ledger.jsonl` is the append-only local ledger:

- session id;
- concise summary;
- success/failure;
- duration and tool count where available;
- link to the session log;
- compact metrics if result artifacts are found.

`session_handoff.md` is the human-readable session boundary summary.

`seen_shared_findings.json` tracks which shared findings were already surfaced
to this peer's session prompt.

`memory_prompt.md` records the exact bounded memory block injected into the most
recent runtime prompt for auditability. Per-session prompt and manifest
snapshots are retained only up to a bounded count; the stable
`memory_prompt.md` and `session_prompt_manifest.json` files remain the latest
audit pointers.

## Prompt Injection

The runtime appends a bounded section titled:

```text
Praxist Peer-Local Structured Memory
```

The block contains:

- memory discipline requirements;
- current peer state;
- selected DIG contract snapshot if one exists;
- recent peer-local experiment ledger entries;
- new shared findings since the last session;
- previous handoff summary;
- anti-anchoring check.

The section is bounded by a character budget. If it grows too large, it is
truncated explicitly. This keeps later sessions grounded without allowing memory
to become an uncontrolled raw transcript replay.

This prompt block is peer-local and session-local. It does not broadcast full
sibling peer contracts, does not replay raw transcripts, and does not grow
linearly across generations.

## DIG Compatibility

When DIG runs, memory includes a bounded view of:

```text
runs/<run_id>/gen_<N>/peers/<peer_id>/dig/selected_contract.yaml
```

The selected-contract schema and generation scope are defined in
[Deep Innovation Gate](deep-innovation-gate.md). Memory neither changes that
contract nor turns it into empirical evidence.

## Shared Findings Refresh

The memory layer reads the generation's shared-findings directory and surfaces
new JSON findings that the peer has not yet seen. This lets later sessions
benefit from sibling peers without depending on full shared transcript replay.

The prompt shows only compact metadata:

- finding id;
- finding type;
- producer;
- title or summary.

After a session ends, surfaced findings are marked as seen.

## Anti-Anchoring Behavior

Every injected memory block asks the peer to answer three questions before
continuing the same direction:

```text
1. What evidence supports continuing the current mechanism?
2. What evidence suggests pivoting, ablating, or simplifying?
3. What is the cheapest evidence that could falsify continuation?
```

This is intended to preserve the continuity benefits of long context while
reducing over-commitment to a stale local idea.

## Task Boundary

The mechanism is task-agnostic:

- it does not mention domain-specific metrics;
- it does not modify any task project;
- it does not change benchmark, evaluator, or data semantics;
- it does not alter peer count, model routing, Gems, or Frontier ranking.

Task-local prompts and role skills remain the right place for domain-specific
research instructions.

## Memory Failure Behavior

If memory files are missing, malformed, or absent, the runtime initializes a
minimal state card and proceeds. A broken memory artifact should not block a peer
from executing its assigned research task.

Session completion always attempts to write:

- a ledger row;
- an updated state card;
- a handoff note.

If the runtime session fails, the handoff and ledger still capture the failure
reason where available.

## Expected Effect

Compared with raw multi-session replay, this mechanism should:

- improve continuity across peer sessions;
- reduce repeated dead-end work;
- make session boundaries auditable;
- improve use of sibling findings;
- keep prompt growth bounded;
- preserve cross-peer diversity by forcing local anti-anchoring checks.

It is a continuity layer, not a new research selector. The generation-level
selection logic still belongs to DIG, Frontier, Gems, the Principal
Investigator (PI) panel, and its Chair.
