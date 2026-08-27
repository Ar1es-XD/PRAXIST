"""``praxist stop`` — selective and bulk termination of Praxist runs.

CLI lifecycle Phase 3 + Phase 4.  Two invocation shapes:

* ``praxist stop <run_id>`` — read the registry entry for ``run_id``,
  validate the recorded PID still belongs to Praxist (TOCTOU guard via
  ``ps`` cmdline prefix), walk descendant processes via the same
  ``ps`` table ``praxist status`` already uses, send SIGTERM, wait for a
  configurable grace period, then SIGKILL stragglers.  Rewrite the
  registry entry as ``stopped``.
* ``praxist stop --all`` — bulk stop path.  Default scope is the union of
  registry entries and ``ps``-scan matches; ``--registry-only`` /
  ``--ps-scan-only`` narrow it.  Each scope ends with the same
  SIGTERM → wait → SIGKILL escalation.

Design constraints from issue #99:

* TOCTOU on stop: between reading the registry entry and signalling
  the PID, the PID may have been recycled.  We re-read the live ``ps``
  table and confirm the command line starts with the prefix recorded
  by ``praxist start`` before signalling.
* Cross-platform descendants: ``/proc`` is Linux-only.  We rely on the
  Phase 1 ``ps`` table (already cross-platform) and build a children
  index from ``(pid, ppid)`` pairs.
* Stop signal protocol: SIGTERM → wait → SIGKILL with a configurable
  grace period (default 5s).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from praxist.cli.registry import (
    STATE_STOPPED,
    RegistryEntry,
    RegistryError,
    entry_is_local,
    entry_lock,
    entry_process_epoch_matches,
    list_entries,
    process_identity_matches,
    process_start_token,
    read_entry,
    registry_lock,
    remove_entry,
    update_state,
)
from praxist.cli.status import (
    pid_is_alive,
    praxist_process_regexes,
    read_ps_table,
    registry_command_matches,
    self_ancestor_pids,
)

if TYPE_CHECKING:
    from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
        ProtectedEntry,
    )

DEFAULT_GRACE_SECONDS = 5.0
"""Seconds between SIGTERM and SIGKILL."""
_LATE_DRAIN_SCAN_SECONDS = 0.2
_LATE_DRAIN_MAX_SCANS = 12
_LATE_DRAIN_TERM_SCANS = 2
_LATE_DRAIN_STABLE_EMPTY_SCANS = 3
SCOPE_UNION = "union"
SCOPE_REGISTRY = "registry"
SCOPE_PS = "ps-scan"
_PROCESS_MATCH = "match"
_PROCESS_GONE = "gone"
_PROCESS_MISMATCH = "mismatch"
_PROCESS_UNKNOWN = "unknown"


@dataclass
class StopOutcome:
    """Structured result of a stop attempt.

    Attributes:
        run_id: The registry run_id when ``praxist stop <run_id>``; None
            for ``--all`` scopes (which can span multiple runs).
        matched_pids: The PID set targeted by the stop.
        descendant_pids: PIDs added to the kill set via descendant
            walking.
        terminated_pids: PIDs that exited within the SIGTERM grace
            window.
        killed_pids: PIDs that required SIGKILL.
        remaining_pids: PIDs still alive after SIGKILL (usually
            permission errors).
        failed_run_ids: Registry runs that could not be stopped safely and
            therefore remain in the running state.
        monitor_sessions: Compatibility field; direct foreground monitors are
            not owned by run lifecycle commands, so new stops leave it empty.
        monitor_stopped_sessions: Compatibility field paired with
            ``monitor_sessions``; new stops leave it empty.
        dry_run: True when the command was invoked with ``--dry-run``.
        warnings: Operator-visible warnings (TOCTOU mismatches, EPERM,
            etc.).
    """

    run_id: str | None
    matched_pids: list[int] = field(default_factory=list)
    descendant_pids: list[int] = field(default_factory=list)
    terminated_pids: list[int] = field(default_factory=list)
    killed_pids: list[int] = field(default_factory=list)
    remaining_pids: list[int] = field(default_factory=list)
    failed_run_ids: list[str] = field(default_factory=list)
    monitor_sessions: list[str] = field(default_factory=list)
    monitor_stopped_sessions: list[str] = field(default_factory=list)
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "matched_pids": list(self.matched_pids),
            "descendant_pids": list(self.descendant_pids),
            "terminated_pids": list(self.terminated_pids),
            "killed_pids": list(self.killed_pids),
            "remaining_pids": list(self.remaining_pids),
            "failed_run_ids": list(self.failed_run_ids),
            "monitor_sessions": list(self.monitor_sessions),
            "monitor_stopped_sessions": list(self.monitor_stopped_sessions),
            "dry_run": self.dry_run,
            "warnings": list(self.warnings),
        }


@dataclass
class GcOutcome:
    """Structured result of an ``praxist stop --gc`` registry sweep.

    Attributes:
        removed_run_ids: Registry entries whose file was unlinked
            (stale rows whose PIDs were dead or whose recorded command
            prefix no longer matched the live process).
        kept_run_ids: Entries left in place because the PID is still
            alive AND the recorded command prefix still matches.
        dry_run: True when the command was invoked with ``--dry-run``;
            ``removed_run_ids`` then lists the *would-be-removed* ids
            without actually unlinking.
        warnings: Operator-visible warnings (per-entry read errors,
            unexpected ``remove_entry`` failures, …).
    """

    removed_run_ids: list[str] = field(default_factory=list)
    kept_run_ids: list[str] = field(default_factory=list)
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "removed_run_ids": list(self.removed_run_ids),
            "kept_run_ids": list(self.kept_run_ids),
            "dry_run": self.dry_run,
            "warnings": list(self.warnings),
        }


class StopError(RuntimeError):
    """Raised when ``praxist stop`` cannot proceed."""


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``praxist stop`` subcommand on the parent parser."""
    parser = subparsers.add_parser(
        "stop",
        help="Stop a Praxist run by run_id, or stop everything with --all.",
        description=(
            "``praxist stop <run_id>`` terminates one specific run via its "
            "registry entry.  ``praxist stop --all`` terminates every "
            "Praxist-recognised process — by default the union of registry "
            "entries and ``ps``-scan matches.\n\n"
            "Registry-backed runs close new admission before discovery. Both "
            "modes send SIGTERM, wait --grace seconds, then SIGKILL any "
            "process still alive; registry-backed runs also perform a bounded "
            "stable-empty rescan for late children."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        help="Run id (filename stem of $PRAXIST_STATE_DIR/runs/<run_id>.json).",
    )
    parser.add_argument(
        "--all",
        dest="all",
        action="store_true",
        help="Stop every recognised Praxist run (registry + ps-scan by default).",
    )
    parser.add_argument(
        "--registry-only",
        dest="registry_only",
        action="store_true",
        help="With --all: only target registry-managed runs.",
    )
    parser.add_argument(
        "--ps-scan-only",
        dest="ps_scan_only",
        action="store_true",
        help="With --all: only target unregistered runs found by the process scan.",
    )
    parser.add_argument(
        "--grace",
        dest="grace_seconds",
        type=float,
        default=DEFAULT_GRACE_SECONDS,
        help=f"Seconds to wait after SIGTERM before SIGKILL (default {DEFAULT_GRACE_SECONDS}).",
    )
    parser.add_argument(
        "--gc",
        dest="gc",
        action="store_true",
        help=(
            "Remove stale registry entries. A stale entry is one "
            "whose recorded PID is no longer alive, or whose live "
            "command line no longer matches the prefix recorded at "
            "``praxist start`` time (PID recycling). No signals are sent."
        ),
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help=(
            "Show what would be signalled without sending any signals. "
            "With --gc, list the would-be-removed entries without "
            "deleting any files."
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit a JSON outcome document instead of the operator summary.",
    )
    parser.set_defaults(func=cmd_stop)


def cmd_stop(args: argparse.Namespace) -> int:
    """Handler for ``praxist stop``."""
    # #166: ``--gc`` is its own dispatch branch; it doesn't signal anything,
    # so the other flag combos that only make sense for signalling are
    # mutually exclusive with it.
    if args.gc:
        if args.all or args.run_id or args.registry_only or args.ps_scan_only:
            sys.stderr.write(
                "praxist stop: --gc cannot be combined with <run_id> / --all / "
                "--registry-only / --ps-scan-only.\n"
            )
            return 2
        gc_outcome = gc_stale_entries(dry_run=args.dry_run)
        if args.as_json:
            sys.stdout.write(json.dumps(gc_outcome.to_dict(), indent=2) + "\n")
        else:
            _write_gc_outcome_summary(gc_outcome)
        return 0

    if args.all and args.run_id:
        sys.stderr.write("praxist stop: --all and <run_id> are mutually exclusive.\n")
        return 2
    if not args.all and not args.run_id:
        sys.stderr.write("praxist stop: expected <run_id>, --all, or --gc.\n")
        return 2
    if args.registry_only and args.ps_scan_only:
        sys.stderr.write(
            "praxist stop: --registry-only and --ps-scan-only are mutually exclusive.\n"
        )
        return 2
    if not math.isfinite(args.grace_seconds) or args.grace_seconds < 0:
        sys.stderr.write("praxist stop: --grace must be a finite non-negative number.\n")
        return 2

    try:
        if args.all:
            scope = (
                SCOPE_REGISTRY
                if args.registry_only
                else SCOPE_PS
                if args.ps_scan_only
                else SCOPE_UNION
            )
            outcome = stop_all(
                scope=scope,
                grace_seconds=args.grace_seconds,
                dry_run=args.dry_run,
            )
        else:
            outcome = stop_run(
                run_id=args.run_id,
                grace_seconds=args.grace_seconds,
                dry_run=args.dry_run,
            )
    except StopError as exc:
        sys.stderr.write(f"praxist stop: {exc}\n")
        return 1

    if args.as_json:
        sys.stdout.write(json.dumps(outcome.to_dict(), indent=2) + "\n")
    else:
        _write_outcome_summary(outcome)
    if outcome.remaining_pids or outcome.failed_run_ids:
        return 1
    return 0


def stop_run(
    *,
    run_id: str,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    dry_run: bool = False,
    now_iso: str | None = None,
) -> StopOutcome:
    """Stop the run identified by ``run_id`` via its registry entry."""
    _validate_grace(grace_seconds)
    try:
        with entry_lock(run_id):
            return _stop_run_locked(
                run_id=run_id,
                grace_seconds=grace_seconds,
                dry_run=dry_run,
                now_iso=now_iso,
            )
    except (RegistryError, ValueError) as exc:
        raise StopError(str(exc)) from exc


def _stop_run_locked(
    *,
    run_id: str,
    grace_seconds: float,
    dry_run: bool,
    now_iso: str | None,
) -> StopOutcome:
    """Stop one run while its lifecycle lock is held."""
    try:
        entry = read_entry(run_id)
    except Exception as exc:  # registry.RegistryError or filesystem issues
        raise StopError(str(exc)) from exc
    if entry_is_local(entry) is False:
        host = entry.extra.get("hostname", "another host")
        raise StopError(f"run_id={run_id} belongs to {host!r}; stop it on that host instead")
    outcome = StopOutcome(run_id=run_id, dry_run=dry_run)
    if entry_process_epoch_matches(entry) is False:
        outcome.warnings.append(
            f"run_id={run_id}: process epoch does not match the current host boot; "
            "registry state was preserved and no process signals were sent"
        )
        outcome.failed_run_ids.append(run_id)
        return outcome
    endpoint_path = Path(entry.run_dir) / "resource_scheduler" / "endpoint.json"
    admission_closed = False
    if not dry_run:
        admission_closed = _write_run_shutdown_fence(
            Path(entry.run_dir),
            outcome,
            source="praxist_stop",
        )
    if not dry_run and endpoint_path.exists():
        try:
            from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
                freeze_all_for_run,
            )

            scheduler_frozen = freeze_all_for_run(Path(entry.run_dir), "praxist_stop")
            admission_closed = admission_closed or scheduler_frozen
            if not scheduler_frozen:
                outcome.warnings.append("central scheduler stop fence was unavailable")
        except Exception as exc:  # noqa: BLE001 - legacy runs have no scheduler.
            outcome.warnings.append(f"central scheduler stop fence failed: {exc}")

    if not dry_run and not admission_closed:
        raise StopError(
            f"run_id={run_id}: could not close experiment admission; no processes were signalled"
        )

    ps_rows, snapshot_tokens = _read_identity_bound_ps_snapshot()
    root_pid = _validate_registry_root(entry, ps_rows, outcome)
    scheduler_targets, scheduler_group_members = _scheduler_owned_state(
        Path(entry.run_dir),
        ps_rows,
    )
    scheduler_pgids = set(scheduler_group_members)
    if root_pid is None and not scheduler_targets and not scheduler_pgids:
        # The controller can exit before a just-forked evaluator becomes
        # visible. Scan run-owned process evidence until it is stably empty.
        if not dry_run:
            _drain_late_run_processes(
                {run_id: Path(entry.run_dir)},
                outcome,
            )
            _record_unresolved_protected_groups(
                run_id,
                Path(entry.run_dir),
                outcome,
            )
        if not dry_run and not outcome.remaining_pids and not outcome.failed_run_ids:
            update_state(run_id, STATE_STOPPED, stopped_at=now_iso or _utc_now_iso())
        return outcome

    roots = {root_pid} if root_pid is not None else set()
    pids = _gather_targets(root_pids=roots, ps_rows=ps_rows)
    pids.update(scheduler_targets)
    outcome.matched_pids = sorted(roots)
    outcome.descendant_pids = sorted(pids - roots)

    if dry_run:
        return outcome

    process_tokens = {pid: token for pid in pids if (token := snapshot_tokens.get(pid, ""))}
    recorded_root_token = entry.extra.get("process_start_token", "").strip()
    if root_pid is not None and recorded_root_token:
        process_tokens[root_pid] = recorded_root_token
    for members in scheduler_group_members.values():
        process_tokens.update(members)

    _signal_process_groups(
        scheduler_pgids,
        signal.SIGTERM,
        outcome,
        group_members=scheduler_group_members,
    )
    root_fallbacks = {root_pid: entry} if root_pid is not None else {}
    _signal_set(
        pids,
        signal.SIGTERM,
        outcome,
        process_tokens=process_tokens,
        fallback_entries=root_fallbacks,
    )
    _await_exit(
        pids,
        grace_seconds,
        outcome,
        process_tokens=process_tokens,
    )
    survivors = _live_process_instances(pids, process_tokens)
    surviving_groups = {
        pgid
        for pgid in scheduler_pgids
        if _process_group_instance_alive(
            pgid,
            scheduler_group_members.get(pgid),
        )
    }
    if survivors or surviving_groups:
        _signal_process_groups(
            surviving_groups,
            signal.SIGKILL,
            outcome,
            group_members=scheduler_group_members,
        )
        _signal_set(
            survivors,
            signal.SIGKILL,
            outcome,
            process_tokens=process_tokens,
            fallback_entries=root_fallbacks,
        )
        time.sleep(min(1.0, grace_seconds))
    remaining = {pid for pid in outcome.remaining_pids if pid_is_alive(pid)}
    remaining.update(_live_process_instances(pids, process_tokens))
    remaining.update(
        pgid
        for pgid in scheduler_pgids
        if _process_group_instance_alive(
            pgid,
            scheduler_group_members.get(pgid),
        )
    )
    outcome.remaining_pids = sorted(remaining)
    _drain_late_run_processes(
        {run_id: Path(entry.run_dir)},
        outcome,
    )
    _record_unresolved_protected_groups(
        run_id,
        Path(entry.run_dir),
        outcome,
    )
    remaining = set(outcome.remaining_pids)
    if not remaining and not outcome.failed_run_ids:
        update_state(run_id, STATE_STOPPED, stopped_at=now_iso or _utc_now_iso())
    return outcome


def stop_all(
    *,
    scope: str = SCOPE_UNION,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    dry_run: bool = False,
    now_iso: str | None = None,
) -> StopOutcome:
    """Stop every recognised Praxist run within ``scope``."""
    _validate_grace(grace_seconds)
    if scope not in (SCOPE_REGISTRY, SCOPE_UNION):
        return _stop_all_locked(
            scope=scope,
            grace_seconds=grace_seconds,
            dry_run=dry_run,
            now_iso=now_iso,
        )

    try:
        with registry_lock():
            observed_ids = sorted({entry.run_id for entry in list_entries()})
            with contextlib.ExitStack() as locks:
                for run_id in observed_ids:
                    locks.enter_context(entry_lock(run_id))
                locked_entries: list[RegistryEntry] = []
                read_warnings: list[str] = []
                for run_id in observed_ids:
                    try:
                        locked_entries.append(read_entry(run_id))
                    except RegistryError as exc:
                        read_warnings.append(
                            f"run_id={run_id}: registry changed before stop: {exc}"
                        )
                return _stop_all_locked(
                    scope=scope,
                    grace_seconds=grace_seconds,
                    dry_run=dry_run,
                    now_iso=now_iso,
                    registry_entries_snapshot=locked_entries,
                    initial_warnings=read_warnings,
                )
    except (RegistryError, ValueError) as exc:
        raise StopError(str(exc)) from exc


def _stop_all_locked(
    *,
    scope: str,
    grace_seconds: float,
    dry_run: bool,
    now_iso: str | None,
    registry_entries_snapshot: list[RegistryEntry] | None = None,
    initial_warnings: list[str] | None = None,
) -> StopOutcome:
    """Stop all selected work while registry-backed run locks are held."""
    outcome = StopOutcome(run_id=None, dry_run=dry_run)
    outcome.warnings.extend(initial_warnings or ())

    roots: set[int] = set()
    registry_pid_to_run_id: dict[int, str] = {}
    registry_entries_by_root: dict[int, RegistryEntry] = {}
    strong_registry_pids: set[int] = set()
    registry_entries: list[RegistryEntry] = []
    unfenced_registry_entries: list[RegistryEntry] = []
    scheduler_targets_by_run: dict[str, set[int]] = {}
    scheduler_group_members_by_run: dict[str, dict[int, dict[int, str]]] = {}

    if scope in (SCOPE_REGISTRY, SCOPE_UNION):
        entries = (
            registry_entries_snapshot if registry_entries_snapshot is not None else list_entries()
        )
        for entry in entries:
            if entry_is_local(entry) is False:
                outcome.warnings.append(
                    f"run_id={entry.run_id}: registry entry belongs to "
                    f"{entry.extra.get('hostname', 'another host')}; skipping"
                )
                continue
            if entry_process_epoch_matches(entry) is False:
                outcome.warnings.append(
                    f"run_id={entry.run_id}: process epoch does not match the current host boot; "
                    "registry state was preserved and no process signals were sent"
                )
                outcome.failed_run_ids.append(entry.run_id)
                continue
            registry_entries.append(entry)
            if entry.extra.get("process_start_token", "").strip():
                strong_registry_pids.add(entry.pid)

    if not dry_run:
        fenced_entries: list[RegistryEntry] = []
        for entry in registry_entries:
            admission_closed = _write_run_shutdown_fence(
                Path(entry.run_dir),
                outcome,
                source="praxist_stop_all",
            )
            endpoint_path = Path(entry.run_dir) / "resource_scheduler" / "endpoint.json"
            if endpoint_path.exists():
                try:
                    from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
                        freeze_all_for_run,
                    )

                    scheduler_frozen = freeze_all_for_run(
                        Path(entry.run_dir),
                        "praxist_stop_all",
                    )
                    admission_closed = admission_closed or scheduler_frozen
                    if not scheduler_frozen:
                        outcome.warnings.append(
                            f"central scheduler stop fence unavailable for {entry.run_id}"
                        )
                except Exception as exc:  # noqa: BLE001
                    outcome.warnings.append(
                        f"central scheduler stop fence failed for {entry.run_id}: {exc}"
                    )
            if admission_closed:
                fenced_entries.append(entry)
            else:
                unfenced_registry_entries.append(entry)
                outcome.failed_run_ids.append(entry.run_id)
                outcome.warnings.append(
                    f"run_id={entry.run_id}: could not close experiment admission; "
                    "the run was not signalled"
                )
        registry_entries = fenced_entries

    ps_rows, snapshot_tokens = _read_identity_bound_ps_snapshot()
    excluded = self_ancestor_pids(ps_rows)
    for entry in unfenced_registry_entries:
        root = _validate_registry_root(entry, ps_rows, outcome)
        protected_roots = {root} if root is not None else set()
        scheduler_targets, scheduler_groups = _scheduler_owned_state(
            Path(entry.run_dir),
            ps_rows,
        )
        protected_roots.update(scheduler_targets)
        protected_roots.update(pid for members in scheduler_groups.values() for pid in members)
        resolved_run_dir = Path(entry.run_dir).expanduser().resolve(strict=False)
        protected_roots.update(
            pid
            for pid in ps_rows
            if _run_dir_from_process_environment(pid) == resolved_run_dir
            or _process_cwd_is_run_workspace(pid, resolved_run_dir)
        )
        protected = _gather_targets(root_pids=protected_roots, ps_rows=ps_rows)
        excluded.update(protected)
        outcome.remaining_pids = sorted(
            set(outcome.remaining_pids) | {pid for pid in protected if pid_is_alive(pid)}
        )
    for entry in registry_entries:
        root = _validate_registry_root(entry, ps_rows, outcome)
        if root is None:
            continue
        roots.add(root)
        registry_pid_to_run_id[root] = entry.run_id
        registry_entries_by_root[root] = entry

    ps_scan_allowed = scope in (SCOPE_PS, SCOPE_UNION) and not unfenced_registry_entries
    if scope == SCOPE_UNION and unfenced_registry_entries:
        outcome.warnings.append(
            "independent ps scan was skipped because at least one registry run "
            "could not close experiment admission"
        )
    if ps_scan_allowed:
        regexes = praxist_process_regexes()
        for pid, (_ppid, _etime, command) in ps_rows.items():
            if pid in excluded:
                continue
            if any(regex.search(command) for regex in regexes):
                if pid in strong_registry_pids and pid not in registry_entries_by_root:
                    continue
                if pid in snapshot_tokens:
                    roots.add(pid)
                elif pid_is_alive(pid):
                    outcome.remaining_pids = sorted(set(outcome.remaining_pids) | {pid})
                    outcome.warnings.append(
                        f"ps-scan matched pid {pid}, but its process identity "
                        "was not stable across discovery; skipping it"
                    )

    pids = _gather_targets(root_pids=roots, ps_rows=ps_rows, excluded=excluded)
    for entry in registry_entries:
        targets, group_members = _scheduler_owned_state(
            Path(entry.run_dir),
            ps_rows,
        )
        scheduler_group_members_by_run[entry.run_id] = group_members
        scheduler_targets_by_run[entry.run_id] = targets
        pids.update(targets)
    outcome.matched_pids = sorted(roots)
    outcome.descendant_pids = sorted(pids - roots)

    if dry_run:
        return outcome

    scheduler_group_members = {
        pgid: members
        for groups in scheduler_group_members_by_run.values()
        for pgid, members in groups.items()
    }
    scheduler_pgids = set(scheduler_group_members)
    process_tokens = {pid: token for pid in pids if (token := snapshot_tokens.get(pid, ""))}
    for root, entry in registry_entries_by_root.items():
        recorded = entry.extra.get("process_start_token", "").strip()
        if recorded:
            process_tokens[root] = recorded
    for members in scheduler_group_members.values():
        process_tokens.update(members)
    _signal_process_groups(
        scheduler_pgids,
        signal.SIGTERM,
        outcome,
        group_members=scheduler_group_members,
    )
    _signal_set(
        pids,
        signal.SIGTERM,
        outcome,
        process_tokens=process_tokens,
        fallback_entries=registry_entries_by_root,
    )
    _await_exit(
        pids,
        grace_seconds,
        outcome,
        process_tokens=process_tokens,
    )
    survivors = _live_process_instances(pids, process_tokens)
    surviving_groups = {
        pgid
        for pgid in scheduler_pgids
        if _process_group_instance_alive(
            pgid,
            scheduler_group_members.get(pgid),
        )
    }
    if survivors or surviving_groups:
        _signal_process_groups(
            surviving_groups,
            signal.SIGKILL,
            outcome,
            group_members=scheduler_group_members,
        )
        _signal_set(
            survivors,
            signal.SIGKILL,
            outcome,
            process_tokens=process_tokens,
            fallback_entries=registry_entries_by_root,
        )
        time.sleep(min(1.0, grace_seconds))
    remaining = {pid for pid in outcome.remaining_pids if pid_is_alive(pid)}
    remaining.update(_live_process_instances(pids, process_tokens))
    remaining.update(
        pgid
        for pgid in scheduler_pgids
        if _process_group_instance_alive(
            pgid,
            scheduler_group_members.get(pgid),
        )
    )
    outcome.remaining_pids = sorted(remaining)
    remaining_by_run = _drain_late_run_processes(
        {entry.run_id: Path(entry.run_dir) for entry in registry_entries},
        outcome,
    )
    for entry in registry_entries:
        _record_unresolved_protected_groups(
            entry.run_id,
            Path(entry.run_dir),
            outcome,
        )

    # A dead orchestrator root does not imply that scheduler-owned groups are
    # gone. Mark an entry stopped only after every target associated with it
    # has drained.
    stopped_at = now_iso or _utc_now_iso()
    run_roots = {run_id: pid for pid, run_id in registry_pid_to_run_id.items()}
    for entry in registry_entries:
        if entry.run_id in outcome.failed_run_ids:
            continue
        owned = scheduler_targets_by_run.get(entry.run_id, set())
        owned_pgids = set(scheduler_group_members_by_run.get(entry.run_id, {}))
        root = run_roots.get(entry.run_id)
        if root is None and pid_is_alive(entry.pid):
            continue
        if remaining_by_run.get(entry.run_id):
            continue
        targets = set(owned)
        if root is not None:
            targets.add(root)
        if not ((targets | owned_pgids) & set(outcome.remaining_pids)):
            try:
                update_state(entry.run_id, STATE_STOPPED, stopped_at=stopped_at)
            except Exception as exc:  # registry write race / missing file
                outcome.warnings.append(f"could not mark run_id={entry.run_id} as stopped: {exc}")
    return outcome


def _write_run_shutdown_fence(
    run_dir: Path,
    outcome: StopOutcome,
    *,
    source: str,
) -> bool:
    """Close run admission before process discovery using the existing sentinel."""

    path = Path(run_dir) / "ORCHESTRATOR_SHUTDOWN"
    try:
        if path.exists():
            return True
        path.write_text(
            f"source={source}\nat={time.time():.0f}\n",
            encoding="utf-8",
        )
        return True
    except OSError as exc:
        outcome.warnings.append(f"could not write run shutdown fence at {path}: {exc}")
        return False


def _run_dir_from_process_environment(pid: int) -> Path | None:
    """Read the exact inherited run directory from Linux process evidence."""

    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for item in raw.split(b"\0"):
        key, separator, value = item.partition(b"=")
        if not separator or key not in {b"PRAXIST_RUN_DIR", b"AUTO_RESEARCH_RUN_DIR"}:
            continue
        try:
            decoded = os.fsdecode(value).strip()
        except (TypeError, UnicodeError):
            continue
        if decoded:
            return Path(decoded).expanduser().resolve(strict=False)
    return None


def _process_cwd_is_run_workspace(pid: int, run_dir: Path) -> bool:
    """Recognize only framework-owned peer workspaces, not an arbitrary run cwd."""

    try:
        cwd = Path(os.readlink(f"/proc/{pid}/cwd")).resolve(strict=False)
        relative = cwd.relative_to(run_dir)
    except (OSError, ValueError):
        return False
    return bool(relative.parts and relative.parts[0] == "peer_workspaces")


def _discover_run_owned_processes(
    run_dir: Path,
) -> tuple[dict[int, str], dict[int, dict[int, str]]]:
    """Discover scheduler-owned or exact run-environment process instances."""

    rows, stable_tokens = _read_identity_bound_ps_snapshot()
    excluded = self_ancestor_pids(rows)
    scheduler_targets, group_members = _scheduler_owned_state(run_dir, rows)
    resolved_run_dir = Path(run_dir).expanduser().resolve(strict=False)
    roots = set(scheduler_targets)
    for pid in rows:
        if pid <= 1 or pid in excluded:
            continue
        if _run_dir_from_process_environment(pid) == resolved_run_dir or (
            _process_cwd_is_run_workspace(pid, resolved_run_dir)
        ):
            roots.add(pid)
    targets = _gather_targets(root_pids=roots, ps_rows=rows, excluded=excluded)
    tokens = {pid: stable_tokens.get(pid, "") for pid in targets}
    for members in group_members.values():
        tokens.update({pid: token for pid, token in members.items() if token})
    return tokens, group_members


def _drain_late_run_processes(
    run_dirs: dict[str, Path],
    outcome: StopOutcome,
) -> dict[str, set[int]]:
    """Bound late-fork races until verified run-owned work is stably absent."""

    run_dirs = {run_id: path for run_id, path in run_dirs.items() if path.is_dir()}
    if not run_dirs:
        return {}
    known_tokens: dict[int, str] = {}
    owners: dict[int, set[str]] = defaultdict(set)
    term_seen_at: dict[int, int] = {}
    killed: set[int] = set()
    term_targets: set[int] = set()
    stable_empty = 0

    for scan in range(_LATE_DRAIN_MAX_SCANS):
        current_groups: dict[int, dict[int, str]] = {}
        for run_id, run_dir in run_dirs.items():
            tokens, groups = _discover_run_owned_processes(run_dir)
            known_tokens.update(tokens)
            for pid in tokens:
                owners[pid].add(run_id)
            current_groups.update(groups)

        live = _live_process_instances(known_tokens, known_tokens)
        if not live:
            stable_empty += 1
            if stable_empty >= _LATE_DRAIN_STABLE_EMPTY_SCANS:
                break
        else:
            stable_empty = 0
            outcome.descendant_pids = sorted(set(outcome.descendant_pids) | live)
            to_term = {pid for pid in live if pid not in term_seen_at}
            to_kill = {
                pid
                for pid in live
                if scan - term_seen_at.get(pid, scan) >= _LATE_DRAIN_TERM_SCANS
                and pid not in killed
            }
            if to_term:
                term_groups = {
                    pgid: members
                    for pgid, members in current_groups.items()
                    if set(members) & to_term
                }
                _signal_process_groups(
                    term_groups,
                    signal.SIGTERM,
                    outcome,
                    group_members=term_groups,
                )
                _signal_set(
                    to_term,
                    signal.SIGTERM,
                    outcome,
                    process_tokens=known_tokens,
                )
                term_targets.update(to_term)
                term_seen_at.update({pid: scan for pid in to_term})
            if to_kill:
                kill_groups = {
                    pgid: members
                    for pgid, members in current_groups.items()
                    if set(members) & to_kill
                }
                _signal_process_groups(
                    kill_groups,
                    signal.SIGKILL,
                    outcome,
                    group_members=kill_groups,
                )
                _signal_set(
                    to_kill,
                    signal.SIGKILL,
                    outcome,
                    process_tokens=known_tokens,
                )
                killed.update(to_kill)
        if scan + 1 < _LATE_DRAIN_MAX_SCANS:
            time.sleep(_LATE_DRAIN_SCAN_SECONDS)

    final_live = _live_process_instances(known_tokens, known_tokens)
    final_kill = final_live - killed
    if final_kill:
        _signal_set(
            final_kill,
            signal.SIGKILL,
            outcome,
            process_tokens=known_tokens,
        )
        killed.update(final_kill)
        time.sleep(_LATE_DRAIN_SCAN_SECONDS)
    if stable_empty < _LATE_DRAIN_STABLE_EMPTY_SCANS:
        stable_empty = 0
        for _verify in range(_LATE_DRAIN_STABLE_EMPTY_SCANS):
            for run_id, run_dir in run_dirs.items():
                tokens, _groups = _discover_run_owned_processes(run_dir)
                known_tokens.update(tokens)
                for pid in tokens:
                    owners[pid].add(run_id)
            final_live = _live_process_instances(known_tokens, known_tokens)
            if final_live:
                stable_empty = 0
                _signal_set(
                    final_live,
                    signal.SIGKILL,
                    outcome,
                    process_tokens=known_tokens,
                )
                killed.update(final_live)
            else:
                stable_empty += 1
            if stable_empty < _LATE_DRAIN_STABLE_EMPTY_SCANS:
                time.sleep(_LATE_DRAIN_SCAN_SECONDS)

    outcome.terminated_pids = sorted(
        set(outcome.terminated_pids) | (term_targets - killed - final_live)
    )
    outcome.remaining_pids = sorted(
        {pid for pid in outcome.remaining_pids if pid_is_alive(pid)} | final_live
    )
    remaining_by_run: dict[str, set[int]] = defaultdict(set)
    for pid in final_live:
        for run_id in owners.get(pid, ()):
            remaining_by_run[run_id].add(pid)
    return dict(remaining_by_run)


def gc_stale_entries(*, dry_run: bool = False) -> GcOutcome:
    """Remove every registry entry whose PID is no longer ours.

    Implements ``praxist stop --gc`` (#166). An entry is *stale* when:

    * the recorded PID has no live process (``pid_is_alive`` returns False), OR
    * the recorded PID is alive but its current command line does not start
      with the prefix written by ``praxist start`` (PID recycling).

    Stale rows are removed via :func:`registry.remove_entry`; live rows are
    kept untouched. With ``dry_run=True`` the would-be-removed ids are
    surfaced in ``removed_run_ids`` but no files are deleted.

    The implementation deliberately mirrors :func:`_validate_registry_root`
    (the TOCTOU guard ``praxist stop <run_id>`` already uses) so the "is this
    entry stale?" definition stays in one place.
    """
    ps_rows = read_ps_table()
    outcome = GcOutcome(dry_run=dry_run)

    for observed in list_entries():
        try:
            with entry_lock(observed.run_id):
                _gc_stale_entry(
                    run_id=observed.run_id,
                    ps_rows=ps_rows,
                    outcome=outcome,
                    dry_run=dry_run,
                )
        except (RegistryError, ValueError) as exc:
            outcome.warnings.append(f"run_id={observed.run_id}: registry changed before gc: {exc}")

    outcome.removed_run_ids.sort()
    outcome.kept_run_ids.sort()
    return outcome


def _gc_stale_entry(
    *,
    run_id: str,
    ps_rows: dict[int, tuple[int, str, str]],
    outcome: GcOutcome,
    dry_run: bool,
) -> None:
    """Classify and optionally remove one registry entry under its run lock."""
    entry = read_entry(run_id)
    if entry_is_local(entry) is False:
        outcome.kept_run_ids.append(entry.run_id)
        outcome.warnings.append(f"run_id={entry.run_id}: remote-host registry entry kept")
        return
    if entry_process_epoch_matches(entry) is False:
        outcome.removed_run_ids.append(entry.run_id)
        if not dry_run:
            remove_entry(entry.run_id)
        return
    identity = process_identity_matches(entry)
    if identity is False:
        outcome.removed_run_ids.append(entry.run_id)
        if not dry_run:
            remove_entry(entry.run_id)
        return
    if identity is True and pid_is_alive(entry.pid):
        outcome.kept_run_ids.append(entry.run_id)
        return
    live = ps_rows.get(entry.pid)
    if live is not None:
        _ppid, _etime, command = live
        if registry_command_matches(entry, command):
            outcome.kept_run_ids.append(entry.run_id)
            return
        # Live PID, but the command no longer matches: PID recycling.
    elif pid_is_alive(entry.pid):
        outcome.kept_run_ids.append(entry.run_id)
        outcome.warnings.append(
            f"run_id={entry.run_id}: pid {entry.pid} is alive but absent "
            f"from ps; keeping (cannot verify ownership)"
        )
        return

    if dry_run:
        outcome.removed_run_ids.append(entry.run_id)
        return
    try:
        if remove_entry(entry.run_id):
            outcome.removed_run_ids.append(entry.run_id)
        else:
            outcome.warnings.append(f"run_id={entry.run_id}: registry file already gone")
    except OSError as exc:
        outcome.warnings.append(f"run_id={entry.run_id}: could not remove registry file: {exc}")


def _write_gc_outcome_summary(outcome: GcOutcome) -> None:
    """Print a human-readable summary of an ``praxist stop --gc`` sweep.

    Stdout: one removed run_id per line so the output streams cleanly
    into shell pipelines (``praxist stop --gc | xargs ...``).
    Stderr: human narrative — what was removed, kept, and any warnings.
    """
    if outcome.dry_run:
        sys.stderr.write("praxist stop --gc (dry-run):\n")
    else:
        sys.stderr.write("praxist stop --gc:\n")
    if outcome.removed_run_ids:
        verb = "would remove" if outcome.dry_run else "removed"
        sys.stderr.write(f"  {verb}: {len(outcome.removed_run_ids)} entry(ies)\n")
        for run_id in outcome.removed_run_ids:
            sys.stderr.write(f"    - {run_id}\n")
    else:
        sys.stderr.write("  no stale entries found\n")
    if outcome.kept_run_ids:
        sys.stderr.write(f"  kept   : {len(outcome.kept_run_ids)} live entry(ies)\n")
    for warning in outcome.warnings:
        sys.stderr.write(f"  warning: {warning}\n")
    for run_id in outcome.removed_run_ids:
        sys.stdout.write(run_id + "\n")


def _validate_registry_root(
    entry: RegistryEntry,
    ps_rows: dict[int, tuple[int, str, str]],
    outcome: StopOutcome,
) -> int | None:
    """Return the PID to signal for ``entry``, or None when nothing to do.

    Implements the TOCTOU guard: a registry entry only counts as a
    live signalling target when (a) the PID exists in ``ps`` and
    (b) its current command line starts with the prefix recorded by
    ``praxist start``.  Mismatches add a warning to ``outcome``.
    """
    identity = process_identity_matches(entry)
    if identity is False:
        outcome.warnings.append(
            f"run_id={entry.run_id}: pid {entry.pid} start identity changed; skipping recycled PID"
        )
        outcome.remaining_pids = sorted(set(outcome.remaining_pids) | {entry.pid})
        return None
    if identity is True:
        return entry.pid if pid_is_alive(entry.pid) else None
    if entry.extra.get("process_start_token", "").strip():
        if pid_is_alive(entry.pid):
            outcome.warnings.append(
                f"run_id={entry.run_id}: pid {entry.pid} has a recorded start "
                "identity that cannot currently be verified; skipping it"
            )
            outcome.remaining_pids = sorted(set(outcome.remaining_pids) | {entry.pid})
        return None
    live = ps_rows.get(entry.pid)
    if live is None:
        if not pid_is_alive(entry.pid):
            return None
        outcome.warnings.append(
            f"run_id={entry.run_id}: pid {entry.pid} is alive but absent from ps; skipping"
        )
        outcome.remaining_pids = sorted(set(outcome.remaining_pids) | {entry.pid})
        return None
    _ppid, _etime, command = live
    if not registry_command_matches(entry, command):
        outcome.warnings.append(
            f"run_id={entry.run_id}: pid {entry.pid} cmdline does not match "
            f"recorded controller and run directory; skipping (TOCTOU guard)"
        )
        outcome.remaining_pids = sorted(set(outcome.remaining_pids) | {entry.pid})
        return None
    return entry.pid


def _validate_grace(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise StopError("grace_seconds must be a finite non-negative number")


def _gather_targets(
    *,
    root_pids: Iterable[int],
    ps_rows: dict[int, tuple[int, str, str]],
    excluded: set[int] | None = None,
) -> set[int]:
    """Walk descendants of ``root_pids`` using the ``ps`` (pid, ppid) table.

    Returns ``{roots} ∪ {all transitive children}``, minus any PIDs in
    ``excluded`` (the operator's own process tree).
    """
    excluded = excluded or set()
    children: dict[int, list[int]] = defaultdict(list)
    for pid, (ppid, _etime, _cmd) in ps_rows.items():
        children[ppid].append(pid)
    seen: set[int] = set()
    stack = [pid for pid in root_pids if pid not in excluded]
    while stack:
        pid = stack.pop()
        if (
            pid in seen or pid in excluded
        ):  # pragma: no cover - defensive: child-push filter already dedupes
            continue
        seen.add(pid)
        for child in children.get(pid, []):
            if child not in seen and child not in excluded:
                stack.append(child)
    return seen


def _signal_set(
    pids: Iterable[int],
    sig: int,
    outcome: StopOutcome,
    *,
    process_tokens: dict[int, str] | None = None,
    fallback_entries: dict[int, RegistryEntry] | None = None,
) -> None:
    """Send ``sig`` to each pid; record permission errors as warnings.

    Children are signalled before their parents (reverse PID sort reduces the
    "orphaned children re-parent to init then escape detection" race). SIGKILL
    targets are appended to ``outcome.killed_pids`` so the operator summary can
    distinguish graceful exits from forced ones.
    """
    sent: list[int] = []
    for pid in sorted(set(pids), reverse=True):
        process_handle = _open_process_handle(pid)
        try:
            if process_tokens is not None:
                state = _process_instance_state(pid, process_tokens)
                if state in {_PROCESS_GONE, _PROCESS_MISMATCH}:
                    continue
                if state == _PROCESS_UNKNOWN:
                    fallback = (fallback_entries or {}).get(pid)
                    verified = fallback is not None and _registry_fallback_matches(fallback)
                    if not verified:
                        if pid_is_alive(pid):
                            already_reported = pid in outcome.remaining_pids
                            outcome.remaining_pids = sorted(set(outcome.remaining_pids) | {pid})
                            if not already_reported:
                                outcome.warnings.append(
                                    f"cannot verify process identity for pid {pid}; "
                                    "leaving it running"
                                )
                        continue
            if process_handle is None:
                os.kill(pid, sig)
            else:
                _signal_process_handle(process_handle, sig)
            sent.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            outcome.warnings.append(f"cannot signal pid {pid}: {exc}")
        finally:
            if process_handle is not None:
                with contextlib.suppress(OSError):
                    os.close(process_handle)
    if sig == signal.SIGKILL:
        outcome.killed_pids = sorted(set(outcome.killed_pids) | set(sent))


def _signal_process_groups(
    pgids: Iterable[int],
    sig: int,
    outcome: StopOutcome,
    *,
    group_members: dict[int, dict[int, str]] | None = None,
) -> None:
    """Signal complete scheduler-owned sessions so late forks cannot escape stop."""

    for pgid in sorted(set(pgids), reverse=True):
        members = group_members.get(pgid) if group_members is not None else None
        if group_members is not None:
            safe, unknown = _process_group_signal_safety(pgid, members)
            for pid in sorted(unknown):
                already_reported = pid in outcome.remaining_pids
                outcome.remaining_pids = sorted(set(outcome.remaining_pids) | {pid})
                if not already_reported:
                    outcome.warnings.append(
                        f"cannot verify process identity for pid {pid} in process "
                        f"group {pgid}; skipping the group signal"
                    )
            if not safe or members is None:
                continue
            _signal_set(
                members,
                sig,
                outcome,
                process_tokens=members,
            )
            continue
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            outcome.warnings.append(f"cannot signal process group {pgid}: {exc}")


def _open_process_handle(pid: int) -> int | None:
    """Open a Linux pidfd so a later signal cannot hit a reused PID."""

    opener = getattr(os, "pidfd_open", None)
    if callable(opener):
        try:
            descriptor = opener(pid, 0)
            return descriptor if isinstance(descriptor, int) else None
        except OSError:
            return None
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        pidfd_open = libc.pidfd_open
        pidfd_open.argtypes = (ctypes.c_int, ctypes.c_uint)
        pidfd_open.restype = ctypes.c_int
        descriptor = int(pidfd_open(pid, 0))
        return descriptor if descriptor >= 0 else None
    except (AttributeError, OSError):
        return None


def _signal_process_handle(descriptor: int, sig: int) -> None:
    """Signal the exact Linux process instance referenced by a pidfd."""

    sender = getattr(signal, "pidfd_send_signal", None)
    if callable(sender):
        sender(descriptor, sig, None, 0)
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        pidfd_send_signal = libc.pidfd_send_signal
        pidfd_send_signal.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
        )
        pidfd_send_signal.restype = ctypes.c_int
        if pidfd_send_signal(descriptor, sig, None, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    except AttributeError as exc:  # pragma: no cover - paired with pidfd_open availability.
        raise OSError("pidfd signal delivery is unavailable") from exc


def _scheduler_owned_targets(run_dir: Path, ps_rows: dict[int, tuple[int, str, str]]) -> set[int]:
    """Include scheduler-owned process groups even after launcher re-parenting."""

    targets, _group_identities = _scheduler_owned_state(run_dir, ps_rows)
    return targets


def _scheduler_owned_state(
    run_dir: Path,
    ps_rows: dict[int, tuple[int, str, str]],
) -> tuple[set[int], dict[int, dict[int, str]]]:
    """Return one verified snapshot of scheduler targets and group members."""

    entries = _verified_scheduler_entries(run_dir)
    manifest_authority: dict[int, list[ProtectedEntry]] = defaultdict(list)
    for entry in entries:
        if entry.pgid > 1:
            manifest_authority[entry.pgid].append(entry)
    rpc_authority: dict[int, tuple[int, str]] = {}
    try:
        from praxist.plugins.workflow_stages.research_loop.backend.experiment_scheduler_client import (
            scheduler_active_process_groups,
        )

        rpc_authority = scheduler_active_process_groups(Path(run_dir))
    except Exception:  # noqa: BLE001 - legacy runs have no scheduler authority.
        pass
    pgids = set(manifest_authority) | set(rpc_authority)
    group_members: dict[int, dict[int, str]] = {pgid: {} for pgid in pgids}
    if pgids:
        for pid in ps_rows:
            try:
                pgid = os.getpgid(pid)
                if pgid in pgids:
                    group_members[pgid][pid] = process_start_token(pid)
            except (ProcessLookupError, PermissionError, OSError):
                continue

    verified_pgids: set[int] = set()
    for pgid in pgids:
        manifest_valid = any(
            entry.pid_start_time is not None
            and pid_is_alive(entry.pid)
            and _protected_entry_identity_matches(entry)
            and _pid_still_in_group(entry.pid, pgid)
            for entry in manifest_authority.get(pgid, ())
        )
        rpc_valid = False
        if authority := rpc_authority.get(pgid):
            pid, expected_token = authority
            rpc_valid = (
                bool(expected_token)
                and pid_is_alive(pid)
                and process_start_token(pid) == expected_token
                and _pid_still_in_group(pid, pgid)
            )
        if manifest_valid or rpc_valid:
            verified_pgids.add(pgid)

    verified_groups = {
        pgid: members
        for pgid, members in group_members.items()
        if pgid in verified_pgids and members
    }
    targets = {pid for members in verified_groups.values() for pid in members}
    return targets, verified_groups


def _protected_entry_identity_matches(entry: ProtectedEntry) -> bool:
    """Recheck a protected launcher without trusting a missing identity."""

    try:
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            _entry_process_identity_matches,
        )

        return entry.pid_start_time is not None and _entry_process_identity_matches(entry)
    except Exception:  # noqa: BLE001 - lifecycle safety is warning-first.
        return False


def _pid_still_in_group(pid: int, pgid: int) -> bool:
    try:
        return os.getpgid(pid) == pgid
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _record_unresolved_protected_groups(
    run_id: str,
    run_dir: Path,
    outcome: StopOutcome,
) -> None:
    """Expose live manifest groups that lack a verifiable launcher identity."""

    try:
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            _entry_process_identity_matches,
            _is_process_group_alive,
            list_all_protected,
        )

        entries = list_all_protected(run_dir=run_dir, prune_dead=False)
    except Exception:  # noqa: BLE001 - corrupt legacy manifests remain best-effort.
        return
    unresolved: set[int] = set()
    for entry in entries:
        if entry.pgid <= 1 or not _is_process_group_alive(entry.pgid):
            continue
        if (
            entry.pid_start_time is not None
            and pid_is_alive(entry.pid)
            and _entry_process_identity_matches(entry)
        ):
            continue
        unresolved.add(entry.pgid)
    if not unresolved:
        return
    if run_id not in outcome.failed_run_ids:
        outcome.failed_run_ids.append(run_id)
    message = (
        f"run_id={run_id}: protected process groups could not be identity-verified: "
        f"{sorted(unresolved)}"
    )
    if message not in outcome.warnings:
        outcome.warnings.append(message)


def _scheduler_owned_process_groups(run_dir: Path) -> set[int]:
    """Return only groups whose recorded leader identity is still verifiable."""

    ps_rows = read_ps_table()
    _targets, group_members = _scheduler_owned_state(run_dir, ps_rows)
    return set(group_members)


def _verified_scheduler_entries(run_dir: Path) -> list[ProtectedEntry]:
    """Reject stale/reused process groups before CLI stop sends signals."""

    try:
        from praxist.plugins.workflow_stages.research_loop.backend.protected_pids import (
            _entry_process_identity_matches,
            list_all_protected,
        )

        entries = list_all_protected(run_dir=run_dir, prune_dead=False)
    except Exception:  # noqa: BLE001 - legacy/corrupt manifests stay best-effort.
        return []
    verified: list[ProtectedEntry] = []
    for entry in entries:
        if entry.pid <= 1 or entry.pid_start_time is None:
            continue
        if not pid_is_alive(entry.pid):
            continue
        if not _entry_process_identity_matches(entry):
            continue
        if entry.pgid > 1:
            try:
                if os.getpgid(entry.pid) != entry.pgid:
                    continue
            except (ProcessLookupError, PermissionError, OSError):
                continue
        verified.append(entry)
    return verified


def _process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _capture_process_tokens(pids: Iterable[int]) -> dict[int, str]:
    """Snapshot process identities before the stop grace window begins."""

    return {pid: token for pid in set(pids) if (token := process_start_token(pid))}


def _read_identity_bound_ps_snapshot() -> tuple[
    dict[int, tuple[int, str, str]],
    dict[int, str],
]:
    """Bind strong start tokens to a stable process-table observation."""

    before_rows = read_ps_table()
    before_tokens = _capture_process_tokens(before_rows)
    after_rows = read_ps_table()
    after_tokens = _capture_process_tokens(after_rows)
    stable_tokens = {
        pid: token
        for pid, token in after_tokens.items()
        if token == before_tokens.get(pid)
        and pid in before_rows
        and before_rows[pid][0] == after_rows[pid][0]
        and before_rows[pid][2] == after_rows[pid][2]
    }
    return after_rows, stable_tokens


def _process_instance_state(pid: int, process_tokens: dict[int, str]) -> str:
    """Classify a captured process without treating probe failure as exit."""

    if not pid_is_alive(pid):
        return _PROCESS_GONE
    expected = process_tokens.get(pid, "")
    if not expected:
        return _PROCESS_UNKNOWN
    observed = process_start_token(pid)
    if not observed:
        return _PROCESS_UNKNOWN
    return _PROCESS_MATCH if observed == expected else _PROCESS_MISMATCH


def _registry_fallback_matches(entry: RegistryEntry) -> bool:
    """Use the recorded controller command when its token is unavailable."""

    if entry.extra.get("process_start_token", "").strip():
        return False
    if process_identity_matches(entry) is False:
        return False
    live = read_ps_table().get(entry.pid)
    if live is None:
        return False
    _ppid, _etime, command = live
    return registry_command_matches(entry, command)


def _live_process_instances(
    pids: Iterable[int],
    process_tokens: dict[int, str],
) -> set[int]:
    return {
        pid
        for pid in set(pids)
        if _process_instance_state(pid, process_tokens) in {_PROCESS_MATCH, _PROCESS_UNKNOWN}
    }


def _process_group_instance_alive(
    pgid: int,
    members: dict[int, str] | None,
) -> bool:
    """Return whether a scheduler group retains a captured process instance."""

    if not members:
        return False
    for pid, token in members.items():
        if _process_instance_state(pid, {pid: token}) != _PROCESS_MATCH:
            continue
        try:
            if os.getpgid(pid) == pgid:
                return _process_group_alive(pgid)
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return False


def _process_group_signal_safety(
    pgid: int,
    members: dict[int, str] | None,
) -> tuple[bool, set[int]]:
    """Allow a group signal only when every current member has a strong match."""

    if not members:
        return False, set()
    current: set[int] = set()
    for pid in read_ps_table():
        try:
            if pid_is_alive(pid) and os.getpgid(pid) == pgid:
                current.add(pid)
        except (ProcessLookupError, PermissionError, OSError):
            continue
    if not current:
        return False, set()

    unknown: set[int] = set()
    all_match = True
    for pid in current:
        token = members.get(pid, "")
        state = _process_instance_state(pid, {pid: token})
        if state == _PROCESS_UNKNOWN:
            unknown.add(pid)
        if state != _PROCESS_MATCH:
            all_match = False
    return all_match and _process_group_alive(pgid), unknown


def _await_exit(
    pids: Iterable[int],
    grace_seconds: float,
    outcome: StopOutcome,
    *,
    process_tokens: dict[int, str] | None = None,
) -> None:
    """Poll ``pids`` until they exit or ``grace_seconds`` elapses."""
    targets = set(pids)
    tokens = process_tokens or {}
    if not targets:
        return
    deadline = time.monotonic() + max(0.0, grace_seconds)
    interval = max(0.05, min(0.25, grace_seconds / 10 if grace_seconds > 0 else 0.05))
    while time.monotonic() < deadline:
        targets = _live_process_instances(targets, tokens)
        if not targets:
            break
        time.sleep(interval)
    outcome.terminated_pids = sorted(pid for pid in set(pids) if pid not in targets)


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO 8601 string (seconds resolution)."""
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def _write_outcome_summary(outcome: StopOutcome) -> None:
    """Print a human-readable summary of the stop outcome on stderr.

    Data (the matched PID list) goes to stdout so machine readers stay
    well behaved; the operator narrative goes to stderr per the CLI
    output discipline.
    """
    label = outcome.run_id or "ALL"
    sys.stderr.write(f"praxist stop ({label}):\n")
    if outcome.matched_pids:
        sys.stderr.write(f"  matched roots    : {outcome.matched_pids}\n")
        sys.stderr.write(f"  descendants      : {outcome.descendant_pids}\n")
        if not outcome.dry_run:
            sys.stderr.write(f"  exited on TERM   : {outcome.terminated_pids}\n")
            sys.stderr.write(f"  forced via KILL  : {outcome.killed_pids}\n")
            if outcome.remaining_pids:
                sys.stderr.write(f"  still alive      : {outcome.remaining_pids}\n")
        else:
            sys.stderr.write("  (dry-run — no signals sent)\n")
    elif outcome.failed_run_ids:
        sys.stderr.write("  no runs could be stopped safely\n")
    else:
        sys.stderr.write("  no matching Praxist runs found\n")
    if outcome.failed_run_ids:
        sys.stderr.write(f"  admission failed : {outcome.failed_run_ids}\n")
    for warning in outcome.warnings:
        sys.stderr.write(f"  warning          : {warning}\n")
    if outcome.matched_pids:
        sys.stdout.write(
            json.dumps(sorted(set(outcome.matched_pids) | set(outcome.descendant_pids))) + "\n"
        )


__all__ = [
    "DEFAULT_GRACE_SECONDS",
    "SCOPE_PS",
    "SCOPE_REGISTRY",
    "SCOPE_UNION",
    "GcOutcome",
    "StopError",
    "StopOutcome",
    "cmd_stop",
    "gc_stale_entries",
    "register",
    "stop_all",
    "stop_run",
]
