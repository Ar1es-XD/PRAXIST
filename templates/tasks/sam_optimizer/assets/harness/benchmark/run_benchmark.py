"""
SAM optimizer internal benchmark runner — single-GPU, in-process.

For multi-peer Praxist runs, peers should call
``evaluations/pareto_tiered/run.py``. That canonical evaluator calls this
internal runner after applying the task's tiered gate protocol. This runner
still owns the low-level mechanics:

  1. GPU is auto-picked from the per-GPU process governor (least-loaded
     slot count among visible GPUs); a slot is acquired before training
     and released on exit. With `GPU_GOVERNOR_MAX_PER_GPU=1` (recommended
     for SAM-style 1-train-per-GPU workloads), no two peers will ever
     share a GPU even if they don't coordinate explicitly.

  2. Compute scales gradually via `--tier {T1,T2,T3}` so weak variants
     are filtered cheaply before getting full T3 compute. A clear
     decision rule is documented in the task README/description.

Tier defaults (one compatible accelerator, bf16, batch 256):
  T1  ~3 min   3 seeds × cifar100 × 10 epochs              # filter degenerates
  T2  ~10 min  3 seeds × {cifar10, cifar100} × 20 epochs   # survivor gate
  T3  ~41 min  5 seeds × {cifar10, cifar100, tiny-imagenet} × 20 epochs

Standalone use (no governor): just run as today; CUDA_VISIBLE_DEVICES
is honored if set, otherwise GPU 0.

Usage:
    # Internal evaluator call under Praxist:
    GPU_GOVERNOR_DIR=<run>/process_governor GPU_GOVERNOR_MAX_PER_GPU=1 \
      python assets/harness/benchmark/run_benchmark.py \
        --optimizer custom --variant-path my_variant.py \
        --tier T1 --output-dir <run>/<peer>/

    # Standalone (single-user, baseline):
    CUDA_VISIBLE_DEVICES=3 python assets/harness/benchmark/run_benchmark.py \
        --optimizer sam --tier T3 --output-dir results/baseline
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# TIER PROTOCOL — single source of truth.
# Peers select via --tier; the task project's tiered_eval block mirrors this.
# ---------------------------------------------------------------------------

TIER_DEFAULTS: Dict[str, Dict] = {
    "T1": {
        "seeds": [42, 43, 44],
        "datasets": ["cifar100"],
        "epochs": 10,
        "batch_size": 256,
        "purpose": "filter degenerate variants (NaN, divergence, far-below-baseline)",
        "expected_minutes": 3,
    },
    "T2": {
        "seeds": [42, 43, 44],
        "datasets": ["cifar10", "cifar100"],
        "epochs": 20,
        "batch_size": 256,
        "purpose": "survivor gate before T3; sufficient to reject weak variants",
        "expected_minutes": 10,
    },
    "T3": {
        "seeds": [42, 43, 44, 45, 46],
        "datasets": ["cifar10", "cifar100", "tiny-imagenet"],
        "epochs": 20,
        "batch_size": 256,
        "purpose": "full evaluation across 3 datasets / 5 seeds",
        "expected_minutes": 41,
    },
}


# ---------------------------------------------------------------------------
# GPU governor integration — kept torch-free so we can pin
# CUDA_VISIBLE_DEVICES BEFORE torch sees the device list.
# ---------------------------------------------------------------------------

CVD_UNSET = object()       # sentinel: env var not set at all
CVD_FORCE_CPU = object()   # sentinel: env var set to "" (CUDA disabled)
CVD_OPAQUE = object()      # sentinel: env var contains UUIDs / non-int — honor verbatim


def _list_visible_gpus():
    """Return either a list of integer GPU ids visible to this process, or
    one of the sentinel values:
      CVD_UNSET     — no env hint; we should probe via nvidia-smi
      CVD_FORCE_CPU — env was set to empty string ("disable CUDA"); honor it
      CVD_OPAQUE    — env contains UUIDs; let torch deal with it, no auto-pick

    The unset / "" distinction matters: empty CUDA_VISIBLE_DEVICES is a
    documented CUDA convention for "force CPU"; we must not silently override.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")  # None if unset
    if cvd is None:
        # No env hint — probe.
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "-L"], text=True, timeout=15,
            )
        except Exception:
            return CVD_UNSET  # caller falls back to GPU 0 with a warning
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        return list(range(n)) if n > 0 else [0]

    cvd_stripped = cvd.strip()
    if cvd_stripped == "":
        return CVD_FORCE_CPU

    ids: List[int] = []
    for tok in cvd_stripped.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            # Non-integer entry (UUID / MIG slice). Honor verbatim.
            return CVD_OPAQUE
    return ids if ids else CVD_OPAQUE


def _try_import_governor():
    """Try to import praxist.plugins.workflow_stages.research_loop.backend.gpu_governor. Returns module or None."""
    # Walk up from this file to find the repo root, then add it to sys.path.
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "praxist" / "plugins" / "workflow_stages" / "research_loop" / "backend" / "gpu_governor.py").exists():
            if str(ancestor) not in sys.path:
                sys.path.insert(0, str(ancestor))
            break
    try:
        from praxist.plugins.workflow_stages.research_loop.backend import gpu_governor  # type: ignore
        return gpu_governor
    except Exception:
        return None


def _try_import_execution_guards():
    """Try to import Praxist resource guard helpers without breaking standalone eval."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "praxist" / "core" / "execution_guards.py").exists():
            if str(ancestor) not in sys.path:
                sys.path.insert(0, str(ancestor))
            break
    try:
        from praxist.core.execution_guards import record_budgeted_action_from_env  # type: ignore
        return record_budgeted_action_from_env
    except Exception:
        return None


def _try_import_dataset_resolver():
    """Import the task-local dataset resolver without requiring a package."""
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        candidate = ancestor / "dataset_metadata" / "resolve_dataset.py"
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("sam_dataset_resolver", candidate)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def _resolve_data_dir(value: str | None) -> str:
    resolver = _try_import_dataset_resolver()
    if resolver is None:
        return str(value or "./data")
    return str(resolver.resolve_dataset_root(value))


def _record_eval_runner_usage_from_env(
    *,
    status: str,
    wall_clock_seconds: float,
    metadata: Dict | None = None,
) -> None:
    """Best-effort eval-runner budget usage accounting."""
    record_budgeted_action_from_env = _try_import_execution_guards()
    if record_budgeted_action_from_env is None:
        return
    try:
        record_budgeted_action_from_env(
            action_type="eval_runner",
            actor_ref="evaluation_runner:sam_optimizer",
            actual_usage={"wall_clock_seconds": max(0.0, float(wall_clock_seconds))},
            expected_units=("wall_clock_seconds",),
            status=status,
            reason="eval_runner_wall_clock_usage",
            metadata=metadata or {},
        )
    except Exception:
        return


def _resolve_governor_dir() -> str:
    """Return the governor dir to use.

    Priority order:
      1. `GPU_GOVERNOR_DIR` env, if set AND non-empty.
      2. Pointer file `/tmp/praxist_active_governor_uid<uid>` written
         by the orchestrator on startup. Defends against the case where
         peers are spawned via a CLI/SDK that clears env vars (observed
         2026-04-30: agent CLI emits GPU_GOVERNOR_DIR= literally empty,
         which split governance across two dirs and broke max_per_gpu).
      3. Per-uid `/tmp/sam_eval_default_governor_uid<uid>` for purely
         standalone runs with no orchestrator.
    """
    explicit = os.environ.get("GPU_GOVERNOR_DIR")
    if explicit:
        return explicit

    try:
        uid = os.getuid()
    except AttributeError:
        uid = 0
    pointer = Path(f"/tmp/praxist_active_governor_uid{uid}")
    if pointer.exists():
        try:
            content = pointer.read_text().strip()
            if content:
                return content
        except OSError:
            pass

    return f"/tmp/sam_eval_default_governor_uid{uid}"


# ---------------------------------------------------------------------------
# Wall-clock training timeout (R1 fix).
# A peer's variant can hang on NaN-loss-loop, CUDA driver wedge, or CPU
# starvation. Without an explicit timeout, the slot is held until the
# orchestrator's per-generation cap fires (24h) — the whole cohort stalls.
# ---------------------------------------------------------------------------

# Module-level handle so main()'s finally can disarm the watchdog before
# release_slot runs (preventing SIGALRM from firing mid-release_slot).
_ACTIVE_WATCHDOG_CANCEL = lambda: None

# R10-NEW1 fix: capture the ORIGINAL visible-GPU pool before
# `pick_and_acquire_gpu` pins CUDA_VISIBLE_DEVICES to a single id.
# Otherwise `_reclaim_stale_slots` would only see the one pinned GPU
# and the cleanup never reaches sibling GPUs' stale slots.
_ORIGINAL_VISIBLE_GPUS: List[int] = []


class TrainingBudgetExceeded(RuntimeError):
    """Raised when wall-clock budget is exceeded mid-training."""


def _check_budget(start_time: float, budget_seconds: float):
    """Raise if wall-clock budget exceeded. Called between epochs/datasets.

    NOTE: This only catches budget overruns at sentinel points. Hangs
    INSIDE train_one_epoch (CUDA wedge, DataLoader stall, NCCL deadlock)
    are caught by the SIGALRM watchdog installed by `_install_alarm_watchdog`,
    which fires regardless of GIL/control-flow state.
    """
    elapsed = time.monotonic() - start_time
    if elapsed > budget_seconds:
        raise TrainingBudgetExceeded(
            f"Wall-clock budget exceeded: {elapsed:.1f}s > {budget_seconds:.1f}s"
        )


# Progress-aware grace: when budget fires mid-training but ≤10% epochs remain,
# the work-saved/work-lost ratio strongly favors letting the seed finish
# rather than discarding 90% of completed compute. The 2× SIGALRM watchdog
# is the hard backstop for actual wedges.
def _check_budget_with_grace(
    start_time: float,
    budget_seconds: float,
    progress_frac: float,
    grace_threshold: float = 0.9,
    log_fn=print,
):
    elapsed = time.monotonic() - start_time
    if elapsed <= budget_seconds:
        return
    if progress_frac >= grace_threshold:
        log_fn(
            f"[BUDGET-GRACE] elapsed {elapsed:.0f}s > budget "
            f"{budget_seconds:.0f}s but progress {progress_frac:.0%} "
            f">= {grace_threshold:.0%} — allowing natural finish."
        )
        return
    raise TrainingBudgetExceeded(
        f"Wall-clock budget exceeded: {elapsed:.1f}s > {budget_seconds:.1f}s "
        f"(progress {progress_frac:.0%} < grace {grace_threshold:.0%})"
    )


def _install_alarm_watchdog(budget_seconds: float, log_fn=print):
    """Install a SIGALRM-based hard wall-clock watchdog.

    Hard-exit semantics (R3-N1+N2 fix):
    The handler calls `os._exit(2)` instead of raising. Why:
      - Raise-based approach is one-shot: signal.alarm(N) is a single
        timer; once it fires, no second alarm without explicit re-arm.
        A peer that catches the first TrainingBudgetExceeded and limps
        on to the next seed has no watchdog protection at all.
      - Even worse, raising from mid-CUDA-call leaves the CUDA context
        in an unknown state. Subsequent torch.cuda.empty_cache() in the
        finally block can DEADLOCK on a wedged context — and the alarm
        is already disarmed, so no second escape.
      - os._exit(2) is the correct "I'm done, unrecoverable" signal:
        process dies, orchestrator sees non-zero exit, governor's
        dead-pid prune frees the slot on the next sweep.

    On clean budget overrun the inner `_check_budget` raises first
    (which the per-seed except handles); the watchdog only fires for
    HANGS that bypass `_check_budget`. In that situation, dying is
    correct.

    Returns a no-op cancel function if SIGALRM isn't available.
    """
    try:
        import signal
    except ImportError:
        return lambda: None
    if not hasattr(signal, "SIGALRM"):
        return lambda: None  # Windows: no SIGALRM

    # Disarm any pre-existing alarm first (defensive — N9): we don't
    # know what the parent / orchestrator may have set; we can't
    # restore its remaining time so we explicitly clobber.
    signal.alarm(0)

    def _handler(signum, frame):
        # Print first, then exit. The orchestrator's peer-output capture
        # will see this line; from the operator's perspective it's a
        # diagnostic for "GPU N held a wedged training peer".
        # R9-2 fix: flush stdout BEFORE diag write so any block-buffered
        # progress lines (we're in a pipe, so stdout is fully buffered)
        # are not lost when os._exit skips Python's normal shutdown.
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.write(
                f"\n[WATCHDOG] SIGALRM fired: budget {budget_seconds:.0f}s "
                f"exceeded; peer is HARD-EXITing (likely CUDA wedge or "
                f"DataLoader stall). Slot will be freed on next "
                f"governor sweep via dead-pid prune.\n"
            )
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(2)

    prev_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(budget_seconds) + 1)
    log_fn(f"[WATCHDOG] SIGALRM set for {int(budget_seconds)+1}s (hard-exit on fire)")

    def cancel():
        try:
            signal.alarm(0)  # disarm
            signal.signal(signal.SIGALRM, prev_handler)
        except Exception:
            pass

    return cancel


# ---------------------------------------------------------------------------
# CPU load gate (R1 fix).
# Five peers × 8 DataLoader workers each + tiny-imagenet JPEG decode can
# saturate even a 168-core host. Before each dataset transition, check
# load average and pause briefly if over a soft cap.
# ---------------------------------------------------------------------------

def _wait_for_cpu_headroom(soft_load_per_core: float = 1.5,
                            max_wait_s: float = 120.0,
                            poll_interval_s: float = 10.0,
                            log_fn=print) -> None:
    """If load avg per core exceeds soft cap, sleep and re-check, up to max_wait_s.

    Designed to absorb transient saturation (e.g. all 5 peers spawning DataLoader
    workers at once on a dataset transition) without hard-failing.
    """
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    deadline = time.monotonic() + max_wait_s  # R4-N3: monotonic for NTP safety
    warned = False
    while True:
        try:
            load1, _, _ = os.getloadavg()
        except (OSError, AttributeError):
            return  # Platform doesn't support; skip the gate.
        per_core = load1 / cpu_count
        if per_core <= soft_load_per_core:
            return
        if time.monotonic() >= deadline:
            log_fn(f"[CPU] load {load1:.1f}/{cpu_count}={per_core:.2f} per core "
                   f"still over {soft_load_per_core} after {max_wait_s}s; proceeding anyway.")
            return
        if not warned:
            log_fn(f"[CPU] load {load1:.1f}/{cpu_count}={per_core:.2f} per core "
                   f"> {soft_load_per_core}; pausing {poll_interval_s}s.")
            warned = True
        time.sleep(poll_interval_s)


def pick_and_acquire_gpu(tag: str) -> Tuple[int, "Callable[[], None]"]:
    """Pick a GPU + acquire a governor slot. Returns (gpu_id, release_fn).

    Stampede-resistant flow (avoids the "all peers see counts=0 → all pick
    GPU 0 → 4 of 5 peers serialize" pathology):

      1. Detect visible GPU pool from CUDA_VISIBLE_DEVICES / nvidia-smi.
      2. Sort pool by current slot-count (governor.list_slots), tiebreak by id.
      3. Try `acquire_slot(g, blocking=False)` on each in that order. The
         first non-full GPU that the flock-protected acquire succeeds on
         becomes our pick. Concurrent peers racing for GPU 0 lose at flock,
         retry on GPU 1, etc. — they SPREAD without re-coordination.
      4. If every GPU is full, sleep with jittered backoff and re-snapshot.
      5. After 3 sweeps with no progress, fall back to a blocking acquire
         on the (currently) least-loaded GPU as a last resort.

    Pre-conditions:
      - This function MUST be called before torch is imported, because it
        sets CUDA_VISIBLE_DEVICES. Once torch is imported the device
        list is cached and re-pinning has no effect.
    """
    import random

    visible_or_sentinel = _list_visible_gpus()

    # R10-NEW1: stash the original (pre-pin) visible pool so the
    # cleanup path can sweep all GPUs even after CUDA_VISIBLE_DEVICES
    # has been narrowed to one id.
    if isinstance(visible_or_sentinel, list):
        global _ORIGINAL_VISIBLE_GPUS
        _ORIGINAL_VISIBLE_GPUS = list(visible_or_sentinel)

    if visible_or_sentinel is CVD_FORCE_CPU:
        print("[GPU] CUDA_VISIBLE_DEVICES='' — honoring 'force CPU' convention.")
        return -1, (lambda: None)

    if visible_or_sentinel is CVD_OPAQUE:
        # UUIDs / MIG slices — torch will resolve. Don't claim a physical id.
        print("[GPU] CUDA_VISIBLE_DEVICES contains opaque ids (UUID/MIG); "
              "honoring verbatim, no governor pick.")
        return -1, (lambda: None)

    if visible_or_sentinel is CVD_UNSET:
        # nvidia-smi probe failed; assume single-GPU layout.
        print("[GPU] WARN: nvidia-smi probe failed; falling back to GPU 0.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return 0, (lambda: None)

    visible: List[int] = visible_or_sentinel  # type: ignore[assignment]
    governor = _try_import_governor()
    governor_available = governor is not None

    if not governor_available:
        gpu_id = visible[0]
        if len(visible) > 1:
            print(f"[GPU] WARN: governor module unavailable; standalone mode. "
                  f"Picking GPU {gpu_id} from visible={visible}. "
                  f"To coordinate with other processes, ensure Praxist is "
                  f"on PYTHONPATH or pin CUDA_VISIBLE_DEVICES manually.")
        else:
            print(f"[GPU] standalone (no governor). Using GPU {gpu_id}.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return gpu_id, (lambda: None)

    # Governor available — set GPU_GOVERNOR_DIR if not already set, so the
    # governor module's _governor_dir() resolution succeeds.
    if not os.environ.get("GPU_GOVERNOR_DIR"):
        os.environ["GPU_GOVERNOR_DIR"] = _resolve_governor_dir()

    pid = os.getpid()
    max_per_gpu_env = os.environ.get("GPU_GOVERNOR_MAX_PER_GPU")
    max_per_gpu = int(max_per_gpu_env) if max_per_gpu_env else 1
    # R8-Issue12 fix: thread peer identity from env into governor manifest
    # so operators can attribute "which peer holds GPU N" without grepping.
    # The orchestrator sets PEER_ID and GENERATION_ID per-peer; if running
    # standalone, both default to "" and the manifest still works.
    peer_id_str = (
        f"{os.environ.get('GENERATION_ID','')}_{os.environ.get('PEER_ID','')}"
    ).strip("_")

    # R2-N8 fix: derive expected_seconds from --tier directly instead of
    # relying on a never-set env var. Falls back to 0 (no budget hint) if
    # tier is omitted (manual / standalone use).
    expected_seconds = 0
    if "--tier" in sys.argv:
        try:
            i = sys.argv.index("--tier")
            tier_arg = sys.argv[i + 1]
            if tier_arg in TIER_DEFAULTS:
                expected_seconds = int(TIER_DEFAULTS[tier_arg]["expected_minutes"] * 60 * 1.5)
        except (IndexError, ValueError):
            pass

    def _try_sweep(visible_pool):
        """Snapshot counts, sort by (count, id), try non-blocking acquire on each.
        Returns gpu_id on success, None if every visible GPU is full this round.

        R2-N3 fix: per-GPU exception isolation. A single transient
        list_slots failure (e.g. NFS hiccup on lock file) must not collapse
        the whole snapshot — that previously sent every peer to GPU 0.
        """
        counts: Dict[int, int] = {}
        for g in visible_pool:
            try:
                counts[g] = len(governor.list_slots(g))
            except Exception as e:
                print(f"[GPU] list_slots({g}) failed: {e}; treating as full.")
                counts[g] = 10**9  # treat as full this sweep; retry next sweep
        order = sorted(visible_pool, key=lambda g: (counts[g], g))
        for g in order:
            if counts[g] >= max_per_gpu:
                continue  # known-full
            try:
                ok = governor.acquire_slot(
                    g, pid=pid, peer=peer_id_str, tag=tag,
                    max_per_gpu=max_per_gpu, blocking=False,
                    expected_seconds=expected_seconds,
                )
            except Exception as e:
                print(f"[GPU] governor.acquire_slot raised on GPU {g}: {e}")
                continue
            if ok:
                return g, counts
        return None, counts

    # R5-N3: removed dead `isinstance(counts_or_msg, str)` branch — `_try_sweep`
    # now absorbs per-GPU list_slots failures locally (counts[g] = 10**9), so
    # there's no longer a "all of list_slots crashed" return value.
    sweep_attempts = 0
    while True:
        result, counts_or_msg = _try_sweep(visible)
        if isinstance(result, int):
            gpu_id = result
            print(f"[GPU] governor sweep: visible={visible}  "
                  f"counts={counts_or_msg}  picked={gpu_id}  "
                  f"max_per_gpu={max_per_gpu}  attempts={sweep_attempts}")
            break
        sweep_attempts += 1
        if sweep_attempts >= 3:
            # Every GPU is full across multiple sweeps — last resort blocking.
            counts = counts_or_msg
            fallback_gpu = sorted(visible, key=lambda g: (counts[g], g))[0]
            print(f"[GPU] all GPUs full after {sweep_attempts} sweeps. "
                  f"Blocking on GPU {fallback_gpu}.")
            ok = governor.acquire_slot(
                fallback_gpu, pid=pid, peer=peer_id_str, tag=tag,
                max_per_gpu=max_per_gpu, blocking=True,
                # 60-min wait — safe because watchdog at budget*2+300s
                # is the outer killswitch. Allows queueing through one
                # full T3-cell when 5 peers × 3 datasets = 15 jobs > 8 GPUs.
                timeout_seconds=3600,
            )
            if not ok:
                raise RuntimeError(
                    f"gpu_governor: timed out blocking on GPU {fallback_gpu}"
                )
            gpu_id = fallback_gpu
            break
        # Jittered backoff before resnapshot — prevents lockstep retries.
        time.sleep(2.0 + random.random() * 3.0)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def release():
        try:
            governor.release_slot(gpu_id, pid=pid)
        except Exception as e:
            print(f"[GPU] WARN: release_slot failed: {e}")

    return gpu_id, release


# ---------------------------------------------------------------------------
# Argument parsing — happens BEFORE torch is imported.
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Parse benchmark runner arguments without importing torch first."""

    parser = argparse.ArgumentParser(
        description="SAM benchmark runner — single GPU, in-process, tiered.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--optimizer", default="sam",
                        choices=["sgd", "adam", "sam", "asam", "custom"])
    parser.add_argument("--variant-path", default="")

    # Tier (overrides --datasets/--seeds/--epochs/--batch-size if set).
    parser.add_argument("--tier", choices=list(TIER_DEFAULTS.keys()), default=None,
                        help="Pre-defined eval tier. Overrides --datasets/--seeds/--epochs.")

    # Manual overrides (used when --tier is omitted).
    parser.add_argument("--datasets", default="cifar10,cifar100,tiny-imagenet")
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument(
        "--data-dir",
        default="",
        help=(
            "Dataset root. Defaults to PRAXIST_SAM_DATA_DIR, SAM_DATA_DIR, "
            "PRAXIST_DATA_DIR, PRAXIST_DATASETS_DIR/sam_optimizer, PRAXIST_DATA_ROOT/sam_optimizer, "
            "then ./data/sam_optimizer."
        ),
    )
    parser.add_argument(
        "--print-data-dir",
        action="store_true",
        help="Resolve and print the dataset root without importing torch or starting training.",
    )
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--num-workers", type=int, default=4)
    # Per-peer dataset-level parallelism (parent-dispatch ThreadPool).
    # Seeds within a dataset are kept SERIAL (they're statistical
    # replicates — parallelizing them gains nothing). Datasets are
    # independent tasks — parallelizing them genuinely doubles/triples
    # throughput on a multi-GPU host.
    #
    # 'auto' = number of datasets in this run.
    # T1 (1 dataset)  → 1 (no fan-out)
    # T2 (2 datasets) → 2 datasets in parallel
    # T3 (3 datasets) → 3 datasets in parallel
    #
    # Combined with cohort=5 and gpu_governor max_per_gpu=1, T3
    # phase uses 5×3=15 desired GPU slots, governor throttles to 8.
    # All 8 GPUs busy until last dataset of last peer finishes.
    parser.add_argument(
        "--parallel-datasets", default="auto",
        help="Per-peer parallel dataset count. 'auto' = number of datasets. "
             "Set to 1 for the legacy fully-serial path.",
    )
    args = parser.parse_args(argv)

    if args.tier is not None:
        cfg = TIER_DEFAULTS[args.tier]
        # CRITICAL FIX: tier should fill DEFAULTS only, NOT clobber
        # explicit user/parent overrides. Otherwise the parent dispatcher
        # invoking children with `--tier T3 --datasets cifar10` would
        # have the child re-expand to all 3 tier datasets — 3× duplicate
        # work per child, blowing past the per-cell budget.
        # We detect "user passed it explicitly" by checking sys.argv tokens.
        argv_tokens = (argv if argv is not None else sys.argv[1:])
        explicit = set(argv_tokens)
        if "--seeds" not in explicit:
            args.seeds = ",".join(str(s) for s in cfg["seeds"])
        if "--datasets" not in explicit:
            args.datasets = ",".join(cfg["datasets"])
        if "--epochs" not in explicit:
            args.epochs = cfg["epochs"]
        if "--batch-size" not in explicit:
            args.batch_size = cfg["batch_size"]

    args.data_dir = _resolve_data_dir(args.data_dir)
    return args


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _enforce_pareto_axis_completeness(final: Dict) -> None:
    """R2 C1 fix: when ANY of the 4 Pareto axes (mean_test_accuracy /
    wall_time_seconds_total / mean_train_test_gap / sharpness_top_eigen)
    has `mean: None` AND tier=T3, force `promotion_eligible=False` and
    emit a loud warning. Otherwise the variant would pass the runner's
    "promotable" check, the peer would publish a finding with None on
    one axis, and the leaderboard would silently exclude it.

    Better behavior: don't claim PROMOTABLE for incomplete axis data.
    The fix surfaces the missing-axis problem at the runner-log layer
    where it's diagnosable, rather than at the leaderboard layer where
    the variant just disappears.

    The public tiered wrapper derives `wall_time_seconds_total` from the
    per-cell train times, so this internal runner checks the other 3 axes here.
    """
    if final.get("tier") != "T3":
        return
    # R3 Issue 1 fix: parent-dispatched single-dataset children write
    # their own multi_benchmark.json with intentionally degenerate
    # cross-dataset aggregates (only 1 dataset = no cross-dataset
    # signal). The parent re-aggregates correctly; we should not lie
    # in the child's JSON. Detect via `parent_dispatch=False` (this
    # flag is set only by the parent path) AND single-dataset run.
    is_parent_dispatch = bool(final.get("parent_dispatch"))
    is_single_dataset_child = (
        not is_parent_dispatch and len(final.get("datasets") or []) == 1
    )
    if is_single_dataset_child:
        return
    # R3 Issue 1 fix (cont.): if cifar100 isn't even in this run's
    # dataset list, sharpness can't be computed by design — skip the
    # sharpness check rather than warn spuriously.
    has_c100 = "cifar100" in (final.get("datasets") or [])
    missing = []
    for k in ("mean_test_accuracy", "mean_train_test_gap"):
        v = final.get(k, {})
        if not isinstance(v, dict) or v.get("mean") is None:
            missing.append(k)
    if has_c100:
        v = final.get("sharpness_top_eigen", {})
        if not isinstance(v, dict) or v.get("mean") is None:
            missing.append("sharpness_top_eigen")
    # R3 Issue 3 fix: still record pareto_axis_incomplete even when
    # promotion_eligible is already False — operators debugging
    # compound failures need both diagnostics.
    if missing:
        final["pareto_axis_incomplete"] = missing
    # Loud-warning + force-non-promotable only when the run was
    # otherwise eligible (so we don't compound an already-failed run).
    if missing and final.get("promotion_eligible"):
        final["promotion_eligible"] = False
        print(
            f"\n[PARETO INCOMPLETE] T3 ran but {len(missing)} Pareto "
            f"axis/axes have no value: {missing}. "
            f"Forcing promotion_eligible=False — peer cannot publish "
            f"this run as a leaderboard-eligible result. Most common "
            f"cause: sharpness probe was budget-skipped on every "
            f"seed (raise --epochs budget or reduce probe iterations).\n"
        )


def _attach_orthogonal_axes(final: Dict) -> None:
    """Compute cross-dataset aggregates + surface sharpness to top level.

    Mutates `final` in place. Called from BOTH the parallel-dispatch and
    the sequential single-dataset finalize paths so the multi_benchmark.json
    schema is consistent regardless of how the runner was invoked
    (R-fix Issue 1).

    Always emits `mean_test_accuracy` / `mean_train_test_gap` /
    `sharpness_top_eigen` keys (with `mean: null` when no data, instead
    of omitting the key) so peers reading `bench["sharpness_top_eigen"]
    ["mean"]` per the prompt example never KeyError (R-fix Issue 2).
    """
    import numpy as _np
    per_dataset = final.get("per_dataset") or {}

    # Cross-dataset accuracy (single orthogonal "Accuracy" axis).
    _ok_acc_means = []
    _ok_gap_means = []
    for ds, agg in per_dataset.items():
        if not isinstance(agg, dict):
            continue
        if agg.get("status") not in ("ok", "partial"):
            continue
        ta = agg.get("test_accuracy", {})
        if isinstance(ta, dict) and isinstance(ta.get("mean"), (int, float)):
            _ok_acc_means.append(ta["mean"])
        gp = agg.get("train_test_gap", {})
        if isinstance(gp, dict) and isinstance(gp.get("mean"), (int, float)):
            _ok_gap_means.append(gp["mean"])
    final["mean_test_accuracy"] = {
        "mean": float(_np.mean(_ok_acc_means)) if _ok_acc_means else None,
        "std_across_datasets": (
            float(_np.std(_ok_acc_means)) if len(_ok_acc_means) > 1 else None
        ),
        "n_datasets": len(_ok_acc_means),
    }
    final["mean_train_test_gap"] = {
        "mean": float(_np.mean(_ok_gap_means)) if _ok_gap_means else None,
        "std_across_datasets": (
            float(_np.std(_ok_gap_means)) if len(_ok_gap_means) > 1 else None
        ),
        "n_datasets": len(_ok_gap_means),
    }

    # Sharpness lives only in cifar100's per-dataset block. Surface to
    # top level. Always emit the key (with `mean: null` if missing or
    # all-seed-failed) so peers can read it without KeyError.
    _c100 = per_dataset.get("cifar100", {})
    _sharp = _c100.get("sharpness_top_eigen") if isinstance(_c100, dict) else None
    if isinstance(_sharp, dict) and isinstance(_sharp.get("mean"), (int, float)):
        final["sharpness_top_eigen"] = _sharp
    else:
        final["sharpness_top_eigen"] = {
            "mean": None,
            "n_seeds": 0,
            "note": (
                "sharpness probe not available — either cifar100 not in "
                "datasets, tier != T3, or all-seed probe failure"
            ),
        }


def aggregate_seeds(per_seed: List[Dict], total_seeds: int) -> Dict:
    """Aggregate per-seed results. Tolerates partial failure: if some seeds
    crashed, returns a partial summary with `partial: true` so the peer can
    react instead of seeing silent NaN.
    """
    import numpy as np
    ok = [r for r in per_seed if r.get("status", "ok") == "ok"]
    failed = [r for r in per_seed if r.get("status") == "failed"]
    if not ok:
        return {
            "status": "all_failed",
            "num_seeds_ok": 0, "num_seeds_failed": len(failed),
            "num_seeds_total": total_seeds,
            "errors": [r.get("error", "") for r in failed][:5],
        }
    accs = [r["best_test_acc"] for r in ok]
    gaps = [r["train_test_gap"] for r in ok]
    times = [r["total_time_seconds"] for r in ok]
    out = {
        "status": "partial" if failed else "ok",
        "num_seeds_ok": len(ok),
        "num_seeds_failed": len(failed),
        "num_seeds_total": total_seeds,
        "test_accuracy": {
            "mean": float(np.mean(accs)), "std": float(np.std(accs)),
            "min": float(np.min(accs)), "max": float(np.max(accs)),
        },
        "train_test_gap": {"mean": float(np.mean(gaps)), "std": float(np.std(gaps))},
        "train_time_seconds": {"mean": float(np.mean(times))},
    }
    # v2026-05-01: aggregate sharpness_top_eigen across seeds (only
    # populated on T3 + cifar100; null on other datasets/tiers).
    # R2 M1 fix: ALWAYS emit the key (with mean=None when no probe
    # data). Per-dataset schema parity with the top-level helper that
    # also always emits.
    sharps = [r.get("sharpness_top_eigen") for r in ok
              if r.get("sharpness_top_eigen") is not None]
    if sharps:
        out["sharpness_top_eigen"] = {
            "mean": float(np.mean(sharps)),
            "std": float(np.std(sharps)),
            "min": float(np.min(sharps)),
            "max": float(np.max(sharps)),
            "n_seeds": len(sharps),
        }
    else:
        out["sharpness_top_eigen"] = {
            "mean": None, "n_seeds": 0,
            "note": "no probe data (skipped or all-seed failure)",
        }
    if failed:
        out["errors"] = [r.get("error", "") for r in failed][:5]
    return out


# ---------------------------------------------------------------------------
# Main: pick GPU first, then import torch, then iterate.
# ---------------------------------------------------------------------------

def main():
    """Run the internal benchmark CLI and write benchmark result artifacts."""

    args = parse_args()
    if getattr(args, "print_data_dir", False):
        print(args.data_dir)
        return
    runner_t0 = time.monotonic()
    runner_status = "succeeded"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    try:
        return _main_after_parse(args, datasets)
    except BaseException:
        runner_status = "failed"
        raise
    finally:
        _record_eval_runner_usage_from_env(
            status=runner_status,
            wall_clock_seconds=time.monotonic() - runner_t0,
            metadata={
                "optimizer": args.optimizer,
                "tier": args.tier or "",
                "datasets": datasets,
                "parallel_datasets": str(args.parallel_datasets),
            },
        )


def _apply_scheduler_assignment(env: Dict[str, str]) -> str:
    """Preserve Praxist physical UUID/MIG assignment across child launches."""

    assigned = env.get("PRAXIST_ASSIGNED_GPU_UUIDS", "").strip()
    if not assigned:
        return ""
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        current = env.get(key, "").strip()
        if current and current != assigned:
            raise RuntimeError(
                f"accelerator binding mismatch: {key}={current!r}, Praxist assignment={assigned!r}"
            )
        env[key] = assigned
    return assigned


def _main_after_parse(args, datasets: List[str]):
    # Resolve --parallel-datasets ('auto' = num_datasets).
    if args.parallel_datasets == "auto":
        n_par = len(datasets)
    else:
        try:
            n_par = max(1, int(args.parallel_datasets))
        except (TypeError, ValueError):
            n_par = 1
    assigned_uuids = [
        item.strip() for item in os.environ.get("PRAXIST_ASSIGNED_GPU_UUIDS", "").split(",") if item.strip()
    ]
    if assigned_uuids:
        n_par = min(n_par, len(assigned_uuids))

    # Multi-dataset PARENT dispatch: spawn one subprocess per dataset,
    # each runs in single-dataset/single-GPU mode. Parent doesn't
    # acquire a governor slot — children do. Governor throttles
    # cross-peer across all 8 GPUs.
    if n_par > 1 and len(datasets) > 1:
        return _dispatch_parallel_datasets(args, datasets, max_workers=n_par)

    # Single-dataset (or --parallel-datasets=1) WORKER mode: keep the
    # existing in-process pipeline — one GPU, serial seeds. This is
    # the leaf invocation that the parent dispatches into.
    tag = f"{args.optimizer}_{args.tier or 'manual'}"
    assigned = _apply_scheduler_assignment(os.environ)
    if assigned:
        gpu_id, release_slot = -1, (lambda: None)
    else:
        gpu_id, release_slot = pick_and_acquire_gpu(tag)
    try:
        _main_after_acquire(args, gpu_id)
    finally:
        try:
            _ACTIVE_WATCHDOG_CANCEL()
        except Exception:
            pass
        release_slot()
        try:
            _reclaim_stale_slots()
        except Exception as e:
            print(f"[GPU] WARN: stale-slot reclamation failed: {e}")


def _dispatch_parallel_datasets(args, datasets: List[str], *, max_workers: int):
    """Parent dispatch: spawn one subprocess per dataset in a thread pool.
    Each subprocess does pick_and_acquire_gpu + serial seeds for its
    single dataset, releasing the GPU on exit.

    Output layout under args.output_dir:
        <fname_prefix>_cells/<dataset>/   <- per-dataset subprocess writes here
            <fname_prefix>_<dataset>_seed<N>.json
            <fname_prefix>_<dataset>_benchmark.json
            <fname_prefix>_multi_benchmark.json    (single-dataset summary)
        <fname_prefix>_<dataset>_seed<N>.json      <- copied up by parent
        <fname_prefix>_<dataset>_benchmark.json    <- copied up by parent
        <fname_prefix>_multi_benchmark.json         <- aggregated by parent
    """
    import json as _json
    import shutil as _shutil
    import subprocess as _subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Filename prefix mirrors _main_after_acquire's fname_prefix logic.
    if args.optimizer == "custom" and args.variant_path:
        variant_slug = Path(args.variant_path).stem.replace(".", "_")
        base = f"custom_{variant_slug}"
    else:
        base = args.optimizer
    fname_prefix = f"{base}_{args.tier}" if args.tier else base

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_root = output_dir / f"{fname_prefix}_cells"
    cells_root.mkdir(parents=True, exist_ok=True)

    print(f"=== SAM benchmark — parent dispatch ===")
    print(f"  optimizer    : {args.optimizer}{(' (' + args.variant_path + ')') if args.variant_path else ''}")
    print(f"  tier         : {args.tier or 'manual'}")
    print(f"  datasets     : {datasets}  (parallel = {max_workers} subprocs)")
    print(f"  seeds        : {args.seeds}")
    print(f"  governor caps cross-peer GPU concurrency")
    print()

    overall_t0 = time.monotonic()

    def _run_one_dataset(ds: str) -> Dict:
        cell_dir = cells_root / ds
        cell_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--optimizer", args.optimizer,
            "--datasets", ds,
            "--seeds", args.seeds,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--rho", str(args.rho),
            "--data-dir", args.data_dir,
            "--output-dir", str(cell_dir),
            "--num-workers", str(args.num_workers),
            "--parallel-datasets", "1",   # children must NOT re-dispatch
        ]
        if args.tier:
            cmd.extend(["--tier", args.tier])
        if args.variant_path:
            cmd.extend(["--variant-path", args.variant_path])
        if args.bf16:
            cmd.append("--bf16")
        else:
            cmd.append("--no-bf16")
        # Under the central scheduler every descendant must inherit the exact
        # assigned UUID mask. Standalone legacy runs may still auto-pick.
        env = dict(os.environ)
        if not _apply_scheduler_assignment(env):
            env.pop("CUDA_VISIBLE_DEVICES", None)
        log_path = cell_dir / "subprocess.log"
        with open(log_path, "w") as logf:
            rc = _subprocess.call(cmd, stdout=logf, stderr=_subprocess.STDOUT, env=env)
        result = {"dataset": ds, "returncode": rc, "log": str(log_path)}
        if rc != 0:
            result["status"] = "subprocess_failed"
            return result
        # Parse the child's multi_benchmark.json (single-dataset).
        child_multi = cell_dir / f"{fname_prefix}_multi_benchmark.json"
        if not child_multi.exists():
            result["status"] = "no_output"
            return result
        with open(child_multi) as f:
            result["data"] = _json.load(f)
        result["status"] = "ok"
        # Copy the per-seed JSONs and benchmark JSON up to parent dir.
        for src in cell_dir.glob(f"{fname_prefix}_{ds}_seed*.json"):
            _shutil.copy2(src, output_dir / src.name)
        bench = cell_dir / f"{fname_prefix}_{ds}_benchmark.json"
        if bench.exists():
            _shutil.copy2(bench, output_dir / bench.name)
        return result

    cell_results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one_dataset, ds): ds for ds in datasets}
        for fut in as_completed(futs):
            ds = futs[fut]
            try:
                cell_results[ds] = fut.result()
            except Exception as e:
                print(f"[DISPATCH] dataset {ds} thread raised: {e}")
                cell_results[ds] = {"dataset": ds, "status": "exception", "error": str(e)}
            r = cell_results[ds]
            if r.get("status") == "ok":
                d = r["data"]
                agg = d.get("per_dataset", {}).get(ds, {})
                ta = agg.get("test_accuracy", {})
                if "mean" in ta:
                    print(f"  [{ds}] OK: mean_acc={ta['mean']:.4f} ± {ta.get('std', 0):.4f} "
                          f"(elapsed {(time.monotonic()-overall_t0)/60:.1f}min)")
                else:
                    print(f"  [{ds}] {agg.get('status','?')}: no aggregated metric")
            else:
                print(f"  [{ds}] FAILED ({r.get('status')}): see {r.get('log','?')}")

    # Aggregate into top-level multi_benchmark.json.
    final = {
        "optimizer": args.optimizer, "variant_path": args.variant_path,
        "tier": args.tier, "datasets": datasets,
        "seeds": [int(s) for s in args.seeds.split(",") if s.strip()],
        "epochs": args.epochs, "batch_size": args.batch_size,
        "bf16": args.bf16,
        "physical_gpu": "multi (parent dispatch)",
        "per_dataset": {},
        "promotion_eligible": False,
        "total_wall_clock_seconds": time.monotonic() - overall_t0,
        "parent_dispatch": True,
        "parallel_datasets": len(datasets),
    }
    every_ds_has_ok = True
    for ds in datasets:
        r = cell_results.get(ds, {})
        if r.get("status") == "ok":
            agg = r["data"].get("per_dataset", {}).get(ds, {})
            final["per_dataset"][ds] = agg
            if agg.get("status") not in ("ok", "partial") or agg.get("num_seeds_ok", 0) == 0:
                every_ds_has_ok = False
        else:
            final["per_dataset"][ds] = {"status": r.get("status", "unknown"),
                                         "num_seeds_ok": 0}
            every_ds_has_ok = False

    final["promotion_eligible"] = bool(args.tier == "T3" and every_ds_has_ok)

    # v2026-05-01: cross-dataset orthogonal Pareto axes (mean accuracy /
    # mean gap / sharpness). Helper attaches keys consistently in BOTH
    # parallel-dispatch and sequential paths.
    _attach_orthogonal_axes(final)
    # R2 C1 fix: any axis with mean=None → promotion_eligible=False
    # + loud warning (instead of silent leaderboard exclusion).
    _enforce_pareto_axis_completeness(final)

    if args.tier:
        target = TIER_DEFAULTS[args.tier]["expected_minutes"]
        status_tag = "PROMOTABLE" if final["promotion_eligible"] else "non-promotable"
        print(f"\n=== PARENT TOTAL wall-clock: {final['total_wall_clock_seconds']/60:.1f} min "
              f"(tier {args.tier} target ~{target} min, status={status_tag}) ===")
    else:
        print(f"\n=== PARENT TOTAL wall-clock: {final['total_wall_clock_seconds']/60:.1f} min ===")

    if args.tier in ("T1", "T2"):
        print(
            f"\n[TIER {args.tier}] These results are for INTERNAL gating only.\n"
            f"DO NOT call share_finding with primary metrics from this run.\n"
            f"Escalate to T3 if the gate passes; only T3 results are promotable.\n"
        )

    # R3-Issue2: atomic-rename so wait_for_file readers see the file
    # only when fully written.
    multi_path = output_dir / f"{fname_prefix}_multi_benchmark.json"
    multi_tmp = multi_path.with_suffix(multi_path.suffix + ".tmp")
    import json as _json2
    with open(multi_tmp, "w") as f:
        _json2.dump(final, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(multi_tmp, multi_path)
    print(f"Multi-dataset summary: {multi_path}")


def _reclaim_stale_slots():
    """Sweep every ORIGINAL-visible GPU's slot file, triggering dead-pid prune.

    Called by main()'s finally on clean exit. Uses `_ORIGINAL_VISIBLE_GPUS`
    captured before CUDA_VISIBLE_DEVICES was narrowed to the pinned id, so
    stale slots on sibling GPUs (e.g. from a SIGTERM-killed peer that
    wasn't this peer) get pruned too. Re-reading the env here would only
    find the one pinned GPU.
    """
    governor = _try_import_governor()
    if governor is None or not _ORIGINAL_VISIBLE_GPUS:
        return
    for g in _ORIGINAL_VISIBLE_GPUS:
        try:
            governor.list_slots(g)  # side-effect: prunes dead/zombie pids
        except Exception:
            pass


def _main_after_acquire(args, gpu_id: int):
    """Body of main() after the governor slot has been acquired."""
    # Now safe to import torch (it will see only the pinned GPU as cuda:0).
    import numpy as np  # noqa: F401
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Import core helpers from baseline/train.py (sibling directory).
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "baseline"))
    from train import (  # noqa: E402
        SAM, ASAM, get_dataset, get_resnet18,
        train_one_epoch, evaluate,
    )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        if os.environ.get("PRAXIST_ASSIGNED_GPU_UUIDS", "").strip():
            raise RuntimeError(
                "Praxist assigned a GPU, but the benchmark runtime cannot access CUDA; "
                "refusing to publish CPU results under the GPU protocol"
            )
        print("WARN: running on CPU (no GPU visible)")

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print(f"=== SAM benchmark (single-GPU, in-process) ===")
    print(f"  optimizer    : {args.optimizer}{(' (' + args.variant_path + ')') if args.variant_path else ''}")
    print(f"  tier         : {args.tier or 'manual'}")
    print(f"  datasets     : {datasets}")
    print(f"  seeds        : {seeds}")
    print(f"  epochs/ds    : {args.epochs}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  bf16         : {args.bf16}")
    print(f"  device       : {device}  (physical GPU {gpu_id})")
    print(f"  output       : {output_dir}")
    print(f"  data         : {args.data_dir}")
    print()

    # ---- inner: build optimizer (closure over args) ----
    def build_optimizer(model):
        if args.optimizer == "sgd":
            return optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                             weight_decay=args.weight_decay), False
        if args.optimizer == "adam":
            return optim.Adam(model.parameters(), lr=args.lr * 0.01,
                              weight_decay=args.weight_decay), False
        if args.optimizer == "sam":
            return SAM(model.parameters(), optim.SGD, rho=args.rho, lr=args.lr,
                       momentum=0.9, weight_decay=args.weight_decay), True
        if args.optimizer == "asam":
            return ASAM(model.parameters(), optim.SGD, rho=args.rho, lr=args.lr,
                        momentum=0.9, weight_decay=args.weight_decay), True
        if args.optimizer == "custom":
            if not args.variant_path:
                raise ValueError("--variant-path required when --optimizer custom")
            mod = _custom_variant_module  # loaded once before seed loop (R4-N5)
            opt = mod.create_optimizer(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay, rho=args.rho)
            return opt, getattr(mod, "USE_SAM_STEPS", False)
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # R4-N5 fix: load the custom variant module exactly once. Module-level
    # work (imports, side effects) runs once, not per-seed × per-dataset.
    # If the import fails, fail-fast with a clean error before any GPU work.
    _custom_variant_module = None
    if args.optimizer == "custom":
        if not args.variant_path:
            raise ValueError("--variant-path required when --optimizer custom")
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom_opt", args.variant_path)
        _custom_variant_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(_custom_variant_module)
        except Exception as e:
            print(f"[FAIL] Could not load --variant-path {args.variant_path}: "
                  f"{type(e).__name__}: {e}")
            # R4-N3 fix: write a STUB multi_benchmark.json (atomic-rename)
            # in addition to load_failure.json. This way peers waiting on
            # the canonical multi_benchmark.json filename via wait_for_file
            # return promptly with `promotion_eligible=False` instead of
            # waiting the full 75-min timeout for a file that will never come.
            output_dir.mkdir(parents=True, exist_ok=True)
            error_str = f"{type(e).__name__}: {e}"
            # Filename prefix mirrors the success path's logic.
            if args.optimizer == "custom" and args.variant_path:
                _vslug = Path(args.variant_path).stem.replace(".", "_")
                _base = f"custom_{_vslug}"
            else:
                _base = args.optimizer
            _fpre = f"{_base}_{args.tier}" if args.tier else _base

            stub_payload = {
                "status": "load_failure",
                "variant_path": args.variant_path,
                "error": error_str,
                "tier": args.tier,
                "promotion_eligible": False,
                "per_dataset": {},
            }
            for fname in ("load_failure.json", f"{_fpre}_multi_benchmark.json"):
                p = output_dir / fname
                tmp = p.with_suffix(p.suffix + ".tmp")
                with open(tmp, "w") as f:
                    json.dump(stub_payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, p)
            return  # release_slot still runs in main()'s finally

    def _scheduler_target(optimizer, use_sam):
        """Return whichever optimizer should be passed to the LR scheduler.

        SAM/ASAM hold the real (momentum-bearing) base optimizer at
        `.base_optimizer`; the cosine scheduler must drive that, not the
        SAM wrapper. Custom variants are encouraged but not required to
        expose `.base_optimizer` — fall back to the wrapper itself if the
        attribute is missing (the variant's first/second_step will then
        read from the wrapper's own param_groups, which is also fine).
        """
        if not use_sam:
            return optimizer
        return getattr(optimizer, "base_optimizer", optimizer)

    # R4-N6 fix: when --optimizer custom, embed the variant filename stem
    # in the prefix so two peers running different custom variants in a
    # shared --output-dir don't overwrite each other's per-seed JSONs.
    if args.optimizer == "custom" and args.variant_path:
        variant_slug = Path(args.variant_path).stem.replace(".", "_")
        base = f"custom_{variant_slug}"
    else:
        base = args.optimizer
    fname_prefix = f"{base}_{args.tier}" if args.tier else base

    # Wall-clock budget: 1.5× the tier's expected minutes (or 90 min for manual mode).
    if args.tier:
        budget_seconds = TIER_DEFAULTS[args.tier]["expected_minutes"] * 60 * 1.5
    else:
        budget_seconds = 90 * 60

    # R2-N11 fix: monotonic clock for elapsed math (resilient to NTP step).
    overall_t0 = time.monotonic()
    # R2-N2 + R8-Issue2 fix: install SIGALRM watchdog as outer deadline.
    # The grace must accommodate cold-cache first-epoch slowdowns (NFS,
    # tiny-imagenet 100k JPEG initial decode) which can take 3-10x
    # steady-state. Original `budget+60s` was too tight: a slow first
    # epoch could blow past it even on a healthy variant. New grace =
    # 2× budget, capped at +5 min minimum, well below the orchestrator's
    # 24h per-generation cap.
    watchdog_seconds = max(budget_seconds * 2, budget_seconds + 300)
    global _ACTIVE_WATCHDOG_CANCEL
    _ACTIVE_WATCHDOG_CANCEL = _install_alarm_watchdog(watchdog_seconds)

    final = {
        "optimizer": args.optimizer, "variant_path": args.variant_path,
        "tier": args.tier, "datasets": datasets, "seeds": seeds,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "bf16": args.bf16,
        "physical_gpu": gpu_id if gpu_id >= 0 else None,
        "physical_gpu_note": (
            "uuid_or_cpu_pinned" if gpu_id < 0 else None
        ),
        "wall_clock_budget_seconds": budget_seconds,
        "per_dataset": {},
        "promotion_eligible": False,  # Only set True for completed T3 (or manual full).
    }

    budget_exceeded = False
    # Hold previous dataset refs across the loop so we can null them before
    # building the next dataset (R3-N10 fix).
    train_set = None
    test_set = None
    for ds in datasets:
        if budget_exceeded:
            break
        print(f"--- dataset {ds} ---")

        # R3-N10 fix: drop previous dataset before allocating the next one.
        # Across (cifar10 → cifar100 → tiny-imagenet), tiny-imagenet's
        # ImageFolder cache is several GB; holding cifar100's pickled
        # arrays in memory across the transition wastes RAM.
        train_set = None
        test_set = None
        import gc
        gc.collect()

        ds_t0 = time.monotonic()
        train_set, test_set, num_classes, input_size = get_dataset(ds, args.data_dir)

        per_seed: List[Dict] = []
        for seed in seeds:
            try:
                _check_budget(overall_t0, budget_seconds)
            except TrainingBudgetExceeded as e:
                print(f"[BUDGET] {e}; skipping remaining seeds for {ds}.")
                budget_exceeded = True
                break

            # R3-N7 fix: CPU load gate is checked HERE, just before we fork
            # DataLoader workers — the actual moment of the storm. R1's
            # placement before `get_dataset` was too early.
            _wait_for_cpu_headroom()

            random.seed(seed); np.random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            try:
                train_loader = DataLoader(
                    train_set, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
                    persistent_workers=(args.num_workers > 0), drop_last=False,
                )
                test_loader = DataLoader(
                    test_set, batch_size=max(args.batch_size, 256), shuffle=False,
                    num_workers=args.num_workers, pin_memory=(args.num_workers > 0),
                    persistent_workers=(args.num_workers > 0),
                )
                model = get_resnet18(num_classes=num_classes, input_size=input_size).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer, use_sam = build_optimizer(model)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    _scheduler_target(optimizer, use_sam),
                    T_max=args.epochs,
                )

                t0 = time.monotonic()
                epoch_log = []
                best_acc = 0.0
                for ep in range(args.epochs):
                    train_loss, train_acc = train_one_epoch(
                        model, train_loader, optimizer, criterion, device,
                        use_sam=use_sam, bf16=args.bf16,
                    )
                    test_loss, test_acc = evaluate(model, test_loader, criterion, device, bf16=args.bf16)
                    scheduler.step()
                    epoch_log.append({
                        "epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc,
                        "lr": scheduler.get_last_lr()[0],
                    })
                    if test_acc > best_acc:
                        best_acc = test_acc
                    # R6-N4 fix: catch BOTH NaN and ±Inf. Previous check
                    # `loss == loss` only catches NaN (because NaN != NaN);
                    # `Inf == Inf` is True, so a loss exploding to ±Inf
                    # without first NaN-ing was undetected and burned the
                    # full epoch budget. math.isfinite handles both.
                    import math
                    if not (math.isfinite(train_loss) and math.isfinite(test_loss)):
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {ep} "
                            f"(train={train_loss}, test={test_loss})"
                        )
                    # Per-epoch budget check, with progress-aware grace:
                    # if ≥90% of epochs in this seed are done, allow finish.
                    progress_frac = (ep + 1) / max(args.epochs, 1)
                    _check_budget_with_grace(
                        overall_t0, budget_seconds, progress_frac,
                    )
                train_time = time.monotonic() - t0

                # v2026-05-01: Sharpness probe — top-eigenvalue of the
                # loss Hessian via power iteration. SAM's whole point is
                # flatness; it must be on the Pareto frontier. Only run
                # on T3 + cifar100 (representative; flatness is a property
                # of the model, not dataset-bound). Cost is ~3-10s per
                # seed for ResNet-18 (single batch, 20 power iterations,
                # create_graph=True for second-order autograd).
                #
                # R-fix Issue 3 + R2 M4: budget-check BEFORE probe with
                # a 120s safety margin (was 60s; raised after R2 noted
                # the unverified guess + power-iteration with
                # create_graph=True over 20 iters of ResNet-18 on
                # 256-batch can reach 30-90s under contended-GPU load).
                # If the seed used most of the budget, skip the probe
                # so the per-seed JSON write below has time to complete
                # before any per-epoch budget check could raise.
                sharpness_top_eigen = None
                if args.tier == "T3" and ds == "cifar100":
                    _remaining = budget_seconds - (time.monotonic() - overall_t0)
                    if _remaining < 120:
                        print(
                            f"[SHARPNESS] {ds} seed={seed}: skipping "
                            f"probe (only {_remaining:.0f}s budget left, "
                            f"need ≥120s safety margin)"
                        )
                    else:
                        try:
                            from sharpness_probes import hessian_top_eigenvalue
                            sharpness_top_eigen = hessian_top_eigenvalue(
                                model, criterion, test_loader, device,
                                n_iterations=20,
                            )
                        except Exception as e:
                            print(
                                f"[SHARPNESS] {ds} seed={seed}: "
                                f"{type(e).__name__}: {e}"
                            )
                            sharpness_top_eigen = None

                summary = {
                    "status": "ok",
                    "best_test_acc": best_acc,
                    "final_test_acc": epoch_log[-1]["test_acc"],
                    "final_train_acc": epoch_log[-1]["train_acc"],
                    "train_test_gap": epoch_log[-1]["train_acc"] - epoch_log[-1]["test_acc"],
                    "total_time_seconds": train_time,
                    "seed": seed, "optimizer": args.optimizer, "dataset": ds,
                    "tier": args.tier,
                    "sharpness_top_eigen": sharpness_top_eigen,
                }
            except TrainingBudgetExceeded as e:
                print(f"[BUDGET] mid-train: {e}")
                summary = {"status": "failed", "error": f"budget_exceeded: {e}",
                           "seed": seed, "optimizer": args.optimizer, "dataset": ds,
                           "tier": args.tier}
                budget_exceeded = True
            except (KeyboardInterrupt, SystemExit):
                # Re-raise: user-initiated or watchdog-initiated exit.
                raise
            except Exception as e:
                # R3-N8 fix: catch ANY exception (including custom variant
                # ImportError/AttributeError/SyntaxError, MemoryError, OOM, etc.)
                # so the seed counts as "failed" and remaining seeds still run.
                print(f"[FAIL] seed {seed} on {ds}: {type(e).__name__}: {e}")
                summary = {"status": "failed", "error": f"{type(e).__name__}: {e}",
                           "seed": seed, "optimizer": args.optimizer, "dataset": ds,
                           "tier": args.tier}
            finally:
                # R2-N1 fix: `del locals()[name]` is a no-op in functions
                # because locals() returns a snapshot dict. The previous
                # seed's tensors must be released by direct reassignment
                # so torch.cuda.empty_cache() can actually free them
                # before the next seed's allocations. Without this, T3
                # tiny-imagenet OOMs around seed 4–5 from fragmentation.
                train_loader = None
                test_loader = None
                model = None
                optimizer = None
                scheduler = None
                import gc
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            per_seed.append(summary)
            # R3-Issue2: atomic-rename for per-seed JSON too.
            _seed_path = output_dir / f"{fname_prefix}_{ds}_seed{seed}.json"
            _seed_tmp = _seed_path.with_suffix(_seed_path.suffix + ".tmp")
            with open(_seed_tmp, "w") as f:
                json.dump({
                    "args": {**vars(args), "seed": seed, "dataset": ds},
                    "summary": summary,
                    "epochs": epoch_log if summary.get("status") == "ok" else [],
                }, f, indent=2)
                f.flush(); os.fsync(f.fileno())
            os.replace(_seed_tmp, _seed_path)

            if summary.get("status") == "ok":
                print(f"  [{ds}] seed {seed}: best_acc={summary['best_test_acc']:.4f}  "
                      f"gap={summary['train_test_gap']:+.4f}  "
                      f"({summary['total_time_seconds']:.1f}s, total elapsed "
                      f"{(time.monotonic()-overall_t0)/60:.1f}min)")

            if budget_exceeded:
                break

        agg = aggregate_seeds(per_seed, total_seeds=len(seeds))
        # R3-Issue2: atomic-rename for per-dataset benchmark JSON.
        _bench_path = output_dir / f"{fname_prefix}_{ds}_benchmark.json"
        _bench_tmp = _bench_path.with_suffix(_bench_path.suffix + ".tmp")
        with open(_bench_tmp, "w") as f:
            json.dump({"optimizer": args.optimizer, "dataset": ds,
                       "tier": args.tier,
                       "per_seed": per_seed, "aggregated": agg}, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(_bench_tmp, _bench_path)
        if agg.get("status") == "all_failed":
            print(f"  [{ds}] ALL FAILED ({len(per_seed)} seeds)\n")
        else:
            ta = agg["test_accuracy"]
            print(f"  [{ds}] aggregated [{agg['status']}]: "
                  f"{ta['mean']:.4f} ± {ta['std']:.4f}  "
                  f"(n={agg['num_seeds_ok']}/{agg['num_seeds_total']}, "
                  f"dataset wall {(time.monotonic()-ds_t0)/60:.1f}min)\n")
        final["per_dataset"][ds] = agg

    final["total_wall_clock_seconds"] = time.monotonic() - overall_t0
    final["budget_exceeded"] = budget_exceeded

    # Promotion-eligibility flag: only set True if tier=T3 fully completed
    # without budget exceedance and every dataset has at least one ok seed.
    # The frontier / peer prompt should refuse to call share_finding with
    # primary metrics from a non-promotable run.
    completed_all_datasets = (len(final["per_dataset"]) == len(datasets)) and not budget_exceeded
    every_ds_has_ok = all(
        agg.get("status") in ("ok", "partial") and agg.get("num_seeds_ok", 0) > 0
        for agg in final["per_dataset"].values()
    )
    final["promotion_eligible"] = bool(
        args.tier == "T3" and completed_all_datasets and every_ds_has_ok
    )

    # v2026-05-01: same orthogonal-axes attach as parallel-dispatch path
    # so the multi_benchmark.json schema is consistent regardless of
    # invocation mode (single-dataset / sequential / parallel).
    _attach_orthogonal_axes(final)
    _enforce_pareto_axis_completeness(final)

    if args.tier:
        target = TIER_DEFAULTS[args.tier]["expected_minutes"]
        status_tag = "BUDGET-EXCEEDED" if budget_exceeded else (
            "PROMOTABLE" if final["promotion_eligible"] else "non-promotable"
        )
        print(f"=== TOTAL wall-clock: {final['total_wall_clock_seconds']/60:.1f} min "
              f"(tier {args.tier} target ~{target} min, status={status_tag}) ===")
    else:
        print(f"=== TOTAL wall-clock: {final['total_wall_clock_seconds']/60:.1f} min ===")

    if args.tier in ("T1", "T2"):
        print(
            f"\n[TIER {args.tier}] These results are for INTERNAL gating only.\n"
            f"DO NOT call share_finding with primary metrics from this run.\n"
            f"Escalate to T3 if the gate passes; only T3 results are promotable.\n"
        )

    # R3-Issue2 fix: atomic-rename publish so wait_for_file readers
    # never observe a partial json.dump-in-progress. Without this, a
    # consumer using `contains_text="\"promotion_eligible\""` could
    # match the early-written field while the rest of the file is
    # still being streamed, causing a truncated read.
    multi_path = output_dir / f"{fname_prefix}_multi_benchmark.json"
    multi_tmp = multi_path.with_suffix(multi_path.suffix + ".tmp")
    with open(multi_tmp, "w") as f:
        json.dump(final, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(multi_tmp, multi_path)
    print(f"Multi-dataset summary: {multi_path}")


if __name__ == "__main__":
    main()
