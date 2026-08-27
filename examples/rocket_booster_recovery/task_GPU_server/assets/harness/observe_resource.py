#!/usr/bin/env python3
"""Observe an unchanged child command over its complete process lifetime.

This initialization-only helper records timestamped host, process-tree, and
NVIDIA telemetry.  It does not alter the child command or its arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _proc_tree(root_pid: int) -> list[int]:
    parent_by_pid: dict[int, int] = {}
    proc_root = Path("/proc")
    for child in proc_root.iterdir():
        if not child.name.isdigit():
            continue
        stat = _read_text(child / "stat")
        if not stat:
            continue
        close = stat.rfind(")")
        fields = stat[close + 2 :].split()
        if len(fields) < 2:
            continue
        try:
            parent_by_pid[int(child.name)] = int(fields[1])
        except ValueError:
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent_by_pid.items():
            if ppid in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return sorted(pid for pid in selected if Path(f"/proc/{pid}").exists())


def _process_totals(pids: list[int]) -> dict[str, int]:
    ticks = 0
    rss_kb = 0
    read_bytes = 0
    write_bytes = 0
    for pid in pids:
        stat = _read_text(Path(f"/proc/{pid}/stat"))
        if stat:
            close = stat.rfind(")")
            fields = stat[close + 2 :].split()
            if len(fields) > 12:
                try:
                    ticks += int(fields[11]) + int(fields[12])
                except ValueError:
                    pass
        status = _read_text(Path(f"/proc/{pid}/status"))
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                try:
                    rss_kb += int(line.split()[1])
                except (IndexError, ValueError):
                    pass
        io_text = _read_text(Path(f"/proc/{pid}/io"))
        for line in io_text.splitlines():
            if line.startswith("read_bytes:"):
                try:
                    read_bytes += int(line.split()[1])
                except (IndexError, ValueError):
                    pass
            elif line.startswith("write_bytes:"):
                try:
                    write_bytes += int(line.split()[1])
                except (IndexError, ValueError):
                    pass
    return {
        "cpu_ticks": ticks,
        "rss_kb": rss_kb,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
    }


def _process_namespace_ids(pids: list[int]) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for pid in pids:
        status = _read_text(Path(f"/proc/{pid}/status"))
        for line in status.splitlines():
            if not line.startswith("NSpid:"):
                continue
            values: list[int] = []
            for token in line.split()[1:]:
                try:
                    values.append(int(token))
                except ValueError:
                    pass
            mapping[str(pid)] = values
            break
    return mapping


def _host_snapshot() -> dict[str, object]:
    load1, load5, load15 = os.getloadavg()
    mem: dict[str, int] = {}
    for line in _read_text(Path("/proc/meminfo")).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields = value.split()
        if fields and fields[0].isdigit():
            mem[key] = int(fields[0])
    pressure = {
        name: _read_text(Path(f"/proc/pressure/{name}")).strip()
        for name in ("cpu", "memory", "io")
    }
    return {
        "loadavg": [load1, load5, load15],
        "mem_total_kb": mem.get("MemTotal"),
        "mem_available_kb": mem.get("MemAvailable"),
        "pressure": pressure,
    }


def _nvidia_snapshot() -> dict[str, object]:
    query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    apps_query = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu = subprocess.run(query, check=False, text=True, capture_output=True, timeout=2)
        apps = subprocess.run(apps_query, check=False, text=True, capture_output=True, timeout=2)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": gpu.returncode == 0,
        "gpu_rows": [line.strip() for line in gpu.stdout.splitlines() if line.strip()],
        "compute_rows": [line.strip() for line in apps.stdout.splitlines() if line.strip()],
        "gpu_stderr": gpu.stderr.strip(),
        "apps_stderr": apps.stderr.strip(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after --")
    if not 0.1 <= args.interval <= 2.0:
        parser.error("--interval must be between 0.1 and 2.0 seconds")
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = args.out_dir / "child_stdout.log"
    stderr_path = args.out_dir / "child_stderr.log"
    samples_path = args.out_dir / "resource_samples.jsonl"
    metadata_path = args.out_dir / "observation.json"
    start_wall = time.time()
    start_monotonic = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_fh:
        child = subprocess.Popen(
            args.command,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,
            env=os.environ.copy(),
        )
        sample_count = 0
        with samples_path.open("w", encoding="utf-8") as samples_fh:
            while True:
                pids = _proc_tree(child.pid)
                row = {
                    "sample_index": sample_count,
                    "utc_epoch_seconds": time.time(),
                    "elapsed_seconds": time.monotonic() - start_monotonic,
                    "root_pid": child.pid,
                    "process_tree_pids": pids,
                    "process_namespace_pids": _process_namespace_ids(pids),
                    "process_totals": _process_totals(pids),
                    "host": _host_snapshot(),
                    "nvidia": _nvidia_snapshot(),
                }
                samples_fh.write(json.dumps(row, sort_keys=True) + "\n")
                samples_fh.flush()
                sample_count += 1
                if child.poll() is not None:
                    break
                time.sleep(args.interval)
        returncode = child.wait()
    metadata = {
        "schema": "rocket_booster_recovery.resource_observation.v1",
        "command": args.command,
        "cwd": os.getcwd(),
        "start_utc_epoch_seconds": start_wall,
        "elapsed_seconds": time.monotonic() - start_monotonic,
        "sampling_interval_seconds": args.interval,
        "sample_count": sample_count,
        "root_pid": child.pid,
        "returncode": returncode,
        "selected_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "PRAXIST_ASSIGNED_GPU_UUIDS": os.environ.get("PRAXIST_ASSIGNED_GPU_UUIDS"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return returncode


if __name__ == "__main__":
    sys.exit(main())
