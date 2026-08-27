#!/usr/bin/env python3
"""Create a bounded inventory for converting a research repo into a Praxist task."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "site-packages",
    "dist",
    "build",
}

DOC_EXT = {".md", ".rst", ".txt", ".adoc", ".tex"}
PDF_EXT = {".pdf"}
REPORT_EXT = {".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".tif", ".tiff"}
CODE_EXT = {
    ".py",
    ".ipynb",
    ".sh",
    ".bash",
    ".zsh",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".cpp",
    ".cc",
    ".c",
    ".h",
    ".hpp",
    ".cu",
    ".m",
    ".jl",
    ".r",
    ".rs",
    ".go",
    ".java",
    ".ts",
    ".js",
}

ENV_NAMES = {
    "environment.yml",
    "environment.yaml",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "uv.lock",
    "poetry.lock",
    "Pipfile",
    "Dockerfile",
    "docker-compose.yml",
    "compose.yaml",
}

DATA_HINTS = re.compile(r"(^|/)(data|dataset|datasets|sim_data|fixtures)(/|$)", re.I)
RESULT_HINTS = re.compile(r"(^|/)(result|results|log|logs|runs|outputs|experiments|wandb|mlruns)(/|$)", re.I)
SIM_HINTS = re.compile(r"(mujoco|isaac|gazebo|ros|gym|gymnasium|simulator|simulation|pybullet|carla|coppelia)", re.I)
METRIC_HINTS = re.compile(
    r"(metric|accuracy|auc|f1|precision|recall|reward|objective|score|loss|success_rate|"
    r"collision|latency|throughput|rmse|mae|iou|bleu|rouge)",
    re.I,
)
COMMAND_HINTS = re.compile(r"^\s*(python|bash|sh|conda|pip|uv|poetry|docker|roslaunch|ros2|make|./)", re.I)


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in DOC_EXT | CODE_EXT | {".log", ".csv", ".tsv"}:
        return True
    return path.name in ENV_NAMES or path.name.lower() in {"readme", "license", "makefile"}


def read_snippet(path: Path, max_bytes: int) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
    except OSError:
        return ""
    if b"\x00" in data:
        return ""
    return data.decode("utf-8", errors="replace")


def add_limited(bucket: list[dict[str, Any]], item: dict[str, Any], limit: int) -> None:
    if len(bucket) < limit:
        bucket.append(item)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Existing research project root")
    parser.add_argument("--out", required=True, help="Output directory for inventory files")
    parser.add_argument("--max-files", type=int, default=20000)
    parser.add_argument("--max-snippets", type=int, default=300)
    parser.add_argument("--max-text-bytes", type=int, default=65536)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"root is not a directory: {root}")

    out.mkdir(parents=True, exist_ok=True)

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ext_counts: Counter[str] = Counter()
    top_dirs: Counter[str] = Counter()
    total_files = 0
    total_bytes = 0
    truncated = False
    metric_snippets: list[dict[str, str]] = []
    command_snippets: list[dict[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = rel(current, root)
        if rel_dir != ".":
            top_dirs[rel_dir.split("/", 1)[0]] += 1

        for name in filenames:
            if total_files >= args.max_files:
                truncated = True
                break
            path = current / name
            try:
                stat = path.stat()
            except OSError:
                continue
            total_files += 1
            total_bytes += stat.st_size
            suffix = path.suffix.lower()
            ext_counts[suffix or "<none>"] += 1
            r = rel(path, root)
            item = {"path": r, "size_bytes": stat.st_size}

            lower_name = name.lower()
            if suffix in DOC_EXT or lower_name.startswith("readme"):
                add_limited(categories["docs"], item, 500)
            if suffix in PDF_EXT:
                add_limited(categories["pdfs"], item, 200)
            if suffix in REPORT_EXT:
                add_limited(categories["reports"], item, 200)
            if suffix in IMAGE_EXT:
                add_limited(categories["images"], item, 300)
            if suffix in CODE_EXT:
                add_limited(categories["code"], item, 1000)
            if name in ENV_NAMES or lower_name in {n.lower() for n in ENV_NAMES}:
                add_limited(categories["environment_files"], item, 200)
            if DATA_HINTS.search(r):
                add_limited(categories["data_candidates"], item, 500)
            if RESULT_HINTS.search(r):
                add_limited(categories["result_log_candidates"], item, 500)
            if SIM_HINTS.search(r):
                add_limited(categories["simulator_candidates"], item, 300)

            if is_probably_text(path) and (len(metric_snippets) < args.max_snippets or len(command_snippets) < args.max_snippets):
                text = read_snippet(path, args.max_text_bytes)
                if text:
                    for line in text.splitlines():
                        stripped = line.strip()
                        if stripped and METRIC_HINTS.search(stripped):
                            add_limited(metric_snippets, {"path": r, "line": stripped[:500]}, args.max_snippets)
                        if stripped and COMMAND_HINTS.search(stripped):
                            add_limited(command_snippets, {"path": r, "line": stripped[:500]}, args.max_snippets)
                        if len(metric_snippets) >= args.max_snippets and len(command_snippets) >= args.max_snippets:
                            break
        if truncated:
            break

    inventory: dict[str, Any] = {
        "schema_version": "praxist.task_initialization_inventory.v1",
        "root": str(root),
        "truncated": truncated,
        "limits": {
            "max_files": args.max_files,
            "max_snippets": args.max_snippets,
            "max_text_bytes": args.max_text_bytes,
        },
        "totals": {"files_seen": total_files, "bytes_seen": total_bytes},
        "top_dirs": top_dirs.most_common(50),
        "extension_counts": ext_counts.most_common(80),
        "categories": categories,
        "metric_snippets": metric_snippets,
        "command_snippets": command_snippets,
    }

    (out / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Project Inventory Summary",
        "",
        f"- root: `{root}`",
        f"- files seen: {total_files}",
        f"- bytes seen: {total_bytes}",
        f"- truncated: {truncated}",
        "",
        "## Categories",
    ]
    for key in sorted(categories):
        lines.append(f"- {key}: {len(categories[key])} recorded")
    lines.extend(["", "## Top Directories"])
    for name, count in top_dirs.most_common(20):
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Metric Clues"])
    for item in metric_snippets[:40]:
        lines.append(f"- `{item['path']}`: {item['line']}")
    lines.extend(["", "## Command Clues"])
    for item in command_snippets[:40]:
        lines.append(f"- `{item['path']}`: {item['line']}")
    (out / "inventory_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out / "inventory.json")
    print(out / "inventory_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
