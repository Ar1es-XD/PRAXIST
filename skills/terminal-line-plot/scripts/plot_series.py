#!/usr/bin/env python3
"""Render a numeric series as an ASCII line chart on stdout."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _as_number(value: Any, *, field: str) -> float:
    if value is None:
        raise ValueError(f"{field} is null")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} is not finite: {value!r}")
    return number


def _display_x(value: Any) -> str:
    text = str(value)
    return text if text else "?"


def parse_points(text: str) -> list[tuple[str, float]]:
    points: list[tuple[str, float]] = []
    for raw in text.replace("\n", ",").split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" in item:
            x_raw, y_raw = item.split(":", 1)
        elif "=" in item:
            x_raw, y_raw = item.split("=", 1)
        else:
            parts = item.split()
            if len(parts) != 2:
                raise ValueError(f"point must be x:y, x=y, or 'x y': {item!r}")
            x_raw, y_raw = parts
        points.append((_display_x(x_raw.strip()), _as_number(y_raw.strip(), field="y")))
    return points


def parse_csv(
    path: Path | None, text: str | None, x_field: str | None, y_field: str | None
) -> list[tuple[str, float]]:
    raw = path.read_text() if path else (text or "")
    rows = list(csv.DictReader(raw.splitlines()))
    if rows and x_field and y_field:
        return [
            (_display_x(row.get(x_field, "")), _as_number(row.get(y_field), field=y_field))
            for row in rows
        ]

    plain_rows = list(csv.reader(raw.splitlines()))
    if not plain_rows:
        return []
    first = plain_rows[0]
    start = 1 if len(first) >= 2 and not _looks_numeric(first[1]) else 0
    out: list[tuple[str, float]] = []
    for row in plain_rows[start:]:
        if len(row) < 2:
            continue
        out.append((_display_x(row[0]), _as_number(row[1], field="y")))
    return out


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _iter_json_records(data: Any) -> Iterable[tuple[Any, Any]]:
    if isinstance(data, dict):
        yield from data.items()
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                yield idx, item
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                yield item[0], item[1]
            else:
                yield idx, item
    else:
        raise ValueError(
            "JSON input must be an object, array of objects, array of pairs, or array of numbers"
        )


def parse_json(
    path: Path | None, text: str | None, x_field: str | None, y_field: str | None
) -> list[tuple[str, float]]:
    raw = path.read_text() if path else (text or "")
    data = json.loads(raw)
    out: list[tuple[str, float]] = []
    for default_x, record in _iter_json_records(data):
        if isinstance(record, dict):
            x_value = record.get(x_field) if x_field else default_x
            y_value = record.get(y_field) if y_field else None
            if y_value is None and len(record) == 1:
                y_value = next(iter(record.values()))
        else:
            x_value = default_x
            y_value = record
        out.append((_display_x(x_value), _as_number(y_value, field=y_field or "y")))
    return out


def _line_char(y0: int, y1: int) -> str:
    if y1 < y0:
        return "╱"
    if y1 > y0:
        return "╲"
    return "─"


def render_chart(
    points: list[tuple[str, float]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    height: int,
    x_step: int,
    provisional_x: set[str],
    higher_better: bool,
    lower_better: bool,
    invert_y_note: bool,
) -> str:
    if len(points) < 2:
        raise ValueError("need at least two points to draw a line chart")
    height = max(5, height)
    x_step = max(2, x_step)
    width = (len(points) - 1) * x_step + 1
    values = [y for _, y in points]
    ymin = min(values)
    ymax = max(values)
    if ymax == ymin:
        pad = max(abs(ymax) * 0.05, 1.0)
    else:
        pad = (ymax - ymin) * 0.06
    ymin -= pad
    ymax += pad

    canvas = [[" " for _ in range(width)] for __ in range(height)]
    coords: list[tuple[int, int, str, float]] = []
    for idx, (x_value, y_value) in enumerate(points):
        x = idx * x_step
        row = round((ymax - y_value) / (ymax - ymin) * (height - 1))
        coords.append((x, row, x_value, y_value))

    def mark(x: int, y: int, char: str) -> None:
        if 0 <= y < height and 0 <= x < width:
            canvas[y][x] = char

    for left, right in zip(coords, coords[1:], strict=False):
        x0, y0, _, _ = left
        x1, y1, _, _ = right
        dx = x1 - x0
        for step in range(1, dx):
            t = step / dx
            x = x0 + step
            y = round(y0 + (y1 - y0) * t)
            if canvas[y][x] == " ":
                mark(x, y, _line_char(y0, y1))

    for x, y, x_value, _ in coords:
        mark(x, y, "○" if x_value in provisional_x else "●")

    note_parts = []
    if higher_better:
        note_parts.append("higher is better")
    if lower_better:
        note_parts.append("lower is better")
    if provisional_x:
        note_parts.append("○ = provisional/incomplete")
    if invert_y_note:
        note_parts.append("axis not inverted")

    lines: list[str] = []
    lines.append(title)
    if note_parts:
        lines.append("; ".join(note_parts))
    if y_label:
        lines.append(f"y: {y_label}")
    lines.append("")
    for row_index, row in enumerate(canvas):
        y_tick = ymax - (ymax - ymin) * row_index / (height - 1)
        lines.append(f"{y_tick:7.2f} │{''.join(row)}")
    lines.append("        └" + "─" * width)

    label = [" " for _ in range(width)]
    for x, _, x_value, _ in coords:
        text = str(x_value)
        start = x - len(text) // 2
        for offset, char in enumerate(text):
            pos = start + offset
            if 0 <= pos < width:
                label[pos] = char
    lines.append("         " + "".join(label))
    if x_label:
        lines.append("         " + x_label)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--points", help="Point list like '0:1.2,1:2.0,2:1.7'.")
    source.add_argument(
        "--csv", type=Path, help="CSV file. Use --x-field and --y-field for headers."
    )
    source.add_argument(
        "--json",
        type=Path,
        help="JSON file: mapping, array of pairs, array of numbers, or array of objects.",
    )
    source.add_argument(
        "--stdin", choices=["points", "csv", "json"], help="Read points, CSV, or JSON from stdin."
    )
    parser.add_argument("--x-field", help="Field name for x values in CSV/JSON objects.")
    parser.add_argument("--y-field", help="Field name for y values in CSV/JSON objects.")
    parser.add_argument("--title", default="Line chart")
    parser.add_argument("--x-label", default="x")
    parser.add_argument("--y-label", default="value")
    parser.add_argument("--height", type=int, default=18)
    parser.add_argument("--x-step", type=int, default=8)
    parser.add_argument(
        "--provisional-x",
        action="append",
        default=[],
        help="Mark this x value with an open circle. May repeat.",
    )
    parser.add_argument("--higher-better", action="store_true")
    parser.add_argument("--lower-better", action="store_true")
    parser.add_argument(
        "--invert-y-note",
        action="store_true",
        help="Add a note that the displayed axis is not inverted.",
    )
    parser.add_argument(
        "--value-table", action="store_true", help="Print plotted values below the chart."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stdin_text = sys.stdin.read() if args.stdin else None
    if args.points is not None:
        points = parse_points(args.points)
    elif args.csv is not None:
        points = parse_csv(args.csv, None, args.x_field, args.y_field)
    elif args.json is not None:
        points = parse_json(args.json, None, args.x_field, args.y_field)
    elif args.stdin == "points":
        points = parse_points(stdin_text or "")
    elif args.stdin == "csv":
        points = parse_csv(None, stdin_text, args.x_field, args.y_field)
    elif args.stdin == "json":
        points = parse_json(None, stdin_text, args.x_field, args.y_field)
    else:
        raise AssertionError("unreachable")

    chart = render_chart(
        points,
        title=args.title,
        x_label=args.x_label,
        y_label=args.y_label,
        height=args.height,
        x_step=args.x_step,
        provisional_x=set(args.provisional_x),
        higher_better=args.higher_better,
        lower_better=args.lower_better,
        invert_y_note=args.invert_y_note,
    )
    print(chart)
    if args.value_table:
        print("\nvalues:")
        for x_value, y_value in points:
            suffix = "  (provisional)" if x_value in set(args.provisional_x) else ""
            print(f"  {x_value}: {y_value:.4g}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
