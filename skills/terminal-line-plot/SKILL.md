---
name: terminal-line-plot
description: Draw readable ASCII line charts directly in the terminal from numeric series, command output, CSV, JSON, or manually extracted points. Use when the user asks for a curve, trend line, score progression, leaderboard trend, metric history, or any plot that should be visible in a CLI/chat transcript without opening a GUI or writing image files.
---

# Terminal Line Plot

Use this skill to turn numeric sequences into terminal-readable line charts.
It is task-agnostic: do not assume any domain, metric direction, filtering rule,
or data layout unless the user or artifacts define it.

## Workflow

1. Identify the series to plot:
   - x values are usually ordered steps, dates, versions, generations, epochs, or run ids.
   - y values must be numeric.
   - If multiple plausible metrics exist, choose the one the user asked for; otherwise state the selected metric before plotting.
2. Preserve metric semantics:
   - Use exact metric names from artifacts when available.
   - Label whether higher or lower is better if known.
   - Do not mix partial and complete points silently; mark provisional or incomplete points in the title, notes, or point list.
3. Use `scripts/plot_series.py` for deterministic plotting.
4. Print a compact table or point list when it prevents misreading the curve.
5. Do not write image files unless the user explicitly asks. Prefer stdout.

## Script Usage

Run the bundled script from the skill directory:

```bash
python /path/to/terminal-line-plot/scripts/plot_series.py \
  --points "0:-4.49,1:-4.53,2:-4.87,3:-4.77,4:-3.23" \
  --title "Best future_fitness by generation" \
  --x-label generation \
  --y-label future_fitness \
  --higher-better
```

Input options:

- `--points "x:y,x:y,..."` for quick point lists.
- `--csv FILE --x-field FIELD --y-field FIELD` for CSV.
- `--json FILE --x-field FIELD --y-field FIELD` for JSON arrays of objects, arrays of pairs, or mappings.
- `--stdin csv|json|points` to read data from stdin.

Useful options:

- `--height N` controls chart height.
- `--x-step N` controls horizontal spacing between points.
- `--provisional-x X` may be repeated to mark incomplete points with `○`.
- `--value-table` prints the plotted values below the chart.
- `--invert-y-note` only changes the note; it does not invert the axis.

## Plotting Guidelines

- Prefer line drawing characters `●`, `○`, `╱`, `╲`, and `─`.
- Keep the x-axis readable. If labels collide, increase `--x-step` or plot fewer points.
- Use a taller chart for noisy or close-valued curves.
- For negative scores where "higher is better", the visually higher point should still mean numerically larger.
- If the active/final point is incomplete, mark it provisional and do not draw conclusions from it.
- For dense data, aggregate intentionally: best per generation, mean per epoch bucket, rolling median, or another stated rule.

## Example Patterns

Manual points:

```bash
python /path/to/terminal-line-plot/scripts/plot_series.py \
  --points "0:7.88,1:7.78,2:8.45,3:8.61,4:9.28" \
  --title "Mean objective value by generation" \
  --y-label "objective" \
  --value-table
```

CSV:

```bash
python /path/to/terminal-line-plot/scripts/plot_series.py \
  --csv metrics.csv \
  --x-field generation \
  --y-field score \
  --title "Score by generation"
```

JSON:

```bash
python /path/to/terminal-line-plot/scripts/plot_series.py \
  --json metrics.json \
  --x-field gen \
  --y-field best_score
```
