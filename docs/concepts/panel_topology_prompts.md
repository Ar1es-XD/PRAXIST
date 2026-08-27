# Panel Topology Prompts

A Principal Investigator (PI) agent independently proposes next-generation
work. In multi-PI topologies, a Chair compares those proposals and commits one
agenda.

> **Principle.** A `panel_topology` plugin can ship its own Jinja prompt
> templates next to its manifest. The bundled prompts under
> `praxist/plugins/workflow_stages/research_loop/backend/multi_pi/prompts/`
> are the fallback, not the only choice.

This page documents the `topology.prompts_dir` contract for panel topology
plugins. It is the override point referenced from
`PanelTopologySpec.prompts_dir` in `praxist/core/panel_topology.py`.

## Why

The multi-PI backend renders three Jinja templates:

| Template | Renderer | Purpose |
|---|---|---|
| `base.jinja2` | `BasePI.render_prompt` | Round-1 independent PI memo |
| `round2_cross_review.jinja2` | `BasePI.run_cross_review` | Round-2 anonymized cross-review |
| `chair.jinja2` | `ChairArbiter.render_prompt` | Chair synthesis prompt |

The bundled directory is the default loader source. A panel-topology plugin
that needs a different collaboration vocabulary can override one or more
templates without changing bundled files.

`topology.prompts_dir` lets a plugin ship its own prompts directory alongside
its topology contract, while keeping the bundled templates as a fallback for
everything the plugin does not override.

## Manifest field

A panel topology plugin opts in by declaring `topology.prompts_dir` in
its `plugin.yaml`:

```yaml
schema_version: 1
name: my_panel
kind: panel_topology
topology:
  topology_ref: panel_topology:my_panel
  prompts_dir: prompts/                  # relative to this plugin directory
  modes: { ... }
  roles: [ ... ]
  rounds: [ ... ]
```

Layout on disk:

```text
my_panel_topology/
├── plugin.yaml
└── prompts/
    └── base.jinja2     # overrides bundled; chair.jinja2 etc. fall through
```

### Path resolution

`panel_topology_from_manifest(...)` (in `praxist/core/panel_topology.py`)
resolves the manifest value through `_resolve_prompts_dir`:

| `prompts_dir` value | Behavior |
|---|---|
| absent / `null` / `""` | `PanelTopologySpec.prompts_dir = None`; use bundled prompts only. |
| relative string (e.g. `prompts/`) | Resolved against the plugin directory. Must point at an existing directory. |
| absolute string | Accepted as-is. Must point at an existing directory. |
| non-string | `ValueError` at manifest time. |
| relative string without a known plugin path | `ValueError` (cannot resolve safely). |
| any value pointing at a missing directory | `ValueError` at manifest time. |

All `ValueError`s are raised at topology resolution, not at first
template lookup. Manifest-authoring bugs surface before any PI starts
rendering.

## Loader chain

When `PanelTopologySpec.prompts_dir` is supplied, both `BasePI` and
`ChairArbiter` build their Jinja `FileSystemLoader` from the search list:

```text
[plugin prompts_dir, bundled prompts dir]
```

Jinja walks the list in order, so a plugin can override one template
(say, `base.jinja2`) and let the rest fall back to the bundled version.
When `prompts_dir` is `None`, the search list contains only the bundled
directory.

This applies to all three render sites: `BasePI.render_prompt`,
`BasePI.run_cross_review`, and `ChairArbiter.render_prompt`.

## What gets threaded where

| Layer | What it does |
|---|---|
| `PanelTopologySpec.prompts_dir` | Frozen `Path \| None` on the resolved topology. |
| `panel_topology_from_manifest(..., plugin_path=...)` | Resolves the manifest value relative to the plugin directory; fails fast on missing dirs. |
| `legacy_two_round_executor.run_panel` | Resolves the topology once, pulls `topology.prompts_dir`, and threads it to both `instantiate_pi_roles(...)` and `ChairArbiter(...)`. |
| `role_bindings.instantiate_pi_roles` | Forwards `prompts_dir` to each PI constructor. |
| `BasePI.__init__` / `ChairArbiter.__init__` | Store `self.prompts_dir` and use it when building the loader. |

## Backward compatibility

A panel topology that does not declare `prompts_dir` produces
`PanelTopologySpec(prompts_dir=None)` and uses the bundled prompts. The bundled
`legacy_multi_pi_two_round/plugin.yaml` follows this path.

## Authoring checklist

When adding a panel topology plugin that ships its own prompts:

1. Place templates under `<plugin_dir>/prompts/` (or another directory
   referenced by `topology.prompts_dir`).
2. Only override the templates you actually need to change. Templates
   you omit fall back to the bundled versions automatically.
3. Preserve the public template variables consumed by `BasePI` and
   `ChairArbiter` — overriding the layout is supported, dropping
   variables silently is not.
4. Add a plugin-local unit test that asserts the override is wired
   (see `tests/unit/test_panel_topology_prompts_override.py` for a
   canonical pattern: render with and without the override, check for a
   plugin-specific token in the output).

## See also

- `praxist/core/panel_topology.py` — `PanelTopologySpec`,
  `panel_topology_from_manifest`, `_resolve_prompts_dir`.
- `praxist/plugins/workflow_stages/research_loop/backend/multi_pi/`
  — `BasePI`, `ChairArbiter`, bundled prompt templates.
