# Generic Plugins

Generic plugins are reusable system components under `praxist/plugins/**` or another
explicit plugin root. A task project may also ship its own generic plugins under
`<task_path>/.praxist/plugins/`; those are discovered as `source="task_project"` only
when `--task-path` selects the task (see
[Task Projects → Boundary Rules](task-projects.md#boundary-rules)).

## Plugin Manifest

Each plugin has a `plugin.yaml` manifest describing:

- plugin kind and name;
- version and stability;
- executable entrypoint or manifest-only contract;
- dependencies and compatibility;
- declared code/assets for replay hash coverage.

The plugin loader discovers candidates, resolves dependencies, checks source
priority, and writes the selected plugin manifest into the run directory.

### Stability as an interface contract

`stability` describes the **interface contract** a plugin promises — schema,
prompt shape, role-API backward compatibility — not whether the plugin is
trustworthy to execute. Trust comes from the plugin's **source**, gated via
`TRUSTED_EXECUTION_SOURCES` (`bundled` and `task_project`).

The two paths are gated differently:

- **Bundled plugins** must declare the kind's strict expected stability
  (e.g. `v1_stable` for `panel_topology`, `agent_runtime`, `workflow_stage`).
  A bundled plugin affects every task project, so the contract there has to
  hold.
- **Task-local plugins** under `<task>/.praxist/plugins/` are
  scope-isolated to one task project and high-churn by design; they may
  declare any `stability` value (commonly `v0_experimental`) without
  triggering the kind-mismatch check. Source trust is enough.

## Minimal Executable Plugin

An executable plugin usually has this shape:

```text
praxist/plugins/<kind_plural>/<name>/
  plugin.yaml
  adapter.py
  README.md          # optional, for complex plugins
```

`plugin.yaml` should declare the plugin ref, compatibility, entrypoint, and code
files that participate in source hashing. `adapter.py` should expose a small
factory or adapter object matching the kind-specific contract.

The plugin content hash covers the manifest and its declared code and assets.
Imported modules or assets omitted from the manifest are outside that plugin
content hash.

Do not name every implementation file `plugin.py` by habit. Use names that
describe the plugin's internal structure.

## Plugin-Supplied Assets

Some plugin kinds accept assets shipped alongside the manifest, declared
through dedicated manifest fields rather than ad-hoc paths.

- `panel_topology`: a plugin may declare `topology.prompts_dir` to ship
  its own Jinja prompt templates for `BasePI` and `ChairArbiter`. The
  bundled prompts for the multi-agent Principal Investigator (PI) panel are
  used as a fallback for anything the plugin does not override. See
  [Panel Topology Prompts](../concepts/panel_topology_prompts.md).

When a plugin ships assets, declare the paths under `code` / `assets`
in `plugin.yaml` so they participate in replay source hashing.

## What Belongs in a Plugin

Put code in a generic plugin when it is reusable across task projects:

- runtime adapters;
- API provider adapters;
- tool servers;
- workflow stages;
- graph maintainers;
- generic budget policies.

Do not put benchmark-specific research facts or task-local role contracts into a
generic plugin. Those belong in the task project.

## Decision Test

Before adding a plugin, ask whether two unrelated task projects could use it
without copying task facts. If the answer is no, it probably belongs in a task
project.

Before adding code to core, ask whether the behavior can be selected, replaced,
or disabled through a plugin. If yes, it belongs in a plugin.

## Plugin Tests

Plugin changes should add:

- plugin-local unit tests when the plugin has real code;
- kind-specific conformance tests under `tests/conformance/`;
- workflow smoke tests when the plugin participates in startup or run execution;
- replay/hash tests when plugin code or assets affect run reproducibility.

The default test path must not require real model keys, network, GPUs, or an
external task checkout.
