# Installation

Praxist release CI qualifies Linux on CPython 3.11 and 3.12. The package also
targets macOS and other CPython 3.11+ environments, but those combinations are
not continuously release-tested. Run `praxist doctor` on every host before
launching research. Install Praxist in the Python environment from which Codex
or Claude Code will operate. Task-specific packages, datasets, simulators, and
accelerator libraries remain in the task environment.

## Prerequisites

```bash
python3 --version
codex --version       # when Codex is the operator interface
claude --version      # when Claude Code is the operator interface
```

The selected agent interface must already be usable. Authentication requirements
for each runtime route are defined in [Credentials](../guides/credentials.md).

## Install And Configure

The supported runtime extras install both maintained peer runtimes and the
Codex-native integration. Choose the agent application whose bundled skills
should be registered; each complete installation command is a single line.

```bash
# Codex
python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]" && praxist setup --interactive --install-skills codex
# Claude Code
python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]" && praxist setup --interactive --install-skills claude
```

Run the command in the intended active virtual environment when the host Python
is externally managed. `python3 -m pip` keeps the package and `praxist`
entrypoint tied to the same interpreter. The base `praxist` package supports
inspection and package-level CLI operations; the documented extras are the
complete research-runtime installation. The explicit public index prevents an
incomplete package mirror from silently omitting a pinned runtime SDK.

The command stops if installation, setup, or readiness fails. A successful
command also stops at that boundary: it does not select a project or launch
research. The
[Quickstart](quickstart.md) is the sole description of the OOBE sequence and
its agent-managed alternative.

## What Installation Changes

The one-line flow:

- installs the tested agent-runtime dependencies;
- exposes `praxist` in the selected Python environment;
- registers bundled skills only for the selected agent host;
- writes only the configuration explicitly selected during setup;
- materializes writable complete examples outside the package;
- runs host diagnostics; and
- stops before project selection and research launch.

It does not install task training dependencies, CUDA, datasets, simulators,
human-facing Codex or Claude Code applications, or a collector service.
Codex-native support downloads a platform-specific runtime package of roughly
100-150 MB, so the first pip installation may take several minutes.

Same-name operator-owned skills are never replaced silently. Interactive setup
offers keep, backup-and-replace, or cancel; non-interactive setup reports the
conflict and stops.

Open the hosted documentation with:

```bash
praxist docs
```

Read the [Quickstart](quickstart.md) and [Your First Task](first-task.md)
before using the separate takeover command.

## Writable Complete Examples

Read-only package resources are never research workspaces. First-use setup copies
bundled complete examples to `${PRAXIST_EXAMPLES_HOME:-~/PraxistExamples}` and
prints each writable path. Existing destinations are preserved during upgrades.
The discovery and materialization commands, available projects, and asset
boundary are owned by [Examples And Templates](../guides/examples-and-templates.md).

## Verify

```bash
praxist --version
praxist doctor
praxist examples list
```

`doctor` checks each detected Praxist-managed skill host. Use
`praxist doctor --target codex` or `praxist doctor --target claude` to inspect
one host explicitly. If the shell cannot find `praxist`, activate the Python
environment used for installation or add that environment's script directory
to `PATH`.

## Uninstall

Stop active runs, remove Praxist-managed user state, then uninstall the Python
package from the same environment:

```bash
praxist uninstall --dry-run
praxist uninstall
python3 -m pip uninstall praxist
```

`praxist uninstall` removes only proven Praxist-managed skills, configuration,
local agreement/usage state, registry, and cache. `--keep-user-data` preserves
user records and caches. Pip removes the package from the active Python
environment. Research projects, writable examples, task environments, run
artifacts, agent applications, datasets, and task dependencies are never
removed.

## Source Development

Contributors use the repository environment rather than the operator install:

```bash
uv sync --group dev --extra docs
uv run praxist --help
```

See [Contributing](../guides/contributing.md) for verification and
[Platform Support](../operations/platform-support.md) for host boundaries.
