# Praxist Agent Skills

Praxist bundles portable skills for Codex and Claude Code onboarding, task initialization, lifecycle
control, diagnostics, scientific context gathering, and terminal plots. The
generated [Skills Reference](../docs/reference/skills.md) is the catalog; each
`skills/*/SKILL.md` file owns its activation contract.

## Install

The local-terminal OOBE installs the package, configures Praxist, and registers
skills in one command:

```bash
python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]" && praxist setup --interactive --install-skills codex
```

It uses local Up/Down, Enter, and Esc controls and stops after readiness checks.
Provider keys are entered through a length-preserving masked prompt and never
through an agent conversation. Installation does not select a project or invoke
a takeover skill.

Refresh bundled skills after an upgrade:

```bash
praxist install-skills --target codex --replace
praxist install-skills --target claude --replace
```

For a source checkout used in active skill development:

```bash
bash scripts/install_codex_skills.sh --target codex
bash scripts/install_codex_skills.sh --target claude
```

The source script uses symlinks. The CLI command uses package-managed
registrations and never removes unrelated skills.

To remove the complete current-user installation, first stop active runs, run
`praxist uninstall`, then uninstall the Python package from its environment.
The canonical safety contract is documented under
[Installation](../docs/getting-started/installation.md#uninstall).

## Start Research After Setup

Read [Your First Task](../docs/getting-started/first-task.md) before using the
separate takeover entrypoint.

Then open the matching agent workflow explicitly:

```bash
praxist --takeover --task-path /absolute/path/to/research-project
praxist --takeover --operator claude --task-path /absolute/path/to/research-project
```

For an already open agent session, continue with the explicit skill path below.

From a research project that already runs on the current machine:

```bash
cd /path/to/your/research-project
codex --yolo
claude --dangerously-skip-permissions
```

Choose one complete takeover. Codex uses a `$` prefix and Claude Code uses a
`/` prefix:

```text
$praxist-takeover-codex
/praxist-takeover-codex
```

for Codex-native saved-login operation, or:

```text
$praxist-takeover
/praxist-takeover
```

for the configured provider. Both run the complete onboarding flow, task
initialization or repair, validation, and detached launch. For separate steps:

```text
praxist-onboarding
praxist-task-initialization
praxist-control
```

Use the `praxist-interactive-task-init` skill when task design should be confirmed before
files are written.

## Operate and Diagnose

Routine requests can be written in natural language; the agent selects the relevant
skill automatically:

```text
Report current research progress and list the strongest variant in every
completed generation with its task-defined performance metrics.

Stop the current Praxist run.

Resume the latest interrupted run from its last safe boundary.

Diagnose why the current run is not improving.
```

Explicit `praxist-control` and `praxist-diagnostic` skill invocations remain
available when the operator wants to force a particular workflow.

The control skill reports the exact monitor command:

```bash
praxist --monitor --run-id <run_id>
```

Human-readable diagnostic and strongest-variant reports are written under
`<task>/docs/praxist_reports/` when requested or triggered by the run-report
module.

## Boundaries

- Onboarding establishes context and waits.
- Initialization changes only the selected task project.
- Control delegates lifecycle actions to the `praxist` CLI.
- Diagnostics are read-only unless task-only improvement is explicitly
  requested.
- Skills do not define CLI flags, task schema, or research artifact semantics;
  they link to the canonical product documentation.

See [Agent Skills](../docs/user-guide/skills.md) for goal-oriented use and
[Direct CLI Operations](../docs/guides/operators.md) for lifecycle behavior.
