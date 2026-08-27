# Agent Skills

Praxist skills are operator workflows for Codex and Claude Code. Invoke a skill
as `$name` in Codex or `/name` in Claude Code. Skill files instruct the agent;
they do not run as background services and they do not replace the `praxist`
CLI.

## Choose by Goal

| Goal | Codex | Claude Code |
|---|---|---|
| Learn Praxist and inspect host readiness | `$praxist-onboarding` | `/praxist-onboarding` |
| Install or repair Praxist runtime dependencies | `$praxist-runtime-install` | `/praxist-runtime-install` |
| Build or repair a task harness without launching | `$praxist-task-initialization` | `/praxist-task-initialization` |
| Confirm task design interactively | `$praxist-interactive-task-init` | `/praxist-interactive-task-init` |
| Initialize and launch with a configured API provider | `$praxist-takeover` | `/praxist-takeover` |
| Initialize and launch with a saved Codex login | `$praxist-takeover-codex` | `/praxist-takeover-codex` |
| Start, stop, resume, monitor, or inspect runs | `$praxist-control` | `/praxist-control` |
| Diagnose run health or generate reports | `$praxist-diagnostic` | `/praxist-diagnostic` |
| Gather literature and benchmark context | `$praxist-scientific-research` | `/praxist-scientific-research` |
| Draw a terminal line chart | `$terminal-line-plot` | `/terminal-line-plot` |

The complete catalog and activation descriptions are generated from the actual
skill metadata in [Skills Reference](../reference/skills.md).

After installation, read [Your First Task](../getting-started/first-task.md).
Then invoke the matching takeover explicitly without having to remember a skill
name:

```bash
praxist --takeover --task-path /absolute/path/to/research-project
```

For agent-managed installation, the agent follows
[the OOBE runbook](../agents/oobe-install.md) and stops after readiness checks.
Takeover remains a separate post-manual action.

## Install or Refresh

An operator installation registers the bundled skills automatically. Refresh
them after upgrading Praxist:

```bash
praxist install-skills --target codex --replace
praxist install-skills --target claude --replace
```

Run only the command for the host you use. Codex installs under
`${CODEX_SKILLS_DIR:-~/.agents/skills}`; Claude Code installs under
`${CLAUDE_SKILLS_DIR:-~/.claude/skills}`.

This replaces only same-name Praxist skills managed by the current package.
Unrelated user skills are not touched. An operator-owned same-name path is
reported as a conflict and remains unchanged. Interactive setup can preserve a
backup before an explicitly approved replacement.

Source contributors can use symlinks for fast iteration:

```bash
bash scripts/install_codex_skills.sh
bash scripts/install_codex_skills.sh --target claude
```

## Workflow Boundaries

- Onboarding inspects and explains; it does not launch research.
- Task initialization writes only the selected task project.
- Control owns lifecycle operations and uses the CLI.
- Diagnostic is analysis-only unless the user explicitly asks for task-level
  improvement.
- Scientific research records sourced context; it never claims literature as
  measured task performance.

Task construction and takeover are explained in
[Your First Task](../getting-started/first-task.md). Report semantics live in
[Run Reports](../guides/user-facing-reports-and-init.md), and lifecycle commands
live in [Direct CLI Operations](../guides/operators.md).
