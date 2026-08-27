# Troubleshooting

Use the smallest command that can identify the failing boundary. Do not edit run
artifacts to make a failed check appear healthy.

## Host or Authentication Is Not Ready

```bash
praxist doctor --json
```

For Codex-native mode:

```bash
praxist setup --profile codex-native --install-skills codex  # or: claude
praxist doctor --codex-native --task-path /absolute/path/to/task --json
```

If a tested runtime package is missing or mismatched, invoke
the `praxist-runtime-install` skill instead of independently upgrading an SDK.

Codex-native authentication behavior and diagnostic-override semantics are
defined in [Credentials](../guides/credentials.md#codex-native-mode-authentication).
Do not apply that repair to another setup profile.

## Package Download Certificate Failure

If installation reports `SSLCertVerificationError` or
`CERTIFICATE_VERIFY_FAILED`, repair the selected Python trust store and rerun
the pip command. On macOS, python.org distributions provide an `Install
Certificates.command` alongside the installed Python. Managed Python or
corporate environments should use their supported CA-bundle configuration.
Do not bypass TLS verification with `trusted-host`, and never place an API key
in a command argument while troubleshooting connectivity.

## Task Does Not Resolve

```bash
praxist resolve /absolute/path/to/task
```

Resolution makes no LLM calls. It reports invalid task configuration, missing
plugin descriptors, unresolved task-local references, and unsupported
agent runtime/API provider combinations before launch.

## A Started Run Disappears

`praxist start --daemonize --json` reports process creation before every
research stage necessarily initializes. Inspect:

```bash
praxist status --json
praxist --monitor --latest
```

Then read the run's `run_summary.json` and launcher log path reported by start.
A stale registry record means the registered process is no longer alive; it is
not proof that the research completed.

## Run Appears Stalled

Use:

Invoke the `praxist-diagnostic` skill in the current agent.

The diagnostic workflow separates a long active experiment from missing stage
artifacts, resource starvation, runtime/API friction, blocked generation close,
or incomplete evidence. The foreground monitor is observational and must not be
used as scientific evidence.

## Stop or Resume

Prefer the lifecycle skill:

Ask the `praxist-control` skill to stop the current run or resume the latest
run.

Interrupted generation, Principal Investigator (PI) panel, and Gems boundaries
require artifact-aware inspection before `resume`. The canonical procedure is
defined in
[Direct CLI Operations](../guides/operators.md).

## Documentation Build Fails

```bash
uv sync --extra docs
uv run python scripts/build_docs_site.py
```

The build is strict. It fails on stale generated references, unowned pages,
duplicate navigation ownership, broken local links, or MkDocs warnings.

For configuration and credential details, use
[Configuration Discipline](../concepts/config_discipline.md) and
[Credentials](../guides/credentials.md).
