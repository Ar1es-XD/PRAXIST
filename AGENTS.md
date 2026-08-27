# Praxist Contributor Contract

This file is the machine-friendly entrypoint for agents and humans editing this
repository. Read it before making architecture, runtime, plugin, task, test, or
documentation changes.

## 1. Repository Identity

Praxist is an autonomous research platform.

The product nickname is Praxist.

The canonical Python package is `praxist`.

The canonical operator entrypoint is:

```bash
praxist start --task-path /path/to/task-project --daemonize --json
```

From a source checkout, run the same lifecycle through `uv`:

```bash
uv run praxist start --task-path /path/to/task-project --daemonize --json
```

The external task path model is mandatory. Real task projects are explicit
inputs, not bundled system source.

The active architecture documents are:

```text
docs/concepts/architecture.md
docs/concepts/runtime-model.md
```

The source documentation entry is:

```text
docs/index.md
```

When Codex or Claude Code is asked to install Praxist for a first-time user,
follow the dedicated OOBE contract before ordinary onboarding or takeover:

```text
docs/agents/oobe-install.md
```

The agent-managed OOBE lane and the local terminal wizard share setup profiles,
configuration, diagnostics, and skill registration. Installation stops after
readiness checks; project selection and takeover require a separate request
after the operator reads the first-task manual. The two lanes do not share an
interaction controller or a second persistent OOBE state store.
After `python3 -m pip install --index-url https://pypi.org/simple "praxist[agents,codex]"`, run
`praxist setup --agent-managed` and
follow its `next_required_action`; provider defaults, saved credentials, and a
passing doctor report do not count as an operator profile selection.

## 2. Current Source Layout

The important root directories are:

```text
praxist/        system package
praxist/core/   protocol, registry, replay, credentials, ledgers, stores
praxist/plugins/ generic reusable plugin catalog
praxist/infrastructure/ internal infra adapters and shims
praxist/product_usage/ opt-in pseudonymous usage protocol and client
praxist/testing/ offline fake workflow fixture support
services/product_usage/ separately deployed usage collector and retention service
examples/           complete runnable reference projects
templates/          tracked task templates and smoke fixtures
tests/              long-lived unit, conformance, workflow, hardening tests
docs/               source docs
scripts/            operator and build wrappers
tasks/              ignored local task workspace
```

The canonical package directory is `praxist/`. Do not introduce parallel
system-package directories. The following unrelated obsolete directory name is
also forbidden:

```text
auto_research/
```

Do not use them in active code, examples, commands, or new API reference pages.

Product usage is an observation boundary, not research-loop semantics. The
built-in client and closed event protocol live in `praxist/product_usage/`; the
neutral lifecycle projection lives in `praxist/infrastructure/`; collector and
deployment assets live in `services/product_usage/`. Observation receives only
validated aggregate lifecycle summaries after stable boundaries. It must not
read task content, result payloads, prompts, logs, paths, credentials, provider
details, or hardware details, and no capture/upload failure may alter a run
result, schedule, artifact, exit code, or control-flow decision.

## 3. Core Boundary

Core code belongs in `praxist/core`.

Core owns contracts and control-plane services:

- protocol dataclasses and event shapes;
- plugin discovery and resolution;
- credential references and redaction;
- model/provider profile contracts;
- budget requests, grants, ledgers, and usage records;
- prompt layout manifests and cache provenance;
- trajectory, artifact, finding, replay, and source snapshots;
- task project resolution;
- workflow stage interfaces;
- tool server interfaces;
- runtime interfaces;
- role skill loading;
- stable storage views.

Core must not contain task facts.

Core must not contain benchmark-specific prompts, roles, evaluations, audits, or
harness code.

Core must not shell out to a specific agent runtime directly. Runtime execution
goes through an `AgentRuntime` plugin.

Core must not make provider-specific API calls directly. Provider selection and
credential binding go through `ModelProvider` and `CredentialResolver` layers.

Core must not read a real task unless startup received an explicit task path.

## 4. Plugin Boundary

Generic reusable plugins belong under `praxist/plugins`.

Current plugin kinds include:

- `agent_runtime`;
- `model_provider`;
- `workflow_stage`;
- `tool_server`;
- `graph_maintainer`;
- `budget_policy`;
- `panel_topology`;
- generic `audit_rule`;
- generic `evaluation`.

Plugins are reusable system components. They are not user projects.

Plugin manifests use `plugin.yaml`.

Executable plugins declare an importable entrypoint in the manifest.

The code implementing a plugin must physically live inside the selected plugin
directory unless the plugin is explicitly manifest-only.

Manifest-only plugins are allowed only when the manifest is the complete
contract. Do not use manifest-only plugins as placeholders for missing
production code.

## 5. Task Project Boundary

Task projects are explicit external inputs.

Real task projects should normally live outside this repository. Curated complete
reference projects may be published under `examples/`, but they are not system
plugins or generic defaults.

On first use of Praxist for a real task, create or copy the task project outside
the Praxist checkout before testing or starting a run. Do not write environments,
caches, run artifacts, or other generated files inside `praxist/`, `templates/`,
`examples/`, or another tracked Praxist source directory. A bundled example may
be inspected in place, but it must be materialized outside the checkout before
executing its tests, evaluators, or research workflow.

Local dogfood task projects may live under the ignored root `tasks/` directory.

Tracked task templates may live under `templates/tasks/`, but they must stay
small and authoring-oriented.

The system must not discover real tasks implicitly.

Valid task startup requires:

```bash
praxist start --task-path /path/to/task-project --daemonize --json
```

A task project normally contains:

```text
task.yaml
description.md
prompt_task.jinja2
roles/
audit_rules/
evaluations/
budget_profiles/
assets/
tests/
README.md
```

Task-local components use task-local refs such as `task_role:*`,
`task_audit:*`, and `task_evaluation:*`.

Task-local roles, audits, evaluations, budget profiles, harnesses, literature
packs, baselines, and benchmark code must stay in the task project.

Task projects must not add an alternate research launcher. The `praxist start`
and `praxist resume` commands own detached execution, run registration, logs,
credentials, plugin discovery, budgets, replay, and workflow semantics.

Tasks that require a dedicated virtual environment should first express that in
`task.yaml` via `runtime_environment`:

```yaml
runtime_environment:
  cwd: task_project
  venv: .venv
  # python: .venv/bin/python  # optional; inferred from venv when omitted
  path_prepend:
    - bin
  env:
    TASK_MODE: dogfood       # non-secret task vars only
```

Praxist validates those paths and injects `PRAXIST_TASK_PYTHON`, `PRAXIST_TASK_VENV`,
`VIRTUAL_ENV`, `PRAXIST_TASK_SHELL_PREFIX`, and `PATH` into peer sessions. Task
prompts, harnesses, and evaluations should call
`"${PRAXIST_TASK_PYTHON:-python}"` rather than hard-coding `python`.

Keep dependency installation as a separate task setup step. Runtime values
required by peer commands belong in `runtime_environment`, not in an alternate
startup path.

Do not add SAM-specific components to `praxist/plugins`.

Do not add a new task by editing many global plugin directories. Add the task as
an external project and reference generic plugins from `task.yaml`.

## 6. Templates And Complete Examples Boundary

Tracked task templates are documentation assets, authoring scaffolds, and
smoke-test fixtures.

They are not the production task catalog.

Each template directory must explain:

- what it demonstrates;
- which files are required;
- which files are optional;
- how to run it;
- how to test it;
- what should not be committed.

Task templates must not contain:

- large datasets;
- private benchmark data;
- raw API keys;
- real long-run outputs;
- task-local `experiments/` directories;
- uncurated run artifacts;
- uploaded deliverables from dogfood runs.

Complete examples under `examples/` serve a different purpose. They are curated,
runnable reference projects that may include a complete task harness, small
frozen data, measured evidence, and task-local tests. They must remain isolated
from Praxist core and generic plugins, document all task-specific assumptions,
exclude runtime outputs and credentials, and include integrity tests for any
frozen evidence. An example is not a scaffold and its domain policy must never
be copied into generic system behavior.

## 7. Runtime Boundary

Agent runtimes execute `AgentRunRequest` and return normalized `AgentRunResult`
or event streams.

Runtime adapters must normalize:

- text output;
- tool calls;
- tool results;
- timeout status;
- cancellation status;
- usage records when available;
- cache usage when available;
- terminal status;
- redacted error details.

Runtime adapters must not leak provider-private response objects into
trajectory.

The production runtime today is:

```text
agent_runtime:claude_sdk
```

The tested runtime dependency baseline is
`claude-agent-sdk==0.2.136`. The optional Codex runtime uses
`openai-codex==0.147.0`; update these pins only with runtime conformance
validation.

`agent_runtime:codex_sdk` is an explicitly selected peer runtime implemented
inside its plugin boundary with the official `openai-codex` Python SDK. It uses
long-lived local app-server clients, direct MCP tool servers, and normalized
typed notifications. OpenAI connects directly; supported Chat Completions
providers such as DeepSeek and OpenRouter use a private run-scoped
`codex-relay` adapter to present the Responses interface expected by Codex.

The Codex CLI used by a human operator to invoke bundled Praxist skills is
separate from peer runtime execution. Do not couple that CLI's interaction
state or process lifecycle to an `AgentRuntime` implementation.

Adding a runtime means adding the plugin, conformance tests, and docs. Do not
special-case it in core startup.

## 8. Model Provider Boundary

Model providers describe API format, endpoint, model defaults, credential
requirements, and cache capability.

Provider adapters are separate from runtime adapters.

A runtime may use one or more provider profiles.

A run may use different models for different stages when task or override config
selects them.

Provider refs include examples such as:

```text
model_provider:openrouter
model_provider:openai_compatible
model_provider:anthropic_messages
model_provider:deepseek_alias
```

Do not hard-code a model name in generic plugin code unless it is declared as a
provider default and overridable through startup or task configuration.

## 9. Credential Boundary

Raw secrets are read only by Python credential resolution code.

Bash wrappers must not parse, truncate, sort, print, or fallback raw keys.

Logs, trajectory, replay output, and generated docs must never include raw
secrets.

Credential output should use redacted `CredentialRef` objects.

Single-key quickstart is supported.

Multi-key robust mode is supported when multiple keys are detected for the same
provider or profile.

Provider failover and credential cooldown belong in Python, not shell scripts.

Tool-scoped credentials, such as Semantic Scholar or Crossref keys, must be
declared separately from model provider keys.

## 10. Budget Boundary

Budget is dynamic.

Agents may request compute, wall-clock, token, data, and tool budget per stage or
per action.

BudgetPolicy may auto-grant small requests, downscope requests, or route large
requests to PI/Chair review.

BudgetPolicy must serve the research result preservation principle. It should
avoid killing promising experiments just because a static global number was
exceeded, unless the run would damage the fact chain, leak secrets, or consume
resources outside an approved envelope.

Usage may be exact, estimated, or unknown.

Unknown usage must be recorded as `usage_unknown`, not silently stored as zero.

Late accounting failure should degrade to warnings when possible so completed
peer findings remain available.

## 11. Result Preservation Principle

Praxist must maximize the chance that useful peer outputs are captured.

This is a product principle, not only a runtime detail.

Do not harden the system in a way that prevents peers from finishing promising
work unless there is a concrete integrity, secret, safety, or irrecoverable
state risk.

Replay, audit, attribution, and budgets should explain uncertainty rather than
drop results.

Weak provenance is acceptable when clearly labeled.

Partial run artifacts are valuable when they preserve facts.

The default posture is:

```text
capture first, label uncertainty, continue when safe
```

## 12. Workflow Stage Boundary

`research_loop` is the mandatory workflow stage.

Optional stages include:

- `ideation_stub`;
- `paper_writing_stub`;
- `reviewer_stub`.

Optional stages may be disabled without changing `research_loop` semantics.

Optional stubs are interface placeholders. They are not complete product
modules.

Workflow stages should emit stage lifecycle events and write replayable
artifacts.

Stage logic belongs in workflow stage plugins, not in bash wrappers or task
harnesses.

## 13. Research Loop Status

The current research loop still contains migrated compatibility code.

That compatibility code is now physically inside the `workflow_stage:research_loop`
plugin boundary.

This is acceptable while the stage preserves old behavior and writes new
control-plane artifacts.

Do not move research-loop implementation back into core.

When replacing legacy internals, keep characterization tests and parity tests
for:

- peer lifecycle;
- generation close;
- stop signal drain;
- frontier promotion;
- PI/Chair agenda;
- finding graph guidance;
- budget accounting;
- result preservation;
- replay inspection.

## 14. Finding Graph Boundary

The finding graph is not only a visualization.

It provides advisory context that influences future peer and PI decisions.

Graph maintainers belong in `praxist/plugins/graph_maintainers`.

Graph query tools belong in `praxist/plugins/tools/finding_graph_query`.

The graph must not rewrite raw findings or frontier ranking.

The graph may produce guidance context, edge ledgers, health reports, and
visualizations.

Future KGE implementations should enter as graph maintainer strategies or
plugins, not as task-specific core code.

## 15. Prompt Layout Boundary

Prompt construction uses PromptLayout V1.

Prompt layout separates:

- frozen blocks;
- semi-static blocks;
- dynamic blocks;
- legacy-rendered blocks when needed.

Frozen-prefix hashes exist to make cache behavior observable and stable.

Agent CLI prompt cache behavior is runtime-managed. Praxist stabilizes structure
and records cache provenance; it does not pretend to force runtime cache
control when the SDK does not expose that raw interface.

Do not put generation-specific, peer-specific, or result-specific facts in frozen
blocks.

Do not remove layout manifests from run artifacts.

## 16. Startup Contract

`praxist start` and `praxist resume` are the only supported research process
launch paths. They own task and run resolution, credentials, provider/model
routing, plugin discovery, budgets, detached process creation, logs, registry
state, trajectory setup, and replay metadata.

Task projects declare runtime values through `runtime_environment` and keep
dependency installation as a separate setup step. They must not add an
alternate startup wrapper. Source-checkout operators use `uv run praxist ...`
so the same CLI contract applies without installing the package globally.

## 17. Run Artifacts

Run artifacts are part of the product surface.

For real task runs, `run_dir` must live outside the Praxist source checkout. When
the operator does not pass `--run-dir`, Python selects the task project's
`runtime_outputs.root` or task-local `experiments/` directory. The startup path
must reject run directories inside the Praxist repository.

Important files include:

```text
run.json
startup_config.json
task_project_manifest.json
plugin_resolution.json
trajectory.jsonl
budget_ledger.jsonl
artifact_index.jsonl
credentials_redacted.json
run_summary.json
prompt_layouts/
findings/
frontier/
research_memory/
replay/
```

Some legacy-compatible runs may still include additional operational files such
as `shared_store.db`, `shared_findings/`, `STOP_SIGNAL`, and graph artifacts.

Do not delete operational files during a live run.

Do not assume `run_dir` is invalid because one surface is incomplete. Prefer
recovery, import, warning, and labeling.

## 18. Testing Contract

Tests are organized by long-lived system responsibility, not by historical
refactor step.

Current top-level test areas include:

```text
tests/core/
tests/conformance/
tests/integration/
tests/workflows/
tests/hardening/
tests/legacy_migration/
tests/plugins/
tests/adversarial/
tests/fixtures/
```

Use offline fixtures whenever possible.

Do not require real API keys, network, external task repos, or GPUs in the
default unit or offline integration test path.

Conformance tests should cover plugin contracts.

Workflow smoke tests should cover startup, fixture run, replay, and artifact
shape.

Offline integration tests should copy tracked task templates into temporary
external paths and then verify startup, plugin resolution, workflow execution,
run artifacts, and replay through public entrypoints. Complete examples should
have focused tests for their frozen files, manifests, task contracts, and
absence of bundled run state.

Hardening tests should prevent regressions in boundaries, docs, redaction, and
repository hygiene.

New core services, plugins, runtime adapters, provider adapters, tool servers,
workflow-stage code, task-template contracts, and operator wrappers must add or
update tests at the same boundary. Prefer long-lived contract tests over tests
that assert incidental implementation shape.

Coverage is part of the default engineering contract. The unit profile is gated
at 90% branch-aware total coverage and 95% statement coverage; integration
coverage is reported but not thresholded.

Run the default suite before pushing:

```bash
python -m unittest discover -s tests -q
python scripts/run_test_coverage.py unit --fail-under 90 --fail-under-statements 95
python scripts/run_test_coverage.py integration
python -m compileall -q praxist tests templates examples scripts
uv run python scripts/build_docs_site.py
git diff --check
```

When a change is intentionally narrow, run the narrow tests first, then the full
suite before final handoff.

## 19. Documentation Contract

Public and semi-public Python APIs need Google-style docstrings.

Private helpers need comments only for non-obvious invariants, recovery
semantics, boundary behavior, or failure policy.

Do not write patch-note comments such as "Round 6 fix" in long-lived docstrings.

Do not add improvement plans, implementation logs, or one-off analysis reports
to `docs/`. Keep durable docs focused on current contracts and user/operator
guides. Use commits, pull-request descriptions, and issue trackers for change
history.

The docs site is built with:

```bash
uv sync --extra docs
uv run python scripts/build_docs_site.py
uv run python scripts/build_docs_site.py --check-generated
```

The build must not read API keys, hit model providers, start MCP servers, use
GPUs, start SAM, or require an external task checkout.

The Markdown under `docs/` is the only authored documentation source. Do not
create a separately edited Wiki or committed static-site tree. CLI reference is
generated from the live parser, Skills reference is generated from `SKILL.md`
front matter, and LLM exports are generated into `site/`.

Every source Markdown page must appear exactly once in `mkdocs.yml`. Update the
navigation when adding or moving a page.

Rebuild the local docs site after changing docs or docstrings. Generated HTML
is not committed in normal pull requests.

## 20. How To Add A Generic Plugin

1. Pick the plugin kind.
2. Create a directory under `praxist/plugins/<kind_plural>/<name>/`.
3. Add `plugin.yaml`.
4. Add executable Python code only when the plugin has runtime behavior.
5. Declare entrypoints and code files in the manifest.
6. Add conformance tests under `tests/conformance/`.
7. Add workflow or plugin tests when startup uses the plugin.
8. Update docs if the plugin introduces a new contract or operator setting.

Do not add a task-specific component as a generic plugin.

Do not make core import a specific plugin implementation.

## 21. How To Add A Task Project

1. Before the first run, create or clone the task outside the Praxist repo, or under
   ignored `tasks/`; never start a real task from tracked Praxist source
   directories.
2. Start from `templates/tasks/template` when possible.
3. Fill in `task.yaml`.
4. Add task-local roles, audits, evaluations, budget profiles, and assets.
5. Declare `runtime_environment` in `task.yaml` when the task needs a venv,
   specific Python executable, cwd, PATH additions, or non-secret task env vars.
6. Ensure task scripts use `"${PRAXIST_TASK_PYTHON:-python}"` for Python
   subprocesses.
7. Keep run outputs under the task-local ignored `experiments/` directory.
8. Add task-local tests.
9. Run Praxist with `--task-path`.

If the task is meant to be an authoring scaffold or deterministic smoke fixture,
put a minimal copy under `templates/tasks/` and document what is intentionally
omitted. Add a project under `examples/` only when it is a complete runnable
reference with redistributable assets, no run history or credentials, a clear
license/provenance boundary, and focused integrity tests.

## 22. How To Add A Runtime Adapter

1. Add or update an `agent_runtime` plugin.
2. Implement `AgentRunRequest` translation.
3. Normalize runtime events.
4. Normalize terminal status.
5. Record usage when available.
6. Mark usage unknown when unavailable.
7. Redact provider-private or secret-bearing data.
8. Add fake or offline conformance tests.
9. Add at least one failure-mode test.
10. Update the runtime guide.

Do not let runtime-specific response objects enter trajectory.

## 23. How To Add A Model Provider

1. Add a `model_provider` plugin.
2. Document supported API format.
3. Declare credential refs and environment variables.
4. Declare cache capability.
5. Declare model defaults, if any.
6. Support explicit operator override.
7. Add provider conformance tests.
8. Add docs and cost notes when pricing or token semantics differ.

Do not assume OpenAI-compatible providers all support the same cache, streaming,
error, or usage shape.

## 24. How To Add A Tool Server

1. Add a `tool_server` plugin.
2. Keep the tool generic.
3. Declare tool names and handler entrypoints.
4. Normalize handler success and failure output.
5. Add tool-scoped credential docs when needed.
6. Add conformance tests.
7. Add workflow smoke coverage if the tool is exposed to agents by default.

Task-specific tool instructions belong in the task project prompt or role skill,
not in a generic tool implementation.

## 25. How To Add A Budget Policy

1. Add a `budget_policy` plugin.
2. Define auto-grant thresholds.
3. Define downscope behavior.
4. Define review routing for large requests.
5. Record decisions in budget ledgers.
6. Preserve partial results whenever possible.
7. Add tests for grant, deny, downscope, and unknown usage.

Budget policy should not be used as a hidden task-specific research strategy.

## 26. Migration Rules

When migrating legacy behavior:

1. Add characterization tests for old behavior.
2. Add the new interface or plugin boundary.
3. Add parity tests comparing old and new surfaces.
4. Move implementation only after behavior is locked.
5. Keep compatibility shims local and temporary.
6. Delete obsolete paths when the new path is proven.
7. Record implementation context in the commit or pull-request description.

Do not keep two active implementations without a clear owner and deprecation
path.

## 27. Code Style Rules

Prefer small, explicit modules.

Prefer structured parsers over ad hoc string parsing.

Prefer dataclasses or typed dictionaries for protocol data.

Keep comments stable and contract-oriented.

Do not introduce large abstractions unless they remove real coupling.

Do not add unrelated refactors while fixing a narrow bug.

Do not commit generated caches, run outputs, API keys, or local virtual
environment files.

## 28. Operational Commands

Start an external task project:

```bash
uv run praxist start --task-path /path/to/task-project --daemonize --json
```

Start with an installed CLI:

```bash
praxist start --task-path /path/to/task-project --daemonize --json
```

Inspect a run:

```bash
python -m praxist.run replay /path/to/run-directory --mode inspect
```

Stop Praxist runtime processes:

```bash
praxist status --json
praxist stop <run-id>
praxist stop --all
```

Build docs:

```bash
uv run python scripts/build_docs_site.py
```

Run tests:

```bash
python -m unittest discover -s tests -q
python scripts/run_test_coverage.py unit integration
```

## 29. Pre-commit Guardrails

Before every commit and before opening a PR, run the guardrail commands.
### Test And Coverage

The default source-checkout test path is offline and does not require model API
keys, network access, GPUs, S3, RunPod, or an external task checkout.

```bash
python -m unittest discover -s tests -q
python scripts/run_test_coverage.py unit --fail-under 90 --fail-under-statements 95
python scripts/run_test_coverage.py integration
```

Coverage uses `coverage.py` and writes ignored local reports to `cover/unit/`
and `cover/integration/`. The `unit` profile covers the offline
non-integration test layers; the `integration` profile covers
`tests/integration`. The unit profile is held at 90% branch-aware total coverage
and 95% statement coverage; integration coverage is reported but not
thresholded. Install dev dependencies first when running in a fresh environment:

```bash
uv sync --group dev
```

---

CI (`.github/workflows/ci.yml`) runs the exact same commands on every push and
pull request; the local helper in `scripts/dev/run_guardrails.py` mirrors them
so dev and CI never disagree.

```bash
uv run ruff check praxist/ tests/
uv run ruff format --check praxist/ tests/
uv run pyrefly check
uv run pytest tests/unit --tb=short -q
uv run python scripts/run_test_coverage.py unit --fail-under 90 --fail-under-statements 95
uv run python scripts/run_test_coverage.py integration
```

Or run them all at once via the tracked dev helper:

```bash
python3 scripts/dev/run_guardrails.py
```

Agent CLI users can wire that helper into the per-tool hook by
symlinking it under the gitignored `.agent/` tree, then pointing
`.agent/settings.local.json` at the symlink:

```bash
mkdir -p .agent
ln -s ../scripts/dev/run_guardrails.py .agent/run_guardrails.py
ln -s ../scripts/dev/leakage_audit.sh  .agent/leakage_audit.sh
```

The companion `scripts/dev/leakage_audit.sh` is a separate boundary
gate: it greps `praxist/` and `tests/` for task-specific tokens
that must stay confined to the external task project. Customize the
`TOKENS` array in the script for your active dogfood task. Lines that
intentionally document a forbidden word can be exempted with a
trailing `# noqa: leakage_audit` marker.

The coverage command runs the offline non-integration unit profile and the
integration profile through `unittest`, then writes per-profile reports under
`cover/unit/` and `cover/integration/`. The unit profile is ratcheted at
90% branch-aware total coverage and 95% statement coverage; integration
coverage is still observational because it asserts cross-component behavior
rather than broad line coverage. Coverage reports are local artifacts and must
not be committed. For local report generation without a gate,
`python scripts/run_test_coverage.py unit integration` remains supported.

If any check fails, fix the failure rather than skipping the hook. Use
`--no-verify` only for explicitly out-of-scope cleanup commits, and explain
the rationale in the commit body.

Pyrefly excludes a small set of legacy files with pre-existing type debt
(see `[tool.pyrefly].project-excludes` in `pyproject.toml`). When a
backfill PR edits one of those files, the same PR must remove it from the
exclude list and fix its type errors.

Ruff `SIM117` is per-file-ignored on two legacy hardening test files for
the same graduation reason (see `[tool.ruff.lint.per-file-ignores]`).

## 30. Push Discipline

Keep changes scoped.

Check `git status` before committing.

Do not revert user changes unless explicitly asked.

Do not rewrite history unless the user explicitly asks.

Push completed work to the active branch by default.

## 31. High-Risk Changes

Treat these as high-risk:

- changing startup argument semantics;
- changing plugin discovery priority;
- changing task project resolution;
- changing credential selection;
- changing prompt layout frozen blocks;
- changing event-driven peer scheduling;
- changing generation close behavior;
- changing finding graph guidance;
- changing replay verification defaults;
- changing budget enforcement;
- moving code across core/plugin/task boundaries.

High-risk changes need focused tests and a concise implementation note in the
commit or pull-request description.

## 32. What Not To Do

Do not put a real task inside `praxist/plugins`.

Do not add a new canonical package name.

Do not add a compatibility `praxist/` or `auto_research/` package.

Do not make bash own run semantics.

Do not hard-code SAM as the only research-loop task.

Do not make docs depend on real API calls.

Do not make default tests require GPUs or model keys.

Do not drop peer results because provenance is imperfect.

Do not log raw secrets.

Do not add long-lived comments that only explain a past patch.

## 33. Source Of Truth Order

When sources disagree, use this order:

1. Current code and tests.
2. `AGENTS.md`.
3. Active design docs in `docs/concepts/` and `docs/guides/`.
4. Generated docs built from current docstrings.
5. README and guide pages.

If a guide contradicts code, fix the guide or the code in the same change.

## 34. Source Control

The integration branch is `main`. All work merges into `main` through a
pull request. Releases are git tags. Long-lived personal development
branches are not used.

### Branch model

- `main` is the only long-lived development branch.
- Releases are annotated semantic-version tags on `main`.
- `hotfix/<version>` branches may be cut from a release tag when a
  published version needs a patched re-release.
- Personal long-lived branches are not used. Active work lives on short-lived
  feature branches.

### Branch naming

- `feat/<slug>` — new feature work
- `fix/<slug>` — bug fix
- `docs/<slug>` — documentation-only changes
- `chore/<slug>` — repository hygiene, dependency bumps, scripts
- `refactor/<slug>` — code-only refactor without behavior change
- `hotfix/<version>` — patch branch cut from a release tag
- `wip/<author>/<slug>` — personal work-in-progress; force-push is
  allowed; no pull request is expected; other contributors do not
  depend on these branches
- `archive/<slug>` — frozen reference branch for historical work

Shared branches do not carry author initials. `wip/<author>/*` is the
only allowed personal namespace.

### Merge style

- Default: squash merge. One feature branch produces one clean commit
  on `main`.
- Exception: documentation-only changes, mass renames, and tooling
  chores may use a merge commit when sub-commit history aids review.
- Rebase merge is not used.

### Main branch protection

`main` enforces:

- No direct push; all changes go through pull request.
- No force-push.
- No deletion.
- Required approving reviews follow the repository's active branch-protection
  settings.
- Required status checks must pass.
- Branch must be up-to-date with `main` before merge.
- Linear history (the squash commit is the only merge style accepted).
- Required conversations resolved before merge.

### Force-push policy

- `main`: never.
- Release tags: never rewrite or delete published tags.
- `feat/*` / `fix/*` / `docs/*` / `chore/*` / `refactor/*`: only the
  branch owner, and only before review begins. After a reviewer
  comments, rebase by adding commits rather than rewriting history.
- `wip/<author>/*`: any time.

### Daily rebase

Each open feature branch rebases onto the latest `main` at the start of
every working day to surface conflicts early.

### Feature freeze

The day before a planned release tag is a feature freeze. No new
feature merges into `main` during freeze. The freeze window runs about
24 hours of integration testing before the release tag is cut.

### Stale branch lifecycle

- Pull requests delete their head branch on merge.
- Feature branches inactive more than two weeks are evaluated by the
  owner for archive or deletion.
- Local branches whose remote tracking shows `[gone]` are pruned with
  `git fetch --prune` and `git branch -d`.

### Release artifacts

- Releases are git tags, not branches.
- Tag pattern: `<major>.<minor>.<patch>`.
- Release notes live in the GitHub release attached to the tag.
- A release branch is cut only if a published release needs a hotfix
  series.

### Documentation rebuild policy

- The local MkDocs site is regenerated by
  `uv run python scripts/build_docs_site.py`.
- Pull requests should keep source docs current and do not commit generated
  HTML output.
- Release publication may build static docs from source as part of the release
  process, but generated site files are not part of the normal source tree.

### Code ownership

`.github/CODEOWNERS` is the authoritative ownership map. Coordinate ownership
before parallel changes to the same files.
