---
name: praxist-takeover-codex
description: Onboard, initialize or repair, validate, and launch a Praxist research task in Codex-native mode through the official Codex SDK runtime and the operator's existing saved ChatGPT login, with catalog-verified gpt-5.6-luna as the default model and without requesting, storing, or using an API key. Use when a user wants a low-interaction no-key Praxist takeover, explicitly requests Codex-native mode, has no provider key, or invokes this skill with no additional text after already logging in to Codex.
---

# Praxist Takeover: Codex-native mode

Run the complete Praxist takeover workflow while making Codex-native mode the
explicit operator choice. The mode uses the operator's saved ChatGPT login. The user invokes this
skill once; the current agent performs the checks and shell operations. Do not ask the user
to type Praxist commands, provider names, model names, or API keys.

## Composition Contract

Load the currently installed `praxist-takeover` `SKILL.md` and follow its
complete current workflow. In particular:

- run complete onboarding first;
- use the complete current task-initialization or repair workflow when needed;
- preserve evaluator, protocol, lane-routing, resource, and launch-readiness
  checks;
- preserve the user's explicit protocol-intent permissions, including an
  intentionally reduced or incomplete authoritative mode; Codex-native mode
  changes runtime/authentication only and must not strengthen task semantics;
- preserve the task-initialization closing-policy regression and positive
  mature normal-close gate whenever the task defines close-grade evidence;
- validate with `praxist resolve`;
- launch detached, confirm status, and report the independent monitor command.
- preserve the common `agent.reasoning_effort` policy; use `max` unless the
  user explicitly selected another supported value.

Do not copy or reconstruct an abbreviated takeover from memory. This skill
overrides only authentication/runtime selection and the unambiguous-path
confirmation behavior described below. If a base takeover instruction conflicts
with this file on those points, this file wins.

Treat invocation of `praxist-takeover-codex` as the user's explicit choice of:

```text
agent runtime: agent_runtime:codex_sdk
agent system: codex_sdk
model provider: model_provider:openai_compatible
model: gpt-5.6-luna unless the user explicitly selects another model
mode: Codex-native mode
authentication: saved ChatGPT login
API-key use: disabled for the Praxist model connection
```

Carry that choice through task initialization, generated launch guidance, the
actual start command, and the final summary. Do not replace it with the normal
DeepSeek/Claude default merely because task initialization recommends that
profile for users who have a DeepSeek key.

Codex-native mode automatically enables Praxist's lossless context
efficiency. Do not add a task-local compressor, memory database, persistent
Codex conversation, or relay. Do not ask the user to configure the optimization.
After launch, state that finding-only wakeups are batched while lifecycle and
resource events remain immediate, and include the diagnostic skill for
measuring session count and cached versus uncached input.

## Preconditions

The research project must already run on this machine with its required code,
data or simulator, and task environment. Apply the base takeover refusal rules
when those task prerequisites are missing.

Praxist must provide the Codex runtime extra. In the selected Praxist Python
environment, verify that `openai_codex` and `mcp` import successfully and that
the runtime distributions are `openai-codex==0.147.0` and
`claude-agent-sdk==0.2.136`. If they are missing or mismatched, use
`praxist-runtime-install` to install the tested runtime dependencies only; do
not enter provider-key configuration and do not ask the user for a key.

Praxist and the research project may intentionally use different Python
environments. Run Praxist CLI/runtime checks with the Praxist interpreter. Run
evaluator and harness tests with the task interpreter; when that interpreter
does not install Praxist, expose the selected Praxist source/package root only
for that test subprocess through `PYTHONPATH`. Do not install Praxist into the
task environment or create another environment merely to make a harness test
import it.

## Codex-native Preflight

Use the same Python environment and Praxist installation that will launch the
run. First run the route-aware host gate:

```bash
praxist doctor --codex-native --task-path /absolute/path/to/task --json
```

Add `--model <selected-model>` when the user explicitly chose a model other
than the profile default.

It must report `agent_runtime:codex_sdk`,
`model_provider:openai_compatible`, `auth_mode: codex-native`, ready SDK/MCP
dependencies, a valid saved login, and `codex_model_catalog: ok` for the
selected model. This single gate uses the SDK-bundled Codex binary, performs a
bounded login probe, and reads the account model catalog without making a
research request. Do not duplicate it with a second model-catalog probe. Never
read `auth.json` directly or print tokens, account identifiers, or credential
contents. A successful login probe means the SDK-bundled binary reported
exactly `Logged in using ChatGPT`.
Ignore `PRAXIST_CODEX_BIN` only for this probe and launch so a stale custom
binary override cannot make preflight and execution disagree. Do not alter the
operator's persistent setting.

The default model for this skill is `gpt-5.6-luna`. Preserve a model explicitly
selected by the user and pass it to the doctor gate. Never infer a different
model from stale task documentation, shell variables, a previous resolve
snapshot, or memory. When login succeeds but the selected model is absent from
the account catalog, stop before resolve/start and report the available model
identifiers; do not launch a cohort to discover the mismatch.

When the login probe succeeds, continue without asking any authentication
question. When it fails, stop before task launch and ask the operator to run
`praxist setup --profile codex-native --install-skills codex` in a local
interactive terminal. That explicit setup route opens and verifies the
SDK-pinned Codex login flow. Do not offer an API key as an automatic fallback,
do not switch providers, and do not invoke a different Codex executable.

## Task-Initialization Overrides

Keep the complete current task-initialization workflow. For a task newly
created through this Codex-native entry only:

- do not launch until the current task-init evaluator fan-out preflight has
  passed, including its task-defined one-unit canary and any explicitly
  declared external-evidence trust check;

- generated resolve/start instructions must use the selected Codex runtime,
  provider, and model rather than the standard DeepSeek/Claude recommendation;
- use `dig_lite.contract.min_rejected_alternatives: 2` unless the user already
  chose a stricter value, while preserving candidate count, mechanism and
  intervention diversity, selected-contract, ablation, forbidden-change,
  metric-signature, fail-fast, validator, and retry requirements;
- do not create a task-local `agent_runtime:codex_sdk` shadow, relay, transport,
  or model-specific schema adapter. Praxist owns runtime compatibility.

Do not silently rewrite the scientific contract of an existing task merely
because it uses a stricter rejection count. Repair it only when the current
task-init workflow is already repairing that harness or a bounded preflight
shows the stricter task-owned value is not stable for the selected model.

## Low-Interaction Task Selection

Use the user-provided project/task path when present; otherwise start from the
Codex invocation directory.

Invocation of this skill is sufficient authorization to use a task path when
one of the following resolves unambiguously:

- the current directory is itself one valid task project; or
- complete task initialization just created one task project for the current
  research project; or
- a bounded local scan finds exactly one launch-ready task belonging to the
  current research project.

Print the absolute path before launch, but do not ask for a redundant
confirmation in those unambiguous cases. Ask one concise path-selection
question only when multiple plausible task projects remain, the candidate is
outside the current research project, or ownership cannot be established.
Never guess among multiple active research projects.

All scientific and safety confirmation gates from task initialization remain in
force. This low-interaction rule removes only redundant task-path confirmation;
it does not authorize inventing missing objectives, data, simulators, evaluators,
or baseline evidence.

## Codex-native Resolve

Run every resolve with the same sanitized environment and explicit runtime,
provider, and model that will be used for start:

```bash
env -u OPENAI_API_KEY \
    -u CODEX_API_KEY \
    -u CODEX_ACCESS_TOKEN \
    -u OPENAI_BASE_URL \
    -u PRAXIST_CODEX_BIN \
    -u MODEL \
    -u PRAXIST_MODEL \
  praxist resolve /absolute/path/to/task \
    --codex-native \
    --runtime agent_runtime:codex_sdk \
    --model-provider model_provider:openai_compatible \
    --model gpt-5.6-luna
```

Use the user-selected model in place of Luna when applicable. A successful
resolve creates a run-like artifact snapshot whose exit condition is
`resolve_only`; report it as configuration validation, not as a research run or
failed launch. Track actual starts only from `praxist start` output.

The inherited task-initialization workflow must also run its real evaluator
summary contract probe. Add `--result-summary <actual_summary_path>` to the same
sanitized resolve command. When the task requires maturity ratios, do not start
if this check cannot extract finite `effort_ratio` and `coverage_ratio`; repair
the task evaluator or follow the explicit user-approved legacy fallback from
task initialization. A stage label is not a substitute for ratio telemetry.
The inherited workflow must also preserve its optional effective-configuration
provenance check for configuration-sensitive treatments; Codex-native mode does
not weaken that check or turn it into a gate for ordinary experiments.

## Codex-native Launch

Do not persist provider settings or alter the user's shell configuration. Build
one launch subprocess whose inherited environment omits these variables:

```text
OPENAI_API_KEY
CODEX_API_KEY
CODEX_ACCESS_TOKEN
OPENAI_BASE_URL
PRAXIST_CODEX_BIN
```

Removing them only from the launch subprocess is required even when onboarding
reports that they are present. This prevents API-key billing or a custom endpoint
from taking precedence while preserving the user's environment for unrelated
work. Do not unset unrelated task credentials.

Start through the selected Praxist executable with explicit matching runtime and
provider flags:

```bash
env -u OPENAI_API_KEY \
    -u CODEX_API_KEY \
    -u CODEX_ACCESS_TOKEN \
    -u OPENAI_BASE_URL \
    -u PRAXIST_CODEX_BIN \
    -u MODEL \
    -u PRAXIST_MODEL \
  praxist start \
    --codex-native \
    --task-path /absolute/path/to/task \
    --agent-system codex_sdk \
    --runtime agent_runtime:codex_sdk \
    --model-provider model_provider:openai_compatible \
    --model gpt-5.6-luna \
    --daemonize \
    --json
```

The command is a behavioral template, not a request for user input. The current agent must
execute it. On platforms without `env -u`, construct an equivalent child
environment with only the listed entries removed.

Pass the selected model explicitly. The default is `gpt-5.6-luna`; preserve an
explicit user selection after the account-catalog probe. Remove `MODEL` and
`PRAXIST_MODEL` from the launch subprocess so stale defaults cannot override
the verified choice.
Preserve explicit cohort, generation, strategy, and run-directory choices when
they do not conflict with Codex-native authentication.

Do not use `praxist configure-llm`, a relay provider, `OPENAI_API_KEY`, DeepSeek,
OpenRouter, or Anthropic for this launch. Saved ChatGPT authentication is valid
only with native `model_provider:openai_compatible` through
`agent_runtime:codex_sdk`.

## Post-Launch Verification

Treat a successful launcher response as startup-confirmed. Parse the JSON
response, then use `praxist status --run-id <run_id> --json` and bounded log
inspection to confirm all of the
following:

- the selected task path matches the intended task;
- runtime is `agent_runtime:codex_sdk`;
- provider is `model_provider:openai_compatible`;
- model is the account-catalog-verified selection;
- the daemon remains alive and did not report saved-login or account-type
  failure;
- no fallback provider or relay was selected.

Praxist independently verifies that the app-server account type is `chatgpt`.
If startup fails that check, report the failure and stop. Never retry with an API
key or another provider.

## Final Handoff

End with a compact table containing:

| Item | Required value |
|---|---|
| Research project | absolute path |
| Task path | absolute path |
| Run id / PID / state | values from start and status |
| Runtime | `agent_runtime:codex_sdk` |
| Provider | `model_provider:openai_compatible` |
| Model | selected catalog-verified model; default `gpt-5.6-luna` |
| Authentication | `saved ChatGPT login` (never include account data) |
| API-key use | `disabled for this launch` |
| Run directory | absolute path |
| Monitor | `praxist --monitor --run-id <run_id>` |

Do not open the foreground monitor automatically. Clearly state that the run is
detached and continues after the current operator-agent session closes.
Use `praxist status --json` and filter its returned records by `run_id`; do not
scan unrelated rows when the direct selector is available. Prefer
`praxist status --run-id <run_id> --json`. The foreground TUI selector remains
`praxist --monitor --run-id <run_id>`.
