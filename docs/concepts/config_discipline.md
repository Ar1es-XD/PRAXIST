# Configuration Discipline

> Core and plugin domain code consume explicit configuration objects. Ambient
> environment variables are read at operator entry boundaries, then resolved
> once into frozen runtime configuration.

`RunConfig` centralizes run-critical settings. Narrow compatibility paths still
read ambient variables, and selected launcher or scheduler boundaries may pass
documented host values into child processes.

This contract keeps startup, replay, API provider routing, credential handling,
and agent runtime execution deterministic.

## Configuration Flow

```text
CLI entry
  argparse + selected environment values
                  |
                  v
       frozen RunConfig / domain contracts
                  |
                  v
         core + resolved plugins
                  |
                  v
 runtime-owned SDK/child-process environment
```

Configuration priority is:

```text
explicit CLI > explicit environment > override spec > task defaults
```

A value is normalized once at startup. Downstream code receives `RunConfig` or
a narrower dataclass such as `AgentRunRequest`, `ModelCallSpec`,
`CredentialRef`, or `RuntimeSandboxIntent`. It must not reconstruct API
provider, model, task, agent runtime, or credential choices from strings later in the call
graph.

## Ingress Boundaries

CLI entrypoints under `praxist/cli/` and `praxist/run.py` may read documented
Praxist configuration variables. They are responsible for:

- parsing operator intent;
- applying precedence;
- resolving task and run paths;
- selecting agent runtime/API provider/model refs;
- passing raw credentials only to credential resolution;
- producing frozen configuration for downstream consumers.

Task projects do not own research startup. The Praxist CLI resolves the task,
run directory, provider, credentials, plugins, budget, and workflow before
launch; task-specific runtime values enter through the validated task contract.

## Configuration Boundary

Code under `praxist/core/`, generic plugin domain logic, and infrastructure
services must receive configuration explicitly. Direct ambient reads are not a
transport mechanism between layers.

The following patterns are out of contract:

- reading `PRAXIST_*` values in core business logic;
- import-time constants populated from environment variables;
- mutating `os.environ` so another in-process layer can discover a value;
- inferring an API provider from model-name punctuation after startup resolved it;
- copying all host environment variables into a runtime or tool process;
- passing raw credentials through task files, prompts, logs, or trajectory.

## Runtime Egress

A runtime adapter may construct an SDK/client or child-process environment from
its explicit execution context. This is egress, not a second configuration
resolution pass.

The adapter must:

- include only the selected API provider credential;
- redact credentials from events, errors, and logs;
- pass non-secret task runtime values needed by shell or MCP children;
- avoid forwarding unrelated host secrets;
- keep runtime-private state under the selected run directory;
- preserve timeout, cancellation, and sandbox intent from the request.

For `agent_runtime:codex_sdk`, the official SDK owns the local app-server. The
runtime creates a private run-scoped relay only for supported Chat Completions
API providers and attaches selected Praxist tools directly over MCP. The
human-facing Codex CLI is a separate operator surface. When native OpenAI is
selected and no API key exists, its saved ChatGPT authentication may authorize
the SDK runtime; personal plugins, skills, hooks, MCP servers, instructions,
and runtime state do not become peer configuration.

## Credentials

Credential resolution converts a raw secret into a redacted `CredentialRef`
plus the minimum runtime material required for the selected API provider. Core
does not inspect API provider keys. An agent runtime may inject the resolved key
into an SDK or private child process when that external interface requires an
environment variable.

A runtime may also expose an optional managed-credential discovery hook. Core
accepts only a redacted `CredentialRef` whose `scope` and `provider` match and
whose `target_ref` is absent or matches the selected API provider ref.
Environment credentials win, and resolve-only startup never performs an agent
runtime authentication probe.

API provider failover, cooldown, and key selection remain Python control-plane
semantics. Shell wrappers and task projects must not duplicate them.

## Replay And Audit

Persisted startup configuration records the resolved non-secret values used by
the run. Replay and diagnostics read that frozen state rather than trying to
reconstruct the parent shell environment.

This provides:

- deterministic comparison between runs;
- one explanation for API provider/model/agent runtime selection;
- stable cost and usage attribution;
- a bounded secret-review surface;
- reproducible runtime and task-environment setup.

Unknown or unavailable values must remain explicit (`usage_unknown`, missing
optional capability, or a warning). Do not convert unknown state to a false
zero or infer it from unrelated environment variables.

## Task Experiment Configuration

Task evaluators own the scientific treatment they execute. When launch-time
arguments, environment overrides, protocol settings, or task-local config files
can change a result without changing variant code, the existing result summary
may publish a secret-free top-level `effective_config` object and an explicit
`effective_config_complete` boolean. Praxist hashes that object and propagates
only its digest, status, and the existing result-summary path; the summary
remains the single owner of the full configuration. The evaluator must publish
resolved treatment values after applying defaults and parsing, not a raw
environment snapshot. An omitted setting and an explicit setting equal to its
resolved default are the same configuration.

An evaluator claiming an exact replication should also publish the parent's
`replication_of_effective_config_sha256`. Praxist reports a match only when the
current configuration is complete and its digest equals the parent digest.
Missing provenance does not change ordinary evidence maturity, promotion,
closing, or legacy evaluator behavior; it only means the run cannot support an
exact-replication claim.

## Verification

Tests should inject configuration through constructors, CLI arguments, or a
bounded environment mapping at the entry boundary. They should also verify
that:

- serialized configuration is redacted;
- runtime child environments exclude unrelated secrets;
- API provider/model normalization happens once;
- task runtime values reach MCP/shell children through explicit context;
- replay does not depend on the ambient environment;
- exact-replication fixtures distinguish identical code under different
  effective task configurations;
- new core/plugin code does not introduce undocumented environment reads.
