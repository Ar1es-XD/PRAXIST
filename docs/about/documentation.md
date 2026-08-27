# Documentation Policy

The versioned Markdown under `docs/` is the only authored product
documentation source. The static website, generated reference pages, search
index, `llms.txt`, and `llms-full.txt` are derived from it. Praxist does not use
a separately edited GitHub Wiki.

## One Fact, One Owner

| Information | Sole owner | Other pages may |
|---|---|---|
| CLI arguments and defaults | `praxist.cli` parser code | Link to generated CLI reference |
| Skill name and activation description | Each `skills/*/SKILL.md` front matter | Link to generated Skills reference |
| Package installation, install commands, and filesystem effects | [Installation](../getting-started/installation.md) | Link; the root landing page may show one canonical command |
| User-facing first-run/OOBE sequence | [Quickstart](../getting-started/quickstart.md) | Name the lane and link |
| Agent-managed OOBE implementation | [Agent OOBE Runbook](../agents/oobe-install.md) | Link without duplicating agent instructions |
| Project prerequisites, research brief, and takeover stages | [Your First Task](../getting-started/first-task.md) | Show one takeover invocation and link |
| Goal-to-skill map and skill installation locations | [Agent Skills](../user-guide/skills.md) | Name a relevant skill and link |
| Task schema, precedence, and scientific ownership | [Task Projects](../guides/task-projects.md) | Explain how a mechanism consumes the task contract |
| Template/example distinction and example materialization commands | [Examples And Templates](../guides/examples-and-templates.md) | Identify an asset and link without redefining the boundary |
| Rocket Booster Recovery scientific details | `examples/rocket_booster_recovery/README.md` | Explain discovery and launch without duplicating its protocol |
| Rocket Booster Recovery (Rust) scientific details | `examples/rocket_booster_recovery_rust/README.md` | Explain discovery and launch without duplicating its protocol |
| Lifecycle semantics | [Direct CLI Operations](../guides/operators.md) | Show one quickstart command and link |
| Core/plugin/task boundary and artifact roles | [Architecture](../concepts/architecture.md) | Summarize purpose and link |
| Configuration ingress and precedence | [Configuration Discipline](../concepts/config_discipline.md) | State a local input and link |
| Agent-session and prompt-layout mental model | [Runtime Model](../concepts/runtime-model.md) | Explain a mechanism-specific effect and link |
| Runtime adapter capabilities | [Agent Runtimes](../guides/agent-runtimes.md) | Identify a selected runtime and link |
| API provider shapes | [API Providers](../guides/model-providers.md) | Identify a selected provider and link |
| Open-source model API shortlist and selection criteria | [Open-Source Model APIs](../guides/open-source-model-apis.md) | Link without duplicating the shortlist |
| Authentication and credential precedence | [Credentials](../guides/credentials.md) | State a prerequisite and link |
| Research-loop sequence | [Research Loop](../guides/research-loop-variant-generation-flow.md) | Refer to a stage without redefining the sequence |
| Maturity, close, incubator, peer-mix, and launch-freeze behavior | [Flexibility Controls](../guides/research-loop-flexibility-controls.md) | Show task configuration only where Task Projects owns the combined profile |
| Deep Innovation Gate (DIG) behavior and artifacts | [Deep Innovation Gate](../guides/deep-innovation-gate.md) | State whether DIG is active and link |
| Quality-Diversity (QD) allocation | [Quality-Diversity Allocation](../guides/qdig-cohort-allocator.md) | Refer to the selected path and link |
| Peer-session memory | [Peer Memory](../guides/peer-local-structured-memory-long-context.md) | Refer to memory as context and link |
| Experiment admission and resource ownership | [Central Experiment Scheduler](../guides/central-resource-scheduler.md) | State the selected profile and link |
| Tool catalog and frontier-tool behavior | [Tool Servers](../guides/tool-servers.md) | Name a tool and link |
| Literature source/provenance policy | [Scientific Literature Lookup](../guides/scientific-literature-lookup.md) | State whether lookup is enabled and link |
| Human-readable report triggers and semantics | [Run Reports](../guides/user-facing-reports-and-init.md) | Link to a generated report or this guide |
| Usage measurement and formulas | [Cost Estimation](../guides/costs.md) | Link to measured artifacts and this guide |
| Lossless token-saving mechanisms | [Cost Optimization](../guides/cost-optimization.md) | State that a route uses the policy and link |
| Contributor contract | `AGENTS.md` | Link without redefining it |
| Software license | Root [`LICENSE.md`](https://github.com/sapientinc/praxist/blob/main/LICENSE.md) | Link without restating license terms |
| User Agreement | [Praxist User Agreement](../legal/user-agreement.md) | Link without restating legal terms |
| Product-usage privacy policy, processing purposes, retention, and user rights | [Privacy Notice](../legal/PRIVACY.md) | Link without restating the policy |
| Exact versioned product-usage consent text | [Praxist User Data Collection Notice](../legal/product-usage-data-notice.md) | The CLI loads this same package resource; other pages identify it and link |
| Product-usage consent commands and collector operation | [Product Usage Controls](../operations/product-usage.md) | Link to the operational procedure |
| Product-usage implementation, endpoint, storage, and audit map | [Product Usage Technical Documentation](../operations/DOCUMENTATION.md) | Link without duplicating implementation details |
| Machine product-usage event schema | `praxist/product_usage/protocol.py` and checked-in JSON Schema | Technical and legal pages explain the contract; code remains authoritative for exact validation |
| Hosted documentation URL | `praxist.cli.docs.DOCUMENTATION_URL` | Mirror it in checked package/site metadata and link to it |

Tutorials sequence actions. Guides explain procedures. Concept pages explain
mental models. Reference pages enumerate machine contracts. The root README is
a product landing page, not a second manual.

A minimal command may appear in a tutorial that needs the action, but its
arguments, defaults, and edge cases remain in the generated reference. A page
may summarize an adjacent contract only far enough to explain its own behavior;
the summary must link to the owner instead of restating the contract.

## Generated Sources

`scripts/build_docs_site.py` creates:

- `docs/reference/cli.md` from the live CLI parser;
- `docs/reference/skills.md` from skill front matter;
- `site/llms.txt` as a compact machine-readable map;
- `site/llms-full.txt` as the complete navigation-ordered corpus.

Generated HTML and LLM exports are not committed. Generated Markdown reference
pages are committed so package users can read them without building the site,
but CI verifies that they exactly match their code-owned inputs.

## Navigation Ownership

Every authored Markdown page must appear exactly once in `mkdocs.yml`. This
makes its primary audience and information role explicit. The docs build
rejects missing pages, duplicate navigation ownership, and broken local links.

## Build

```bash
uv sync --extra docs
uv run python scripts/build_docs_site.py
uv run python scripts/build_docs_site.py --check-generated
```

The build runs in strict mode and does not contact API providers, read API
keys, or start research services.

## Hosted Documentation

Documentation validation runs on every pull request and every push to `main`.
Successful pushes to `main` publish the generated site to the repository's
private GitHub Pages project. Access follows repository read permission and
requires GitHub authentication; the generated HTML remains derived output and
is not committed.

The repository variable `PRAXIST_PAGES_ENABLED=true` is the deployment switch.
Maintainers can also trigger the `docs` workflow manually. Pull requests build
and validate the complete site but never publish it.

The canonical site URL is surfaced through `praxist docs`. Contributors use the
local build commands above only to preview unmerged changes.
