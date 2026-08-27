# Praxist Test Architecture

Tests are organized around long-term contracts. New tests should normally go
in one of these directories:

- `core/`: generic core invariants such as registry, replay, ledgers,
  credentials, artifacts, and resource guards.
- `conformance/`: plugin-kind contracts. These tests should scan or exercise
  plugin manifests and adapters through public loaders.
- `integration/`: offline cross-component tests that execute an explicit
  external task fixture through startup, plugin resolution, workflow execution,
  run artifacts, and replay.
- `workflows/`: cross-plugin workflow smoke tests, such as the fake workflow fixture
  research-loop runs and SAM resolve-only startup.
- `hardening/`: adversarial regression tests for redaction, spoofing,
  replay drift, artifact integrity, budget semantics, and prompt layout.
- `adversarial/`: high-leverage synthetic boundary tests for cross-module
  contracts; these should avoid brittle prose/source checks.
- `legacy_migration/`: characterization and parity tests for compatibility
  behavior.
- `plugins/`: plugin-specific tests that are too domain-specific for the
  generic conformance harness.
- `unit/`: small package, CLI, CI, and tooling contract tests.
- `helpers/`: reusable test fixtures and utilities.

Default local verification:

```bash
python -m unittest discover -s tests
python scripts/run_test_coverage.py unit --fail-under 90 --fail-under-statements 95
python scripts/run_test_coverage.py integration
git diff --check
```

Useful focused runs:

```bash
python -m unittest tests.core
python -m unittest tests.conformance
python -m unittest tests.integration
python -m unittest tests.workflows
python -m unittest tests.hardening
python -m unittest tests.adversarial
python -m unittest tests.legacy_migration
python -m unittest tests.plugins
python -m unittest tests.unit
```

Offline fixture smoke profile:

```bash
python -m compileall -q tests
python -m unittest tests.core tests.conformance tests.integration tests.workflows
python scripts/run_test_coverage.py unit --fail-under 90 --fail-under-statements 95
python scripts/run_test_coverage.py integration
```

Integration tests must stay offline: no real API keys, network, external task
repos, GPUs, S3, RunPod, or a live task-specific benchmark. They should copy a tracked template
task into a temporary external path and run through the same public entrypoints
operators use.

The unit coverage profile is ratcheted at 90% branch-aware total coverage and
95% statement coverage. It runs the established unittest package layers plus
the pytest-based `tests/product_usage` suite under the same coverage session.
Integration coverage is reported for visibility, but it is not thresholded
because the profile is meant to prove cross-component behavior rather than
maximize line coverage.

Coverage reports are generated with coverage.py and written to ignored local
directories. The `unit` coverage profile means the offline non-integration
test layers (`core`, `conformance`, `workflows`, `hardening`, `adversarial`,
`legacy_migration`, `unit`, and the product-usage pytest suite); the
`integration` profile is `tests/integration`.

- `cover/unit/coverage.json`
- `cover/unit/coverage.xml`
- `cover/integration/coverage.json`
- `cover/integration/coverage.xml`

Plugin authors should add small plugin-local tests under `tests/plugins/` and
make sure the plugin also passes the relevant `tests/conformance/` suite.
