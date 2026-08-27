# Baseline Regression Suite

Run from the example root:

```bash
./scripts/run_tests.sh
```

The default suite verifies the single first-contact success predicate, accelerator
assignment propagation, and effective-configuration identity. Run a real evaluator
integrity canary explicitly with:

```bash
RUN_GPU_INTEGRITY_TEST=1 ./scripts/run_tests.sh
```

Praxist core owns lane, closing, and scheduler integration tests, so this example
does not duplicate tests that depend on a particular Praxist source checkout.
