# Adversarial Tests

These tests encode long-lived architecture contracts at high-risk product
boundaries.

Rules for this directory:

- Prefer synthetic task/plugin/run fixtures over source-text checks.
- Exercise public loaders, verifiers, tool handlers, or stable boundary APIs.
- Avoid exact model names, SAM-specific prose, current docs wording, and
  implementation sequence labels unless they are the behavior under test.
- Keep tests deterministic, local, and network-free.
- When a test fails against the current system, treat it as a production
  contract gap, not a reason to weaken the test.

Focused run:

```bash
python -m unittest tests.adversarial
```
