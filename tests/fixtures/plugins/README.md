# Test Plugin Fixtures

This tree contains offline plugin fixtures used by the unit test suite.
They are intentionally outside `praxist/plugins` so the bundled production plugin
catalog only contains generic system components.

Tests expose this root through `PRAXIST_BUNDLED_PLUGIN_ROOTS` in `tests/__init__.py`.
Production runs should not depend on these fixtures.
