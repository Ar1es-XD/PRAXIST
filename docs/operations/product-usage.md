# Product Usage Controls

Praxist includes optional, pseudonymized product-usage reporting. Collection is
off until the current operating-system user explicitly consents, and declining
or withdrawing does not affect installation or research. Accepting the
[Praxist User Agreement](../legal/user-agreement.md) or the
[Fair Source License](https://github.com/sapientinc/praxist/blob/main/LICENSE.md)
does not enable collection.

Read the [Privacy Notice](../legal/PRIVACY.md) for the complete privacy policy,
purposes, retention rules, and user rights. The
[Praxist User Data Collection Notice](../legal/product-usage-data-notice.md) is
the exact versioned text presented when consent is requested. The
[Product Usage Technical Documentation](DOCUMENTATION.md) describes the
implementation, endpoint, schema, storage, and failure isolation.

## Review and Choose

Review the current in-product notice and record a choice:

```bash
praxist product-usage notice
praxist product-usage consent
```

Inspect or withdraw the choice at any time:

```bash
praxist product-usage status --json
praxist product-usage withdraw
```

A plain pip installation and every non-interactive path leave consent unset.
Interactive setup presents the notice in a scrollable local terminal view.
Withdrawal stops future capture and removes unsent local events. See the
Privacy Notice for the treatment of events already delivered.

## Collector Development

Install server dependencies only on a collector development or deployment host:

```bash
uv sync --group dev --extra product-usage-server
export DATABASE_URL='postgresql+psycopg://user:password@host/database'
export COLLECTOR_INGESTION_ENABLED=true
export COLLECTOR_MAX_TABLE_BYTES=$((2 * 1024 * 1024 * 1024))
uv run alembic -c services/product_usage/alembic.ini upgrade head
uv run praxist-collector
```

Run retention in a separate process or container:

```bash
uv run praxist-retention
```

Collector ingestion is disabled until explicitly enabled. Deployment assets
and their operational controls are under `services/product_usage/`;
implementation details and audit entry points are owned by the technical
documentation.
