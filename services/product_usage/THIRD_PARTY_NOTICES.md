# Third-party notices

The built-in client directly depends on:

| Project | Purpose | License | Source |
| --- | --- | --- | --- |
| Pydantic | Closed protocol validation and JSON Schema generation | MIT | <https://github.com/pydantic/pydantic> |

The separately installed `product-usage-server` extra also depends on:

| Project | Purpose | License | Source |
| --- | --- | --- | --- |
| Alembic | Database migrations | MIT | <https://github.com/sqlalchemy/alembic> |
| FastAPI | Collector HTTP application | MIT | <https://github.com/fastapi/fastapi> |
| Psycopg | PostgreSQL driver | LGPL-3.0 with exceptions | <https://github.com/psycopg/psycopg> |
| SQLAlchemy | Database access | MIT | <https://github.com/sqlalchemy/sqlalchemy> |
| Uvicorn | Collector ASGI server | BSD-3-Clause | <https://github.com/encode/uvicorn> |

Transitive dependencies are resolved by the Python package installer and
retain their own license metadata. Development tools declared in
`pyproject.toml` are not included in the runtime wheel:

| Project | Purpose | License | Source |
| --- | --- | --- | --- |
| Hatch | Python build backend | MIT | <https://github.com/pypa/hatch> |
| pytest | Tests | MIT | <https://github.com/pytest-dev/pytest> |
| Ruff | Linting | MIT | <https://github.com/astral-sh/ruff> |
| Pyrefly | Static type checking | MIT | <https://github.com/facebook/pyrefly> |

No AGPL component or copied AGPL source is included in this repository.
Release automation must generate a version-resolved license inventory and SBOM
before production distribution.
