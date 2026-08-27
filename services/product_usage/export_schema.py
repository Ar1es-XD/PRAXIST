"""Export the checked-in draft JSON Schemas from the Pydantic source of truth."""

from __future__ import annotations

import json
from pathlib import Path

from praxist.product_usage.protocol import batch_schema, event_schema

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIRECTORY = ROOT / "praxist" / "product_usage" / "schemas" / "v2"


def main() -> None:
    SCHEMA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    outputs = {
        "usage-event.schema.json": event_schema(),
        "usage-batch.schema.json": batch_schema(),
    }
    for filename, schema in outputs.items():
        target = SCHEMA_DIRECTORY / filename
        target.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
