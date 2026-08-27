from __future__ import annotations

import json
from pathlib import Path

from praxist.product_usage.protocol import batch_schema, event_schema


def test_checked_in_schemas_match_models() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "usage-event.schema.json": event_schema(),
        "usage-batch.schema.json": batch_schema(),
    }
    for filename, schema in expected.items():
        checked_in = json.loads(
            (root / "praxist" / "product_usage" / "schemas" / "v2" / filename).read_text(
                encoding="utf-8"
            )
        )
        assert checked_in == schema


def test_json_schema_exposes_uuid4_and_utc_second_boundaries() -> None:
    encoded = json.dumps(event_schema(), sort_keys=True)
    assert '"format": "uuid4"' in encoded
    assert r"\\d{2}:\\d{2}:\\d{2}Z$" in encoded
