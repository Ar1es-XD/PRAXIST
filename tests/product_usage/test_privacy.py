from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from praxist.product_usage.protocol import RunStartedEvent
from tests.helpers.product_usage import event_dict


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "ip_address",
        "ip_hash",
        "path",
        "filename",
        "working_directory",
        "hostname",
        "prompt",
        "task_content",
        "command",
        "environment",
        "log",
        "stack_trace",
        "raw_error",
        "provider",
        "model",
        "domain",
        "credential",
        "operating_system",
        "hardware",
        "python_version",
        "timezone",
        "user_agent",
        "headers",
        "cookie",
        "metadata",
        "received_at",
        "peer_id",
        "installation_method",
        "research_direction",
        "result_showcase",
        "default_attribution",
    ],
)
def test_forbidden_and_unknown_fields_are_rejected(forbidden_field: str) -> None:
    payload = event_dict()
    payload[forbidden_field] = "/Users/alice/secret"

    with pytest.raises(ValidationError):
        RunStartedEvent.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    "error_code",
    [
        "PRX-/Users/alice/project",
        "PRX-raw error message",
        "THIRD-PARTY-429",
        "PRX-provider.example.com",
        "PRX_1001",
        "PRX-UNREGISTERED-CODE",
    ],
)
def test_error_code_is_bounded_code_not_free_text(error_code: str) -> None:
    summary = {
        "scope": "peer",
        "stage": "launch",
        "error_type": "unknown",
        "error_code": error_code,
        "reason_code": "unknown",
        "count": 1,
        "count_capped": False,
    }
    with pytest.raises(ValidationError):
        RunStartedEvent.model_validate_json(json.dumps(event_dict(error_summaries=[summary])))
