"""Canonical consent notice bundled with every product-usage client."""

from __future__ import annotations

from praxist.user_agreement import product_usage_notice_text


def consent_notice_v2() -> str:
    """Load the sole authored notice for the V2 product-usage protocol."""

    return product_usage_notice_text()
