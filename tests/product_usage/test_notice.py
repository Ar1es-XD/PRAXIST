from __future__ import annotations

from praxist.product_usage.notice import consent_notice_v2


def test_v2_notice_discloses_environment_linkage_retention_and_withdrawal() -> None:
    notice = " ".join(consent_notice_v2().split())

    assert "pseudonymous" in notice.lower()
    assert "environment id" in notice.lower()
    assert "across Research Runs" in notice
    assert "180 days" in notice
    assert "development placeholder" not in notice.lower()
    assert "https://telemetry.theaiscientist.com/v1/events" in notice
    assert "without following redirects" in notice
    assert "notice version: 3" in notice.lower().replace("*", "")
    assert "praxist product-usage withdraw" in notice
    assert "already delivered" in notice
    assert "Apache" not in notice
