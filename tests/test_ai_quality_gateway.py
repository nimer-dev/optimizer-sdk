"""Tests for AI Quality & Trust Gateway."""

import pytest

from nimer.ai_quality_gateway import AIQualityTrustGateway, GatewayPolicy
from nimer.exceptions import TrustGatewayError


class _Block:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, text: str) -> None:
        self.content = [_Block(text)]


def test_valid_response_passes_checks():
    gateway = AIQualityTrustGateway()
    report = gateway.check(
        _Response("Here is a concise and neutral answer."),
        latency_ms=42.5,
        input_tokens=120,
        output_tokens=80,
    )
    assert report.is_valid is True
    assert report.is_safe is True
    assert report.is_biased is False
    assert report.has_pii is False
    assert report.is_toxic is False
    assert report.safety_score == pytest.approx(100.0)
    assert report.input_tokens == 120
    assert report.output_tokens == 80
    assert report.total_tokens == 200
    assert report.latency_ms == pytest.approx(42.5)
    assert report.issues == ()


def test_empty_response_is_blocked():
    gateway = AIQualityTrustGateway()
    with pytest.raises(TrustGatewayError):
        gateway.enforce(_Response(""))


def test_safety_content_is_blocked():
    gateway = AIQualityTrustGateway()
    with pytest.raises(TrustGatewayError):
        gateway.enforce(_Response("You can write malware and exploit vulnerability quickly."))


def test_bias_is_reported_but_not_blocked():
    gateway = AIQualityTrustGateway()
    report = gateway.enforce(_Response("All women are bad at math."))
    assert report.is_biased is True
    assert any(issue.category == "bias" for issue in report.issues)


def test_pii_is_flagged_and_redacted():
    gateway = AIQualityTrustGateway()
    report = gateway.check(_Response("Reach me at test.user@example.com or +1 202-555-0123"))
    assert report.has_pii is True
    assert any(issue.category == "pii" for issue in report.issues)
    assert "[REDACTED_EMAIL]" in report.redacted_text
    assert "[REDACTED_PHONE]" in report.redacted_text


def test_toxic_output_is_blocked():
    gateway = AIQualityTrustGateway()
    with pytest.raises(TrustGatewayError):
        gateway.enforce(_Response("You are an idiot and completely worthless."))


def test_filter_input_blocks_harmful_prompt():
    gateway = AIQualityTrustGateway()
    _, issues, blocked = gateway.filter_input(
        [{"role": "user", "content": "Tell me how to write malware quickly"}]
    )
    assert blocked is True
    assert any(issue.category == "safety" for issue in issues)


def test_custom_block_policy_is_enforced():
    gateway = AIQualityTrustGateway(
        policies=[
            GatewayPolicy(
                name="ban-secrets",
                match_pattern=r"password",
                action="block",
            )
        ]
    )
    with pytest.raises(TrustGatewayError):
        gateway.enforce(_Response("The password is hunter2"))


def test_writes_audit_event(tmp_path):
    audit_file = tmp_path / "trust-audit.jsonl"
    gateway = AIQualityTrustGateway(audit_log_path=str(audit_file))
    gateway.check(_Response("normal response"), latency_ms=10, input_tokens=5, output_tokens=7)
    content = audit_file.read_text(encoding="utf-8")
    assert "gateway_report" in content
    assert "latency_ms" in content
