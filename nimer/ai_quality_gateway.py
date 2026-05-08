"""AI Quality & Trust Gateway for model responses.

This module runs three categories of checks on model output:
- Validation checks (shape/empty/repetition)
- Bias checks (stereotype phrase detection)
- Safety checks (high-risk harmful content cues)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .exceptions import TrustGatewayError


@dataclass(frozen=True)
class GatewayIssue:
    """Single issue found during a trust check."""

    category: str
    severity: str
    message: str
    matched_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class GatewayPolicy:
    name: str
    match_pattern: str
    action: str  # block | redact | warn
    is_active: bool = True
    description: str = ""


@dataclass(frozen=True)
class GatewayReport:
    """Structured report returned after gateway checks."""

    is_valid: bool
    is_safe: bool
    is_biased: bool
    has_pii: bool
    is_toxic: bool
    safety_score: float
    redacted_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    issues: tuple[GatewayIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "is_safe": self.is_safe,
            "is_biased": self.is_biased,
            "has_pii": self.has_pii,
            "is_toxic": self.is_toxic,
            "safety_score": self.safety_score,
            "redacted_text": self.redacted_text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "issues": [
                {
                    "category": issue.category,
                    "severity": issue.severity,
                    "message": issue.message,
                    "matched_terms": list(issue.matched_terms),
                }
                for issue in self.issues
            ],
        }


class AIQualityTrustGateway:
    """Runs validation, bias, and safety checks over model responses."""

    _SAFETY_TERMS: tuple[str, ...] = (
        "build a bomb",
        "make a bomb",
        "weaponize",
        "kill them",
        "how to murder",
        "bypass security",
        "exploit vulnerability",
        "write malware",
    )
    _BIAS_TERMS: tuple[str, ...] = (
        "all women are",
        "all men are",
        "all muslims are",
        "all christians are",
        "all jews are",
        "all black people are",
        "all white people are",
        "inferior race",
        "superior race",
    )
    _TOXICITY_TERMS: tuple[str, ...] = (
        "idiot",
        "stupid",
        "worthless",
        "hate you",
        "go die",
        "shut up",
        "moron",
    )
    _REPEATED_WORD_PATTERN = re.compile(r"\b(\w+)(?:\s+\1){8,}\b", re.IGNORECASE)
    _EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    _PHONE_PATTERN = re.compile(
        r"(?:(?<!\w)(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}(?!\w))"
    )

    def __init__(
        self,
        *,
        block_on_validation_failure: bool | None = None,
        block_on_safety_failure: bool | None = None,
        redact_pii: bool | None = None,
        audit_log_path: str | None = None,
        policies: list[GatewayPolicy] | None = None,
    ) -> None:
        if block_on_validation_failure is None:
            block_on_validation_failure = (
                os.getenv("NIMER_GATEWAY_BLOCK_VALIDATION", "true").lower() == "true"
            )
        if block_on_safety_failure is None:
            block_on_safety_failure = (
                os.getenv("NIMER_GATEWAY_BLOCK_SAFETY", "true").lower() == "true"
            )
        if redact_pii is None:
            redact_pii = os.getenv("NIMER_GATEWAY_REDACT_PII", "true").lower() == "true"
        if audit_log_path is None:
            audit_log_path = os.getenv("NIMER_TRUST_AUDIT_LOG_PATH", "")
        self._block_on_validation_failure = block_on_validation_failure
        self._block_on_safety_failure = block_on_safety_failure
        self._redact_pii = redact_pii
        self._audit_log_path = audit_log_path.strip()
        if policies is None:
            policies = self._load_policies_from_env()
        self._policies = tuple(policies)

    @staticmethod
    def _load_policies_from_env() -> list[GatewayPolicy]:
        raw = os.getenv("NIMER_GATEWAY_POLICIES", "").strip()
        if not raw:
            return []
        try:
            loaded = json.loads(raw)
            out: list[GatewayPolicy] = []
            if isinstance(loaded, list):
                for row in loaded:
                    if not isinstance(row, dict):
                        continue
                    action = str(row.get("action", "warn")).lower()
                    if action not in {"block", "redact", "warn"}:
                        action = "warn"
                    out.append(
                        GatewayPolicy(
                            name=str(row.get("name", "custom-policy")),
                            match_pattern=str(row.get("match_pattern", "")),
                            action=action,
                            is_active=bool(row.get("is_active", True)),
                            description=str(row.get("description", "")),
                        )
                    )
            return out
        except Exception:
            return []

    def filter_input(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[GatewayIssue], bool]:
        """Sanitize inbound prompt content and decide whether to block."""
        combined = self._flatten_prompt(messages, system=system)
        _, issues, _ = self._scan_text(combined, include_validation=False)
        blocked = any(
            issue.category in {"safety", "toxicity"}
            or (issue.category == "policy" and "(block)" in issue.message)
            for issue in issues
        )
        if not self._redact_pii:
            return messages, issues, blocked
        sanitized: list[dict[str, Any]] = []
        for msg in messages:
            updated = dict(msg)
            content = updated.get("content")
            if isinstance(content, str):
                updated["content"] = self._redact_text(content)
            sanitized.append(updated)
        return sanitized, issues, blocked

    def check(
        self,
        response: Any,
        *,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> GatewayReport:
        """Run all checks and return a structured report."""
        text = self._extract_text(response)
        redacted_text, issues, safety_score = self._scan_text(text, include_validation=True)

        is_valid = not any(issue.category == "validation" for issue in issues)
        is_safe = not any(issue.category == "safety" for issue in issues)
        is_biased = any(issue.category == "bias" for issue in issues)
        has_pii = any(issue.category == "pii" for issue in issues)
        is_toxic = any(issue.category == "toxicity" for issue in issues)
        report = GatewayReport(
            is_valid=is_valid,
            is_safe=is_safe,
            is_biased=is_biased,
            has_pii=has_pii,
            is_toxic=is_toxic,
            safety_score=safety_score,
            redacted_text=redacted_text,
            input_tokens=max(input_tokens, 0),
            output_tokens=max(output_tokens, 0),
            total_tokens=max(input_tokens, 0) + max(output_tokens, 0),
            latency_ms=max(latency_ms, 0.0),
            issues=tuple(issues),
        )
        self._write_audit_event(report)
        return report

    def enforce(
        self,
        response: Any,
        *,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> GatewayReport:
        """Run checks and raise if a blocking policy is violated."""
        report = self.check(
            response,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        if self._block_on_validation_failure and not report.is_valid:
            raise TrustGatewayError("AI Quality & Trust Gateway blocked invalid response.")
        if self._block_on_safety_failure and not report.is_safe:
            raise TrustGatewayError("AI Quality & Trust Gateway blocked unsafe response.")
        if report.is_toxic:
            raise TrustGatewayError("AI Quality & Trust Gateway blocked toxic response.")
        if any(
            issue.category == "policy" and "(block)" in issue.message
            for issue in report.issues
        ):
            raise TrustGatewayError("AI Quality & Trust Gateway blocked response by custom policy.")
        return report

    def _extract_text(self, response: Any) -> str:
        """Best-effort extraction from Anthropic-like response objects."""
        content = getattr(response, "content", None)
        if content is None:
            return str(response or "")

        chunks: list[str] = []
        if isinstance(content, list):
            for block in content:
                block_type = getattr(block, "type", None)
                block_text = getattr(block, "text", None)
                if isinstance(block, dict):
                    block_type = block.get("type")
                    block_text = block.get("text")
                if block_type == "text" and isinstance(block_text, str):
                    chunks.append(block_text)
        elif isinstance(content, str):
            chunks.append(content)

        return "\n".join(chunks)

    @staticmethod
    def _find_terms(text: str, terms: tuple[str, ...]) -> list[str]:
        lowered = text.lower()
        return [term for term in terms if term in lowered]

    def _find_pii(self, text: str) -> list[str]:
        matches: list[str] = []
        if self._EMAIL_PATTERN.search(text):
            matches.append("email")
        if self._PHONE_PATTERN.search(text):
            matches.append("phone")
        return matches

    def _flatten_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | list[dict[str, Any]] | None = None,
    ) -> str:
        parts: list[str] = []
        if isinstance(system, str):
            parts.append(system)
        elif isinstance(system, list):
            for chunk in system:
                if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
                    parts.append(chunk["text"])
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
        return "\n".join(parts)

    def _scan_text(
        self,
        text: str,
        *,
        include_validation: bool,
    ) -> tuple[str, list[GatewayIssue], float]:
        issues: list[GatewayIssue] = []
        pii_hits = self._find_pii(text)
        redacted_text = self._redact_text(text) if self._redact_pii else text

        if include_validation and not text.strip():
            issues.append(
                GatewayIssue(
                    category="validation",
                    severity="high",
                    message="AI response contains no text content.",
                )
            )

        if include_validation and self._REPEATED_WORD_PATTERN.search(text):
            issues.append(
                GatewayIssue(
                    category="validation",
                    severity="medium",
                    message="AI response appears to contain degenerate repetition.",
                )
            )

        safety_hits = self._find_terms(text, self._SAFETY_TERMS)
        if safety_hits:
            issues.append(
                GatewayIssue(
                    category="safety",
                    severity="high",
                    message="Content contains potentially harmful guidance.",
                    matched_terms=tuple(safety_hits),
                )
            )

        toxicity_hits = self._find_terms(text, self._TOXICITY_TERMS)
        if toxicity_hits:
            issues.append(
                GatewayIssue(
                    category="toxicity",
                    severity="high",
                    message="Content contains toxic language.",
                    matched_terms=tuple(toxicity_hits),
                )
            )

        bias_hits = self._find_terms(text, self._BIAS_TERMS)
        if bias_hits:
            issues.append(
                GatewayIssue(
                    category="bias",
                    severity="medium",
                    message="Content contains potentially biased language.",
                    matched_terms=tuple(bias_hits),
                )
            )
        if pii_hits:
            issues.append(
                GatewayIssue(
                    category="pii",
                    severity="medium",
                    message="Content contains possible PII.",
                    matched_terms=tuple(pii_hits),
                )
            )
        for policy in self._policies:
            if not policy.is_active:
                continue
            if not policy.match_pattern:
                continue
            try:
                matched = re.search(policy.match_pattern, text, flags=re.IGNORECASE) is not None
            except re.error:
                matched = False
            if not matched:
                continue
            issues.append(
                GatewayIssue(
                    category="policy",
                    severity="high" if policy.action == "block" else "medium",
                    message=f"Matched policy '{policy.name}' ({policy.action}).",
                    matched_terms=(policy.match_pattern,),
                )
            )
            if policy.action == "redact":
                redacted_text = re.sub(
                    policy.match_pattern,
                    "[REDACTED_POLICY]",
                    redacted_text,
                    flags=re.IGNORECASE,
                )
        return redacted_text, issues, self._score_issues(issues)

    @staticmethod
    def _score_issues(issues: list[GatewayIssue]) -> float:
        penalty = {
            "validation": 35.0,
            "safety": 45.0,
            "toxicity": 40.0,
            "bias": 20.0,
            "pii": 20.0,
            "policy": 15.0,
        }
        score = 100.0
        for issue in issues:
            score -= penalty.get(issue.category, 10.0)
        return round(max(0.0, min(100.0, score)), 3)

    def _redact_text(self, text: str) -> str:
        redacted = self._EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        return self._PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)

    def _write_audit_event(self, report: GatewayReport) -> None:
        if not self._audit_log_path:
            return
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "gateway_report": report.to_dict(),
        }
        try:
            path = Path(self._audit_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            # Gateway observability must never break request handling.
            return
