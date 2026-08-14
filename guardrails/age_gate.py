"""Server-side legal drinking age enforcement."""

from __future__ import annotations

import re
from typing import Optional

from config.settings import Settings, get_settings
from guardrails.models import GuardrailDecision, normalize_query
from utils.logging import get_logger

logger = get_logger(__name__)

_AGE_CLAIM_RE = re.compile(
    r"\b(?:i(?:['’]m| am)|i am only|my age is)\s+(\d{1,2})\b",
    re.IGNORECASE,
)
_UNDERAGE_PHRASES = (
    "i am underage",
    "i'm underage",
    "i am a minor",
    "i'm a minor",
    "i am not 18",
    "i'm not 18",
    "not of legal age",
    "under 18",
    "under 21",
)


class AgeGate:
    """Blocks alcohol-related product access unless legal age is verified server-side."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    @property
    def minimum_age(self) -> int:
        return int(self.settings.minimum_legal_drinking_age)

    def check(
        self,
        *,
        query: str = "",
        age_verified: bool = False,
        declared_age: Optional[int] = None,
        session_verified: bool = False,
        header_verified: bool = False,
    ) -> GuardrailDecision:
        if not self.settings.age_gate_enabled:
            return GuardrailDecision.allow(policy="age_gate")

        claimed = self._claimed_age(query)
        if claimed is not None and claimed < self.minimum_age:
            logger.info("age_gate_underage_claim", claimed_age=claimed)
            return self._underage_denial(claimed)
        if self._admits_underage(query):
            logger.info("age_gate_underage_phrase")
            return self._underage_denial(None)
        if declared_age is not None and declared_age < self.minimum_age:
            logger.info("age_gate_declared_underage", declared_age=declared_age)
            return self._underage_denial(declared_age)

        verified = bool(session_verified)
        if self.settings.age_gate_strict_server_enforcement:
            # Frontend flags are corroborating signals only; session or this request
            # must still present an explicit adult confirmation.
            request_verified = bool(age_verified)
            if not verified and not request_verified:
                logger.info("age_gate_unverified")
                return GuardrailDecision.deny(
                    "age_gate",
                    (
                        "Please confirm that you are of legal drinking age "
                        f"({self.minimum_age}+) before continuing."
                    ),
                    metadata={"requires_verification": True},
                )
            if request_verified:
                verified = True
        else:
            verified = bool(session_verified or age_verified or header_verified)

        if not verified:
            return GuardrailDecision.deny(
                "age_gate",
                (
                    "Please confirm that you are of legal drinking age "
                    f"({self.minimum_age}+) before continuing."
                ),
                metadata={"requires_verification": True},
            )

        logger.info("age_gate_passed", header_verified=header_verified)
        return GuardrailDecision.allow(
            policy="age_gate",
            metadata={"age_verified": True, "minimum_age": self.minimum_age},
        )

    def _underage_denial(self, age: Optional[int]) -> GuardrailDecision:
        return GuardrailDecision.deny(
            "age_gate",
            (
                "This assistant is only available to adults of legal drinking age. "
                "Alcohol-related content cannot be provided."
            ),
            metadata={"underage": True, "declared_age": age},
        )

    def _claimed_age(self, query: str) -> Optional[int]:
        match = _AGE_CLAIM_RE.search(query or "")
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _admits_underage(self, query: str) -> bool:
        normalized = normalize_query(query)
        return any(phrase in normalized for phrase in _UNDERAGE_PHRASES)
