"""Ensure cocktail and consumption answers include a responsible-drinking warning."""

from __future__ import annotations

import re

from guardrails.models import GuardrailDecision, normalize_query
from utils.logging import get_logger

logger = get_logger(__name__)

RESPONSIBLE_DRINKING_WARNING = (
    "Please drink responsibly. This information is intended for adults of legal drinking age."
)

_CONSUMPTION_RE = re.compile(
    r"\b("
    r"cocktail|mocktail|recipe|mix|mixer|serve|serving|garnish|highball|"
    r"martini|negroni|old fashioned|shot|shots|neat|on the rocks|"
    r"how (?:to|do i) (?:drink|serve|mix|make)|"
    r"alcohol|drinking|consume|consumption|abv|proof|pour"
    r")\b",
    re.IGNORECASE,
)


class ResponsibleDrinkingGuardrail:
    """Flags alcohol-consumption queries and appends a concise professional warning."""

    warning = RESPONSIBLE_DRINKING_WARNING

    def check(self, query: str) -> GuardrailDecision:
        needed = bool(_CONSUMPTION_RE.search(query or ""))
        return GuardrailDecision.allow(
            policy="responsible_drinking",
            needs_responsible_drinking=needed,
            metadata={"needs_warning": needed},
        )

    def apply(self, answer: str, *, required: bool) -> str:
        text = (answer or "").strip()
        if not required or not text:
            return text
        if self._already_present(text):
            return text
        logger.info("responsible_drinking_warning_appended")
        return f"{text}\n\n{self.warning}"

    def _already_present(self, text: str) -> bool:
        normalized = normalize_query(text)
        return "drink responsibly" in normalized or "legal drinking age" in normalized
