"""Refuse competitor comparisons and rankings."""

from __future__ import annotations

import re

from guardrails.models import GuardrailDecision
from utils.logging import get_logger

logger = get_logger(__name__)

COMPETITOR_COMPANIES = (
    "diageo",
    "bacardi",
    "brown-forman",
    "brown forman",
    "brownforman",
    "lvmh",
    "moet hennessy",
    "moët hennessy",
    "rémy cointreau",
    "remy cointreau",
    "campari group",
    "constellation brands",
    "beam suntory",
    "suntory",
    "heineken",
    "ab inbev",
    "anheuser-busch",
)

COMPETITOR_BRANDS = (
    "johnnie walker",
    "johnny walker",
    "smirnoff",
    "captain morgan",
    "tanqueray",
    "gordon's gin",
    "gordons gin",
    "guinness",
    "baileys",
    "crown royal",
    "don julio",
    "casamigos",
    "grey goose",
    "patron",
    "patrón",
    "bombay sapphire",
    "dewars",
    "dewar's",
    "jack daniel",
    "jack daniels",
    "woodford reserve",
    "old crow",
    "hennessy",
    "moet",
    "moët",
    "veuve clicquot",
    "dom perignon",
    "jose cuervo",
    "1800 tequila",
    "maker's mark",
    "makers mark",
    "jim beam",
    "courvoisier",
    "martell",
)

_COMPARISON_RE = re.compile(
    r"\b(vs\.?|versus|compared to|compare(d)? with|better than|worse than|"
    r"which is better|rankings?|market share|competitor)\b",
    re.IGNORECASE,
)

_COMPANY_RE = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in COMPETITOR_COMPANIES) + r")\b",
    re.IGNORECASE,
)
_BRAND_RE = re.compile(
    r"\b(" + "|".join(re.escape(name) for name in COMPETITOR_BRANDS) + r")\b",
    re.IGNORECASE,
)

_REFUSAL = (
    "I do not compare Pernod Ricard brands with competitor companies or brands, "
    "and I cannot provide competitive rankings or recommendations. "
    "I can discuss Pernod Ricard’s own brands and heritage instead."
)


class CompetitorGuardrail:
    """Blocks Diageo/Bacardi/Brown-Forman and similar competitive discussion."""

    def check(self, query: str) -> GuardrailDecision:
        text = query or ""
        company = _COMPANY_RE.search(text)
        brand = _BRAND_RE.search(text)
        comparison = bool(_COMPARISON_RE.search(text))
        if not company and not brand:
            return GuardrailDecision.allow(policy="competitor")
        logger.info(
            "competitor_refused",
            company=bool(company),
            brand=bool(brand),
            comparison=comparison,
        )
        return GuardrailDecision.deny(
            "competitor",
            _REFUSAL,
            metadata={
                "competitor": (company or brand).group(0) if (company or brand) else "",
                "comparison": comparison,
            },
        )
