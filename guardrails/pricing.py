"""Refuse prices, estimates, and purchasing assistance."""

from __future__ import annotations

import re

from config.sources import BRAND_SOURCES, PERNOD_RICARD_SOURCES, official_url_for_source_name
from guardrails.models import GuardrailDecision, normalize_query
from utils.logging import get_logger

logger = get_logger(__name__)

_PRICE_PATTERNS = (
    r"\bprice(s|d)?\b",
    r"\bpricing\b",
    r"\bcost(s)?\b",
    r"\bhow much (does|is|are|do|for|to buy)\b",
    r"\bwhat(?:'s| is) it worth\b",
    r"\bbuy\b",
    r"\bbuying\b",
    r"\bpurchase\b",
    r"\bpurchasing\b",
    r"\border now\b",
    r"\badd to cart\b",
    r"\bcheckout\b",
    r"\bretailer\b",
    r"\bwhere to buy\b",
    r"\bwhere can i (?:buy|get|order)\b",
    r"\bcheapest\b",
    r"\bdiscount\b",
    r"\bmsrp\b",
    r"\brrp\b",
    r"[$€£]\s?\d",
    r"\b\d+\s?(usd|eur|gbp|dollars|euros|pounds)\b",
)

_PRICE_RE = re.compile("|".join(_PRICE_PATTERNS), re.IGNORECASE)

_BRAND_ALIASES = (
    ("absolut", "Absolut"),
    ("chivas", "Chivas Regal"),
    ("jameson", "Jameson"),
    ("glenlivet", "The Glenlivet"),
    ("beefeater", "Beefeater"),
    ("ballantine", "Ballantine's"),
    ("royal salute", "Royal Salute"),
    ("malibu", "Malibu"),
    ("kahlua", "Kahlúa"),
    ("kahlúa", "Kahlúa"),
    ("mumm", "G.H. Mumm"),
    ("perrier", "Perrier-Jouët"),
    ("pernod", "Pernod Ricard"),
    ("ricard", "Pernod Ricard"),
)


class PricingGuardrail:
    """Never quote prices or enable purchasing; redirect to the official brand site."""

    def check(self, query: str) -> GuardrailDecision:
        if not _PRICE_RE.search(query or ""):
            return GuardrailDecision.allow(policy="pricing")
        url = self.official_url_for_query(query)
        logger.info("pricing_refused", redirect_url=url)
        message = (
            "I cannot provide prices, pricing estimates, or purchasing guidance. "
            f"For official product information, please visit {url}."
        )
        return GuardrailDecision.deny(
            "pricing",
            message,
            redirect_url=url,
            metadata={"reason": "price_or_purchase"},
        )

    def official_url_for_query(self, query: str) -> str:
        normalized = normalize_query(query)
        for alias, name in _BRAND_ALIASES:
            if alias in normalized:
                return official_url_for_source_name(name)
        for source in BRAND_SOURCES + PERNOD_RICARD_SOURCES:
            if source.name.lower() in normalized:
                return source.official_site or source.base_url
        return official_url_for_source_name("Pernod Ricard")
