"""Refuse political, harmful, and unrelated general-purpose queries."""

from __future__ import annotations

import re

from config.sources import BRAND_SOURCES, PERNOD_RICARD_SOURCES
from guardrails.models import GuardrailDecision, normalize_query
from utils.logging import get_logger

logger = get_logger(__name__)

_REFUSAL = (
    "I can only discuss Pernod Ricard, its brands, and related product heritage. "
    "Please ask about a group brand or cocktail serve instead."
)

_HARMFUL_RE = re.compile(
    r"\b("
    r"bomb|explosive|weapon|kill|suicide|self-harm|harm yourself|"
    r"how to get (?:a )?minor drunk|spike (?:a |the )?drink|date rape|"
    r"hack|phishing|malware|steal"
    r")\b",
    re.IGNORECASE,
)

_POLITICS_RE = re.compile(
    r"\b("
    r"election|president|prime minister|democrat|republican|labour party|"
    r"parliament|congress|politic|geopolitics|war in|vote for"
    r")\b",
    re.IGNORECASE,
)

_INJECTION_RE = re.compile(
    r"\b("
    r"ignore (?:all |previous |the )?instructions|reveal (?:your )?(?:system )?prompt|"
    r"jailbreak|developer mode|override (?:the )?polic"
    r")\b",
    re.IGNORECASE,
)

_GREETING_RE = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you)[\s!.]*$", re.I)

_DOMAIN_TERMS = {
    "pernod",
    "ricard",
    "absolut",
    "chivas",
    "jameson",
    "glenlivet",
    "beefeater",
    "ballantine",
    "royal salute",
    "malibu",
    "kahlua",
    "kahlúa",
    "mumm",
    "perrier",
    "jouet",
    "jouët",
    "whisky",
    "whiskey",
    "vodka",
    "gin",
    "rum",
    "liqueur",
    "champagne",
    "cocktail",
    "brand",
    "distill",
    "heritage",
    "tasting",
    "serve",
    "abv",
}


def _domain_terms() -> set[str]:
    terms = set(_DOMAIN_TERMS)
    for source in PERNOD_RICARD_SOURCES + BRAND_SOURCES:
        terms.add(source.name.lower())
        for token in source.name.lower().replace("-", " ").split():
            if len(token) > 3:
                terms.add(token)
    return terms


class OffTopicGuardrail:
    """Single-sentence refusal for politics, harm, injection, and unrelated requests."""

    def __init__(self) -> None:
        self._domain = _domain_terms()

    def check(self, query: str) -> GuardrailDecision:
        text = query or ""
        if _HARMFUL_RE.search(text):
            logger.info("off_topic_harmful")
            return GuardrailDecision.deny("off_topic", _REFUSAL, metadata={"reason": "harmful"})
        if _INJECTION_RE.search(text):
            logger.info("off_topic_injection")
            return GuardrailDecision.deny("off_topic", _REFUSAL, metadata={"reason": "injection"})
        if _GREETING_RE.match(text.strip()):
            return GuardrailDecision.allow(policy="off_topic", metadata={"greeting": True})
        if _POLITICS_RE.search(text):
            logger.info("off_topic_politics")
            return GuardrailDecision.deny("off_topic", _REFUSAL, metadata={"reason": "politics"})
        if self._in_domain(text):
            return GuardrailDecision.allow(policy="off_topic")
        logger.info("off_topic_unrelated")
        return GuardrailDecision.deny("off_topic", _REFUSAL, metadata={"reason": "unrelated"})

    def _in_domain(self, query: str) -> bool:
        normalized = normalize_query(query)
        return any(term in normalized for term in self._domain)
