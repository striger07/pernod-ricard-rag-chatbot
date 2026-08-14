"""Refuse medical, health, addiction, and legal advice."""

from __future__ import annotations

import re
from typing import Optional

from config.settings import Settings, get_settings
from guardrails.models import GuardrailDecision
from utils.logging import get_logger

logger = get_logger(__name__)

_MEDICAL_LEGAL_RE = re.compile(
    r"\b("
    r"doctor|medical|medicine|health risk|health|pregnant|pregnancy|breastfeed|"
    r"cancer|liver|cirrhosis|addiction|alcoholic|alcoholism|rehab|withdrawal|"
    r"overdose|poison|intoxicat|hangover cure|is it safe to drink|"
    r"drink[- ]driving|drunk driving|dui|dwi|blood alcohol|\bbac\b|"
    r"can i drive|should i drive|driving after|drink and drive|"
    r"legal advice|lawsuit|sue|attorney|lawyer|illegal|law in|"
    r"underage drinking|fetal alcohol|aa meeting|alcoholics anonymous"
    r")\b",
    re.IGNORECASE,
)


class MedicalLegalGuardrail:
    """Do not give professional medical or legal advice; redirect to Drinkaware and NIAAA."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def check(self, query: str) -> GuardrailDecision:
        match = _MEDICAL_LEGAL_RE.search(query or "")
        if not match:
            return GuardrailDecision.allow(policy="medical_legal")
        drinkaware = self.settings.drinkaware_url
        niaaa = self.settings.niaaa_url
        logger.info("medical_legal_refused", term=match.group(0))
        message = (
            "I cannot provide medical, health, addiction, or legal advice. "
            "For independent guidance, please consult a qualified professional and visit "
            f"Drinkaware ({drinkaware}) and NIAAA ({niaaa})."
        )
        return GuardrailDecision.deny(
            "medical_legal",
            message,
            redirect_url=drinkaware,
            metadata={"niaaa_url": niaaa, "drinkaware_url": drinkaware, "term": match.group(0)},
        )
