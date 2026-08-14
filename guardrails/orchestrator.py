"""Run all query/response policy checks in a fixed priority order."""

from __future__ import annotations

from typing import Optional

from config.settings import Settings, get_settings
from guardrails.age_gate import AgeGate
from guardrails.competitor import CompetitorGuardrail
from guardrails.medical_legal import MedicalLegalGuardrail
from guardrails.models import GuardrailDecision
from guardrails.off_topic import OffTopicGuardrail
from guardrails.pricing import PricingGuardrail
from guardrails.responsible_drinking import ResponsibleDrinkingGuardrail
from utils.logging import get_logger

logger = get_logger(__name__)


class GuardrailOrchestrator:
    """Evaluates inbound queries before retrieval and polishes outbound answers."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        age_gate: Optional[AgeGate] = None,
        pricing: Optional[PricingGuardrail] = None,
        responsible: Optional[ResponsibleDrinkingGuardrail] = None,
        competitor: Optional[CompetitorGuardrail] = None,
        medical: Optional[MedicalLegalGuardrail] = None,
        off_topic: Optional[OffTopicGuardrail] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.age_gate = age_gate or AgeGate(self.settings)
        self.pricing = pricing or PricingGuardrail()
        self.responsible = responsible or ResponsibleDrinkingGuardrail()
        self.competitor = competitor or CompetitorGuardrail()
        self.medical = medical or MedicalLegalGuardrail(self.settings)
        self.off_topic = off_topic or OffTopicGuardrail()

    def inspect_query(
        self,
        query: str,
        *,
        age_verified: bool = False,
        declared_age: Optional[int] = None,
        session_verified: bool = False,
        header_verified: bool = False,
    ) -> GuardrailDecision:
        age = self.age_gate.check(
            query=query,
            age_verified=age_verified,
            declared_age=declared_age,
            session_verified=session_verified,
            header_verified=header_verified,
        )
        if age.blocked:
            logger.info("guardrail_blocked", policy=age.policy)
            return age

        for checker in (self.off_topic, self.competitor, self.medical, self.pricing):
            decision = checker.check(query)
            if decision.blocked:
                logger.info("guardrail_blocked", policy=decision.policy)
                return decision

        drinking = self.responsible.check(query)
        logger.info(
            "guardrail_passed",
            needs_responsible_drinking=drinking.needs_responsible_drinking,
        )
        return GuardrailDecision.allow(
            policy="allow",
            needs_responsible_drinking=drinking.needs_responsible_drinking,
            metadata={"age_verified": True},
        )

    def polish_answer(self, answer: str, *, needs_responsible_drinking: bool) -> str:
        return self.responsible.apply(answer, required=needs_responsible_drinking)
