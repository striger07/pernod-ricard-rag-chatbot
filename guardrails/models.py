"""Shared guardrail decision model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GuardrailDecision:
    """Outcome of a single policy check or the combined orchestrator result."""

    blocked: bool
    policy: str
    message: str = ""
    redirect_url: Optional[str] = None
    needs_responsible_drinking: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def allow(
        cls,
        policy: str = "allow",
        *,
        needs_responsible_drinking: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "GuardrailDecision":
        return cls(
            blocked=False,
            policy=policy,
            needs_responsible_drinking=needs_responsible_drinking,
            metadata=metadata or {},
        )

    @classmethod
    def deny(
        cls,
        policy: str,
        message: str,
        *,
        redirect_url: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "GuardrailDecision":
        return cls(
            blocked=True,
            policy=policy,
            message=message,
            redirect_url=redirect_url,
            metadata=metadata or {},
        )


def normalize_query(text: str) -> str:
    return " ".join((text or "").lower().split())
