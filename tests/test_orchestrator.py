from config.settings import Settings
from guardrails.orchestrator import GuardrailOrchestrator


def test_orchestrator_priority_age_before_pricing():
    settings = Settings(age_gate_enabled=True, _env_file=None)
    orchestrator = GuardrailOrchestrator(settings)
    decision = orchestrator.inspect_query("What is the price of Absolut?")
    assert decision.blocked is True
    assert decision.policy == "age_gate"


def test_orchestrator_flags_responsible_drinking_after_pass():
    settings = Settings(age_gate_enabled=True, _env_file=None)
    orchestrator = GuardrailOrchestrator(settings)
    decision = orchestrator.inspect_query(
        "How do I make a Jameson ginger highball?",
        age_verified=True,
        declared_age=30,
    )
    assert decision.blocked is False
    assert decision.needs_responsible_drinking is True
    polished = orchestrator.polish_answer("Serve Jameson with ginger ale.", needs_responsible_drinking=True)
    assert "drink responsibly" in polished.lower()
