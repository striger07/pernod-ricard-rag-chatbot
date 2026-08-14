from config.settings import Settings
from guardrails.age_gate import AgeGate


def test_unverified_is_blocked():
    gate = AgeGate(Settings(age_gate_enabled=True, minimum_legal_drinking_age=18, _env_file=None))
    decision = gate.check(query="Tell me about Absolut", age_verified=False)
    assert decision.blocked is True
    assert decision.policy == "age_gate"


def test_verified_adult_is_allowed():
    gate = AgeGate(Settings(age_gate_enabled=True, _env_file=None))
    decision = gate.check(query="Tell me about Absolut", age_verified=True, declared_age=25)
    assert decision.blocked is False


def test_declared_underage_blocked_even_if_flag_set():
    gate = AgeGate(Settings(age_gate_enabled=True, minimum_legal_drinking_age=18, _env_file=None))
    decision = gate.check(query="Absolut vodka", age_verified=True, declared_age=16)
    assert decision.blocked is True
    assert decision.metadata.get("underage") is True


def test_query_admits_underage():
    gate = AgeGate(Settings(age_gate_enabled=True, _env_file=None))
    decision = gate.check(query="I am 16, tell me about Jameson", age_verified=True)
    assert decision.blocked is True


def test_session_verified_allows_follow_up():
    gate = AgeGate(Settings(age_gate_enabled=True, _env_file=None))
    decision = gate.check(query="What is Chivas Regal?", age_verified=False, session_verified=True)
    assert decision.blocked is False
