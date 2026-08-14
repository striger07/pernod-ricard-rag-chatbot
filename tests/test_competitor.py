from guardrails.competitor import CompetitorGuardrail


def test_diageo_comparison_refused():
    decision = CompetitorGuardrail().check("Is Absolut better than Smirnoff from Diageo?")
    assert decision.blocked is True
    assert decision.policy == "competitor"


def test_bacardi_mention_refused():
    decision = CompetitorGuardrail().check("Compare Malibu with Bacardi rum")
    assert decision.blocked is True


def test_portfolio_question_allowed():
    decision = CompetitorGuardrail().check("Tell me about Chivas Regal and The Glenlivet")
    assert decision.blocked is False
