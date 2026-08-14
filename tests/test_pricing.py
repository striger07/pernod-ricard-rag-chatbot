from guardrails.pricing import PricingGuardrail


def test_price_question_is_refused_with_official_url():
    decision = PricingGuardrail().check("How much does Absolut vodka cost?")
    assert decision.blocked is True
    assert decision.policy == "pricing"
    assert "absolut.com" in (decision.redirect_url or "")
    assert "cannot provide prices" in decision.message.lower()


def test_where_to_buy_is_refused():
    decision = PricingGuardrail().check("Where can I buy Jameson whiskey?")
    assert decision.blocked is True
    assert "jamesonwhiskey.com" in (decision.redirect_url or "")


def test_heritage_question_is_allowed():
    decision = PricingGuardrail().check("Where is Absolut vodka produced?")
    assert decision.blocked is False
