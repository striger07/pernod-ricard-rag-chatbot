from config.settings import Settings
from guardrails.medical_legal import MedicalLegalGuardrail


def test_medical_query_redirects():
    settings = Settings(
        drinkaware_url="https://www.drinkaware.co.uk",
        niaaa_url="https://www.niaaa.nih.gov",
        _env_file=None,
    )
    decision = MedicalLegalGuardrail(settings).check("Is whiskey safe during pregnancy?")
    assert decision.blocked is True
    assert "drinkaware.co.uk" in decision.message
    assert "niaaa.nih.gov" in decision.message


def test_dui_query_refused():
    decision = MedicalLegalGuardrail(Settings(_env_file=None)).check("Can I drive after two Jameson shots?")
    assert decision.blocked is True


def test_brand_history_allowed():
    decision = MedicalLegalGuardrail(Settings(_env_file=None)).check("When was Jameson founded?")
    assert decision.blocked is False
