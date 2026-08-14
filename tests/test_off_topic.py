from guardrails.off_topic import OffTopicGuardrail
from guardrails.responsible_drinking import RESPONSIBLE_DRINKING_WARNING, ResponsibleDrinkingGuardrail


def test_politics_refused_in_one_sentence():
    decision = OffTopicGuardrail().check("Who should I vote for in the election?")
    assert decision.blocked is True
    assert decision.policy == "off_topic"
    assert decision.message.count(".") >= 1
    assert "Pernod Ricard" in decision.message


def test_python_homework_refused():
    decision = OffTopicGuardrail().check("Write a Python script to scrape websites")
    assert decision.blocked is True


def test_brand_query_allowed():
    decision = OffTopicGuardrail().check("What is the heritage of Beefeater gin?")
    assert decision.blocked is False


def test_responsible_warning_appended_once():
    guard = ResponsibleDrinkingGuardrail()
    query_decision = guard.check("How do I make a Beefeater gin and tonic?")
    assert query_decision.needs_responsible_drinking is True
    answer = guard.apply("Serve Beefeater with tonic and a citrus garnish.", required=True)
    assert RESPONSIBLE_DRINKING_WARNING in answer
    again = guard.apply(answer, required=True)
    assert again.count(RESPONSIBLE_DRINKING_WARNING) == 1
