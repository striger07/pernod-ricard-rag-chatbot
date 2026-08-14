from config.settings import Settings
from rag.hallucination import INSUFFICIENT_INFORMATION, HallucinationBoundary
from retrieval.models import RetrievalResult, RetrievedChunk


def test_refusal_text_is_exact():
    assert INSUFFICIENT_INFORMATION == "I don't have that information."
    assert HallucinationBoundary(Settings(_env_file=None)).refusal_text() == "I don't have that information."


def test_boundary_blocks_low_confidence():
    settings = Settings(retrieval_confidence_threshold=0.35, _env_file=None)
    boundary = HallucinationBoundary(settings)
    retrieval = RetrievalResult(query="unknown brand xyz", chunks=[], confidence=0.12)
    assert boundary.evaluate(retrieval) is False


def test_boundary_allows_high_confidence():
    settings = Settings(retrieval_confidence_threshold=0.35, _env_file=None)
    boundary = HallucinationBoundary(settings)
    chunk = RetrievedChunk(
        content="Absolut is a Swedish vodka.",
        title="Absolut",
        url="https://www.absolut.com/",
        source="Absolut",
        confidence_score=0.8,
    )
    retrieval = RetrievalResult(query="absolut origin", chunks=[chunk], confidence=0.72)
    assert boundary.evaluate(retrieval) is True


async def test_generate_or_refuse_bypasses_llm():
    settings = Settings(retrieval_confidence_threshold=0.35, _env_file=None)
    boundary = HallucinationBoundary(settings)
    retrieval = RetrievalResult(query="price of chivas", chunks=[], confidence=0.01)
    called = {"n": 0}

    async def generate():
        called["n"] += 1
        return "should not run"

    text = await boundary.generate_or_refuse(retrieval, generate)
    assert text == "I don't have that information."
    assert called["n"] == 0


async def test_generate_or_refuse_calls_llm_when_sufficient():
    settings = Settings(retrieval_confidence_threshold=0.35, _env_file=None)
    boundary = HallucinationBoundary(settings)
    retrieval = RetrievalResult(query="absolut", chunks=[], confidence=0.9)

    async def generate():
        return "Absolut is a Swedish vodka brand."

    text = await boundary.generate_or_refuse(retrieval, generate)
    assert text == "Absolut is a Swedish vodka brand."
