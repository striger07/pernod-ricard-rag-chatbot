"""RAGAS evaluation over the live RAG architecture."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from api.schemas import ChatRequest
from api.sessions import SessionStore
from config.settings import Settings, get_settings
from evaluation.dataset import EVALUATION_DATASET
from guardrails.orchestrator import GuardrailOrchestrator
from rag.engine import RagEngine
from rag.hallucination import HallucinationBoundary
from rag.llm_service import GrokLLMService, LLMServiceError
from retrieval.pipeline import HybridRetrievalPipeline
from utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

RagasFn = Callable[[list[dict[str, Any]], Settings], dict[str, float]]


def _normalise_scores(raw: dict[str, Any]) -> dict[str, float]:
    mapping = {
        "faithfulness": "faithfulness",
        "context_precision": "context_precision",
        "llm_context_precision_without_reference": "context_precision",
        "answer_relevancy": "answer_relevancy",
        "answer_relevance": "answer_relevancy",
        "response_relevancy": "answer_relevancy",
    }
    scores: dict[str, float] = {}
    for key, value in raw.items():
        target = mapping.get(str(key).lower())
        if target is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        scores[target] = numeric
    return scores


_JUDGE_KEYS = ("faithfulness", "context_precision", "answer_relevancy")


def _clamp_score(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return max(0.0, min(1.0, numeric))


def parse_judge_payload(text: str) -> dict[str, float]:
    """Parse a compact RAGAS-style JSON score object from a model completion."""
    blob = text.strip()
    start = blob.find("{")
    end = blob.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("judge response did not contain a JSON object")
    payload = json.loads(blob[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge JSON must be an object")
    scores: dict[str, float] = {}
    for key in _JUDGE_KEYS:
        numeric = _clamp_score(payload.get(key))
        if numeric is None:
            raise ValueError(f"judge JSON missing numeric {key}")
        scores[key] = numeric
    return scores


def _truncate_contexts(contexts: list[str], *, limit: int = 4, chars: int = 500) -> list[str]:
    trimmed: list[str] = []
    for item in contexts[:limit]:
        text = " ".join(str(item).split())
        if len(text) > chars:
            text = text[: chars - 3] + "..."
        if text:
            trimmed.append(text)
    return trimmed


def _compact_judge_prompt(row: dict[str, Any]) -> str:
    contexts = _truncate_contexts(list(row.get("contexts") or []))
    numbered = "\n".join(f"{index}. {text}" for index, text in enumerate(contexts, start=1)) or "(none)"
    return (
        "You are evaluating a retrieval-augmented generation answer using RAGAS definitions.\n"
        "Return JSON only with exactly these keys: faithfulness, context_precision, answer_relevancy.\n"
        "Each value must be a float between 0 and 1.\n"
        "faithfulness: fraction of the answer that is supported by the retrieved contexts.\n"
        "context_precision: fraction of retrieved contexts that are relevant to the question.\n"
        "answer_relevancy: how well the answer addresses the question.\n\n"
        f"Question:\n{row.get('question')}\n\n"
        f"Answer:\n{row.get('answer')}\n\n"
        f"Reference:\n{row.get('ground_truth')}\n\n"
        f"Retrieved contexts:\n{numbered}\n"
    )


def run_compact_ragas_metrics(rows: list[dict[str, Any]], settings: Settings) -> dict[str, float]:
    """Score Faithfulness, Context Precision, and Answer Relevancy with one Groq JSON call per sample."""
    import time

    from openai import OpenAI

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to run RAGAS judge metrics")
    judge_model = settings.groq_eval_model or settings.groq_model
    interval = 1.0 / max(settings.groq_eval_requests_per_second, 0.01)
    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.groq_api_base,
        timeout=max(settings.groq_timeout_seconds, 60.0),
        max_retries=2,
    )
    logger.info("ragas_judge_configured", model=judge_model, mode="compact")
    collected: list[dict[str, float]] = []
    for index, row in enumerate(rows):
        if index:
            time.sleep(interval)
        completion = client.chat.completions.create(
            model=judge_model,
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You output JSON only. No markdown. No extra keys.",
                },
                {"role": "user", "content": _compact_judge_prompt(row)},
            ],
        )
        content = ""
        try:
            content = completion.choices[0].message.content or ""
        except (AttributeError, IndexError) as exc:
            raise RuntimeError("Groq RAGAS judge returned an empty completion") from exc
        collected.append(parse_judge_payload(content))
        logger.info("ragas_sample_scored", category=row.get("category"), **collected[-1])
    averages = {
        key: sum(item[key] for item in collected) / len(collected) for key in _JUDGE_KEYS
    }
    logger.info("ragas_raw_scores", **averages)
    return averages


def run_library_ragas_metrics(rows: list[dict[str, Any]], settings: Settings) -> dict[str, float]:
    """Execute official RAGAS metrics. Token-heavy; avoid on Groq free-tier daily caps."""
    from langchain_core.embeddings import Embeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
    from ragas.run_config import RunConfig

    from ingestion.embedder import BGEEmbedder

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to run RAGAS judge metrics")

    class LocalEmbeddings(Embeddings):
        def __init__(self) -> None:
            self._embedder = BGEEmbedder(settings)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._embedder.encode(texts)

        def embed_query(self, text: str) -> list[float]:
            encoded = self._embedder.encode([text])
            return encoded[0] if encoded else []

    judge_model = settings.groq_eval_model or settings.groq_model
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=settings.groq_eval_requests_per_second,
        check_every_n_seconds=0.25,
        max_bucket_size=1,
    )
    judge = ChatOpenAI(
        model=judge_model,
        api_key=settings.groq_api_key,
        base_url=settings.groq_api_base,
        temperature=0,
        timeout=max(settings.groq_timeout_seconds, 180.0),
        max_retries=10,
        max_completion_tokens=4096,
        rate_limiter=rate_limiter,
    )
    wrapped = LangchainLLMWrapper(judge)
    logger.info("ragas_judge_configured", model=judge_model, mode="library")
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())
    samples = [
        SingleTurnSample(
            user_input=str(row["question"]),
            retrieved_contexts=_truncate_contexts(list(row.get("contexts") or [])),
            response=str(row.get("answer") or ""),
            reference=str(row.get("ground_truth") or ""),
        )
        for row in rows
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=wrapped),
            LLMContextPrecisionWithoutReference(llm=wrapped),
            ResponseRelevancy(llm=wrapped, embeddings=embeddings),
        ],
        run_config=RunConfig(
            timeout=600,
            max_retries=10,
            max_wait=60,
            max_workers=1,
        ),
        batch_size=1,
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    numeric = frame.mean(numeric_only=True).to_dict()
    scores = _normalise_scores(numeric)
    logger.info(
        "ragas_raw_scores",
        faithfulness=scores.get("faithfulness"),
        context_precision=scores.get("context_precision"),
        answer_relevancy=scores.get("answer_relevancy"),
    )
    return scores


def run_ragas_metrics(rows: list[dict[str, Any]], settings: Settings) -> dict[str, float]:
    """Execute RAGAS Faithfulness, Context Precision, and Answer Relevancy."""
    if settings.ragas_use_library_metrics:
        return run_library_ragas_metrics(rows, settings)
    return run_compact_ragas_metrics(rows, settings)


class RagasEvaluator:
    """Runs the production RAG stack, then RAGAS metrics, then writes a report."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        engine: Optional[RagEngine] = None,
        ragas_fn: Optional[RagasFn] = None,
    ) -> None:
        self.settings = settings or get_settings()
        if engine is None:
            eval_llm_settings = self.settings.model_copy(
                update={
                    "groq_model": self.settings.groq_eval_model or self.settings.groq_model,
                }
            )
            engine = RagEngine(
                settings=self.settings,
                guardrails=GuardrailOrchestrator(self.settings),
                retriever=HybridRetrievalPipeline(self.settings),
                llm=GrokLLMService(eval_llm_settings),
                boundary=HallucinationBoundary(self.settings),
                sessions=SessionStore(),
            )
        self.engine = engine
        self.ragas_fn = ragas_fn or run_ragas_metrics

    async def collect_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for sample in EVALUATION_DATASET:
            request = ChatRequest(
                message=sample["question"],
                age_verified=True,
                declared_age=30,
            )
            retrieval = self.engine.retrieve(sample["question"])
            try:
                response = await self.engine.chat(request, header_verified=True)
                answer = response.answer
                blocked = response.blocked
                policy = response.policy
            except LLMServiceError as exc:
                logger.error("ragas_sample_llm_failed", category=sample["category"], error=str(exc))
                raise RuntimeError(
                    f"Groq generation failed for evaluation sample {sample['category']!r}: {exc}"
                ) from exc
            contexts = [chunk.content for chunk in retrieval.chunks if chunk.content.strip()]
            row = {
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "category": sample["category"],
                "answer": answer,
                "contexts": contexts,
                "confidence": retrieval.confidence,
                "blocked": blocked,
                "policy": policy,
            }
            rows.append(row)
            logger.info(
                "ragas_sample_collected",
                category=sample["category"],
                confidence=retrieval.confidence,
                contexts=len(contexts),
                policy=policy,
            )
        return rows

    def score(self, rows: list[dict[str, Any]]) -> dict[str, float]:
        return self.ragas_fn(rows, self.settings)

    def write_report(self, rows: list[dict[str, Any]], scores: dict[str, float], passed: bool) -> Path:
        output_dir = Path(self.settings.ragas_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "thresholds": {
                "faithfulness": self.settings.ragas_faithfulness_threshold,
                "context_precision": self.settings.ragas_context_precision_threshold,
                "answer_relevancy": self.settings.ragas_answer_relevancy_threshold,
            },
            "scores": scores,
            "samples": rows,
        }
        json_path = output_dir / f"ragas_report_{stamp}.json"
        md_path = output_dir / "ragas_report.md"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        lines = [
            "# RAGAS evaluation report",
            "",
            f"- Generated: {payload['generated_at']}",
            f"- Passed: {passed}",
            "",
            "| Metric | Score | Threshold |",
            "| --- | --- | --- |",
        ]
        for metric, threshold_attr in (
            ("faithfulness", self.settings.ragas_faithfulness_threshold),
            ("context_precision", self.settings.ragas_context_precision_threshold),
            ("answer_relevancy", self.settings.ragas_answer_relevancy_threshold),
        ):
            score = scores.get(metric)
            score_text = "n/a" if score is None else f"{score:.3f}"
            lines.append(f"| {metric} | {score_text} | {threshold_attr:.2f} |")
        lines.extend(["", "## Samples", ""])
        for row in rows:
            lines.append(f"### {row['category']}: {row['question']}")
            lines.append(f"- Confidence: {row.get('confidence')}")
            lines.append(f"- Policy: {row.get('policy')}")
            lines.append(f"- Answer: {row.get('answer')}")
            lines.append("")
        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("ragas_report_written", json_path=str(json_path), markdown_path=str(md_path))
        return md_path

    def passed(self, scores: dict[str, float]) -> bool:
        checks = (
            scores.get("faithfulness", 0.0) >= self.settings.ragas_faithfulness_threshold,
            scores.get("context_precision", 0.0) >= self.settings.ragas_context_precision_threshold,
            scores.get("answer_relevancy", 0.0) >= self.settings.ragas_answer_relevancy_threshold,
        )
        return all(checks)

    async def run(self) -> int:
        configure_logging(self.settings)
        logger.info("ragas_evaluation_start", samples=len(EVALUATION_DATASET))
        rows = await self.collect_rows()
        scores = self.score(rows)
        ok = self.passed(scores)
        path = self.write_report(rows, scores, ok)
        logger.info("ragas_evaluation_complete", passed=ok, report=str(path), **scores)
        print(json.dumps({"passed": ok, "scores": scores, "report": str(path)}, indent=2))
        return 0 if ok else 1


def main() -> int:
    import asyncio

    return asyncio.run(RagasEvaluator().run())


if __name__ == "__main__":
    sys.exit(main())
