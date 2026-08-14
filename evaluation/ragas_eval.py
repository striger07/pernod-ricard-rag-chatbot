"""RAGAS evaluation over the live RAG architecture."""

from __future__ import annotations

import json
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
from rag.llm_service import GrokLLMService
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
            scores[target] = float(value)
        except (TypeError, ValueError):
            continue
    return scores


def run_ragas_metrics(rows: list[dict[str, Any]], settings: Settings) -> dict[str, float]:
    """Execute RAGAS Faithfulness, Context Precision, and Answer Relevancy."""
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to run RAGAS judge metrics")

    judge = ChatOpenAI(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        base_url=settings.groq_api_base,
        temperature=0,
        timeout=settings.groq_timeout_seconds,
    )
    wrapped = LangchainLLMWrapper(judge)
    samples = [
        SingleTurnSample(
            user_input=str(row["question"]),
            retrieved_contexts=list(row.get("contexts") or []),
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
            ResponseRelevancy(llm=wrapped),
        ],
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


class RagasEvaluator:
    """Runs the production RAG stack, then RAGAS metrics, then writes a report."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        engine: Optional[RagEngine] = None,
        ragas_fn: Optional[RagasFn] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or RagEngine(
            settings=self.settings,
            guardrails=GuardrailOrchestrator(self.settings),
            retriever=HybridRetrievalPipeline(self.settings),
            llm=GrokLLMService(self.settings),
            boundary=HallucinationBoundary(self.settings),
            sessions=SessionStore(),
        )
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
            response = await self.engine.chat(request, header_verified=True)
            contexts = [chunk.content for chunk in retrieval.chunks if chunk.content.strip()]
            row = {
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "category": sample["category"],
                "answer": response.answer,
                "contexts": contexts,
                "confidence": retrieval.confidence,
                "blocked": response.blocked,
                "policy": response.policy,
            }
            rows.append(row)
            logger.info(
                "ragas_sample_collected",
                category=sample["category"],
                confidence=retrieval.confidence,
                contexts=len(contexts),
                policy=response.policy,
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
