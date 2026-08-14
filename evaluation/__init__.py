"""Evaluation package."""

from evaluation.dataset import EVALUATION_DATASET
from evaluation.ragas_eval import RagasEvaluator

__all__ = ["EVALUATION_DATASET", "RagasEvaluator"]
