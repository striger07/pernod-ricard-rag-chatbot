"""Structured Pernod Ricard evaluation dataset for RAGAS."""

from __future__ import annotations

from typing import TypedDict


class EvaluationSample(TypedDict):
    question: str
    ground_truth: str
    category: str


EVALUATION_DATASET: list[EvaluationSample] = [
    {
        "question": "What is Absolut vodka and where is it produced?",
        "ground_truth": "Absolut is a Swedish vodka brand in the Pernod Ricard portfolio, associated with production in Ahus, Sweden.",
        "category": "product_knowledge",
    },
    {
        "question": "What is the heritage of Jameson Irish whiskey?",
        "ground_truth": "Jameson is a triple-distilled Irish whiskey in the Pernod Ricard portfolio, typically described as smooth and used in mixed drinks as well as sipped neat.",
        "category": "brand_history",
    },
    {
        "question": "Tell me about Chivas Regal.",
        "ground_truth": "Chivas Regal is a blended Scotch whisky in the Pernod Ricard portfolio with origins linked to the Chivas brothers in Aberdeen. The 12-year-old expression is widely known.",
        "category": "brand_history",
    },
    {
        "question": "How do I serve a Beefeater gin and tonic?",
        "ground_truth": "Beefeater is a London Dry gin. A classic serve is Beefeater with tonic water and a citrus garnish, enjoyed in moderation by adults of legal drinking age.",
        "category": "cocktail",
    },
    {
        "question": "What is The Glenlivet known for?",
        "ground_truth": "The Glenlivet is a Speyside single malt Scotch whisky in the Pernod Ricard portfolio, often described with citrus, pineapple, and vanilla notes.",
        "category": "product_knowledge",
    },
    {
        "question": "Which champagne houses are in the Pernod Ricard portfolio?",
        "ground_truth": "G.H. Mumm and Perrier-Jouët are Champagne houses in the Pernod Ricard portfolio.",
        "category": "product_knowledge",
    },
    {
        "question": "What is Malibu?",
        "ground_truth": "Malibu is a Caribbean rum liqueur with coconut flavour owned by Pernod Ricard, commonly used in tropical mixed drinks.",
        "category": "product_knowledge",
    },
    {
        "question": "Who owns Kahlúa?",
        "ground_truth": "Kahlúa is a coffee liqueur in the Pernod Ricard portfolio, used in coffee-based cocktails such as the Espresso Martini.",
        "category": "product_knowledge",
    },
]


def dataset_as_dicts() -> list[dict[str, str]]:
    return [dict(sample) for sample in EVALUATION_DATASET]
