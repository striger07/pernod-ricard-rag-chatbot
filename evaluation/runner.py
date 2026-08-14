"""Evaluation entrypoint."""

from __future__ import annotations

import sys

from evaluation.ragas_eval import main as ragas_main


def run() -> int:
    return ragas_main()


if __name__ == "__main__":
    sys.exit(run())
