"""Pytest pass/fail logger for the assignment test suite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

RESULTS_PATH = Path("data/indexes/pytest_results.log")


def pytest_sessionstart(session) -> None:  # type: ignore[no-untyped-def]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        f"# Pytest session {datetime.now(timezone.utc).isoformat()}\n",
        encoding="utf-8",
    )


def pytest_runtest_logreport(report) -> None:  # type: ignore[no-untyped-def]
    if report.when != "call":
        return
    status = "PASS" if report.passed else "FAIL" if report.failed else "SKIP"
    line = f"{status}\t{report.nodeid}\n"
    with RESULTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line)
