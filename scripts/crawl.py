"""CLI: crawl knowledge sources."""

from ingestion.pipeline import main

if __name__ == "__main__":
    summary = main(["--crawl-only"])
    print(summary)
