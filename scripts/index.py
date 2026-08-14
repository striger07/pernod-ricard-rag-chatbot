"""CLI: chunk, embed, and index into Qdrant."""

from ingestion.pipeline import main

if __name__ == "__main__":
    summary = main(["--index-only"])
    print(summary)
