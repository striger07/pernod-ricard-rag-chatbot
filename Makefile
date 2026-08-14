SHELL := /bin/bash
.DEFAULT_GOAL := help

CONDA_ENV := pernod_rag
export PYTHONPATH := $(CURDIR)

CONDA_ACTIVATE := source "$$(conda info --base)/etc/profile.d/conda.sh" && conda activate $(CONDA_ENV)

.PHONY: help setup-env install run run-ui test lint format crawl index evaluate docker-up docker-down

help:
	@echo "Pernod Ricard RAG — commands (Conda env: $(CONDA_ENV))"
	@echo "  make setup-env   Create conda env pernod_rag (Python 3.10)"
	@echo "  make install     conda activate pernod_rag && pip install -r requirements.txt"
	@echo "  make run         Start FastAPI with uvicorn"
	@echo "  make run-ui      Start the Streamlit UI"
	@echo "  make test        Run pytest"
	@echo "  make lint        Run ruff"
	@echo "  make format      Run black + ruff --fix"
	@echo "  make crawl       Crawl knowledge sources (synthetic fallback if blocked)"
	@echo "  make index       Chunk, embed, and upsert into Qdrant + BM25"
	@echo "  make evaluate    Run RAGAS evaluation"
	@echo "  make docker-up   docker compose up --build"
	@echo "  make docker-down docker compose down"

setup-env:
	conda create -n $(CONDA_ENV) python=3.10 -y
	@echo "Created env $(CONDA_ENV). Next: conda activate $(CONDA_ENV) && make install"

install:
	$(CONDA_ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt

run:
	$(CONDA_ACTIVATE) && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	$(CONDA_ACTIVATE) && streamlit run frontend/app.py --server.port 8501

test:
	$(CONDA_ACTIVATE) && pytest -q

lint:
	$(CONDA_ACTIVATE) && ruff check .

format:
	$(CONDA_ACTIVATE) && black . && ruff check --fix .

crawl:
	$(CONDA_ACTIVATE) && python -m scripts.crawl

index:
	$(CONDA_ACTIVATE) && python -m scripts.index

evaluate:
	$(CONDA_ACTIVATE) && python -m evaluation.ragas_eval

docker-up:
	docker compose up --build

docker-down:
	docker compose down
