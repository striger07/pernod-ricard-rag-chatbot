# Pernod Ricard RAG Chatbot

Production Retrieval-Augmented Generation assistant for Pernod Ricard brand knowledge. The stack is FastAPI, hybrid retrieval (Qdrant + BM25 + RRF + MMR), Groq (`llama-3.3-70b-versatile`), mandatory policy guardrails, and a Streamlit interface.

This assistant is intended only for adults of legal drinking age. It never quotes prices, never advises on purchases, and never provides medical or legal advice.

## Architecture

```text
User
 ↓
Streamlit
 ↓
FastAPI
 ↓
Guardrails
 ↓
Query Processing
 ↓
Hybrid Retrieval
 ├── Dense → Qdrant
 └── BM25
 ↓
RRF
 ↓
MMR
 ↓
Confidence Boundary
 ↓
Groq LLM
 ↓
Citation + Response
 ↓
Streamlit
```

Request path in code:

1. Streamlit age-gate (session) plus FastAPI age-gate (server).
2. Policy orchestrator: off-topic, competitor, medical/legal, pricing.
3. Dense search in Qdrant and BM25 over the persisted corpus.
4. Reciprocal Rank Fusion (`k=60` by default), then MMR diversity.
5. If retrieval confidence is below `RETRIEVAL_CONFIDENCE_THRESHOLD`, the API returns exactly `I don't have that information.` and does not call Groq.
6. Otherwise Groq answers from untrusted retrieved documents with citation rules.
7. Cocktail/consumption answers receive a responsible-drinking warning.

## Setup

Python **3.10** and **Conda** (`pernod_rag`). Do not use `venv`.

```bash
conda create -n pernod_rag python=3.10 -y
conda activate pernod_rag
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY`. Optional: `ADMIN_API_TOKEN` for `/ingest`.

Start Qdrant, index knowledge, then run API and UI:

```bash
conda activate pernod_rag
docker compose up -d qdrant
make index
make run
```

In another terminal:

```bash
conda activate pernod_rag
make run-ui
```

- API: http://127.0.0.1:8000/health
- UI: http://127.0.0.1:8501

If live crawling is blocked, ingestion loads `data/synthetic/` automatically.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `APP_NAME` / `APP_ENV` / `APP_HOST` / `APP_PORT` | Application identity and bind address |
| `LOG_LEVEL` | Structured log level |
| `CORS_ALLOWED_ORIGINS` | Allowed browser origins (Streamlit) |
| `ADMIN_API_TOKEN` | Shared secret for `POST /ingest` |
| `GROQ_API_KEY` / `GROQ_API_BASE` / `GROQ_MODEL` | Groq OpenAI-compatible client (`llama-3.3-70b-versatile`) |
| `GROQ_EVAL_MODEL` | Model used for `make evaluate` generation and judging (`llama-3.1-8b-instant`) |
| `GROQ_EVAL_REQUESTS_PER_SECOND` | Spacing between RAGAS judge calls |
| `RAGAS_USE_LIBRARY_METRICS` | `false` uses compact RAGAS-style JSON scores; `true` uses the official ragas library (much higher Groq token use) |
| `GROQ_TIMEOUT_SECONDS` / `GROQ_MAX_TOKENS` / `GROQ_TEMPERATURE` | Generation controls |
| `OPENAI_API_KEY` / `OPENAI_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_DIMENSION` | Optional OpenAI embedding fallback |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | Primary BAAI/bge-m3 settings |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_MAX_RETRIES` / `EMBEDDING_DEVICE` | Embedder runtime |
| `QDRANT_HOST` / `QDRANT_PORT` / `QDRANT_GRPC_PORT` | Vector database |
| `QDRANT_API_KEY` / `QDRANT_HTTPS` / `QDRANT_COLLECTION` / `QDRANT_TIMEOUT_SECONDS` | Qdrant auth and collection |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | Hallucination cutoff |
| `DENSE_TOP_K` / `BM25_TOP_K` / `FINAL_TOP_K` | Candidate list sizes |
| `RRF_K` / `MMR_LAMBDA` / `MMR_CANDIDATES` | Fusion and diversity |
| `BM25_INDEX_PATH` / `BM25_K1` / `BM25_B` | Sparse index |
| `CHUNK_MAX_CHARS` / `CHUNK_MIN_CHARS` / `CHUNK_OVERLAP_CHARS` / `SEMANTIC_SIMILARITY_THRESHOLD` | Semantic chunking |
| `CRAWL_*` / `SYNTHETIC_DATA_DIR` | Crawler and fallback corpus |
| `AGE_GATE_ENABLED` / `MINIMUM_LEGAL_DRINKING_AGE` / `AGE_GATE_STRICT_SERVER_ENFORCEMENT` / `AGE_VERIFICATION_HEADER` | Age gate |
| `DRINKAWARE_URL` / `NIAAA_URL` | Medical/legal redirects |
| `BACKEND_URL` | Streamlit → FastAPI base URL |
| `RAGAS_FAITHFULNESS_THRESHOLD` / `RAGAS_CONTEXT_PRECISION_THRESHOLD` / `RAGAS_ANSWER_RELEVANCY_THRESHOLD` / `RAGAS_OUTPUT_DIR` | Evaluation |

Never commit `.env`. Secrets are never written to logs.

## Knowledge ingestion

Pipeline: **Crawl4AI → BeautifulSoup cleaning → semantic chunking → BAAI/bge-m3 embeddings → Qdrant + BM25**.

```bash
conda activate pernod_rag
make crawl    # fetch pages; falls back to data/synthetic/
make index    # chunk, embed, upsert Qdrant, persist BM25
```

Admin HTTP:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: $ADMIN_API_TOKEN" \
  -d '{"recreate": false}'
```

Metadata (`title`, `url`, `source`, `timestamp`) is preserved from crawl through citation.

## Retrieval strategy

- **Dense:** cosine search in Qdrant using `BAAI/bge-m3` (1024-d).
- **BM25:** `rank_bm25` over the chunked corpus, pickled at `BM25_INDEX_PATH`.
- **RRF:** `score = Σ 1 / (k + rank)` across the two lists (`k` default 60). Lists are fused, not concatenated.
- **MMR:** re-ranks the fused pool for diversity against the query embedding.
- **Confidence:** combination of dense, RRF, MMR, and dual-retriever agreement, clipped to `[0, 1]`. Below threshold → exact refusal string, no LLM call.

## Guardrails

| Policy | Behaviour |
| --- | --- |
| Age gate | First-visit confirmation in Streamlit **and** server-side check. Underage users are blocked from alcohol content. Frontend flags are not trusted alone. |
| Pricing | No prices, estimates, or purchase help. Redirect to the official brand site. |
| Responsible drinking | Cocktail/consumption answers include a short professional warning. |
| Competitor | No Diageo, Bacardi, Brown-Forman, or peer-brand comparisons/rankings. |
| Medical / legal | No professional advice. Redirect to Drinkaware and NIAAA. |
| Off-topic | Politics, harm, jailbreaks, and unrelated requests get a single-sentence refusal. |
| Tone | Premium, professional, warm, concise. No slang or emojis. |

Prompt-level rules treat retrieved documents as untrusted data and forbid fabricated citations.

## Evaluation

Unit/integration tests (no production credentials):

```bash
conda activate pernod_rag
make test
# or
pytest
```

Pass/fail lines are appended to `data/indexes/pytest_results.log`.

RAGAS (Faithfulness, Context Precision, Answer Relevancy) uses the **same** RAG engine and a Groq judge:

```bash
conda activate pernod_rag
make evaluate
```

Reports: `data/indexes/ragas_report.md` and timestamped JSON. Exit code `1` if a metric is below its threshold. `GROQ_API_KEY` is required. Evaluation uses `GROQ_EVAL_MODEL` so it does not consume the 70B daily token cap used by the chat API.

## Deployment

```bash
cp .env.example .env
docker compose up --build
```

Services:

- `qdrant` on `:6333`
- `api` on `:8000`
- `ui` on `:8501` (`BACKEND_URL=http://api:8000`)

Production notes:

- Put a reverse proxy (TLS) in front of API and UI.
- Set a strong `ADMIN_API_TOKEN`.
- Point `QDRANT_HOST` at a managed cluster when not using Compose.
- Replace the in-memory session store if you run multiple API replicas.
- First embedding load downloads `BAAI/bge-m3`; size the container memory accordingly.

## LLM choice

**Production chat:** Groq-hosted `llama-3.3-70b-versatile` via the OpenAI-compatible API (`https://api.groq.com/openai/v1`). Chosen for fast inference, low cost, and quality comparable to GPT-3.5+ for grounded brand Q&A. The assignment’s suggested models (GPT-4o, Claude Sonnet 3.5, Gemini 1.5 Pro) remain supported by the same `GrokLLMService` abstraction — change `GROQ_API_BASE` and `GROQ_MODEL` in `.env` to point at another OpenAI-compatible provider.

**Evaluation:** `make evaluate` uses `GROQ_EVAL_MODEL` (`llama-3.1-8b-instant` by default) for answer generation and RAGAS judging. This keeps evaluation off the 70B daily token cap used by the live chat API.

**Cost and limits:** Groq’s free tier imposes daily token limits on larger models (for example ~100k tokens/day on `llama-3.3-70b-versatile`). Expect occasional 429 rate-limit errors under heavy use. For a demo or submission, usage is typically negligible; for sustained production traffic, budget for a paid Groq tier or switch to GPT-4o / Claude / Gemini via the same service interface.

## Design decisions

- **Local BGE-M3** rather than OpenAI embeddings for the default path, so indexing can run without an OpenAI key. OpenAI remains configurable.
- **Hybrid retrieval** because brand names and heritage phrases benefit from lexical BM25 as well as dense similarity.
- **Hard confidence boundary** instead of letting the LLM hedge: unsupported questions return a single canonical sentence.
- **Guardrails as dedicated modules** plus prompt policy, matching the assignment’s “both middleware and prompt” rule.
- **Groq via the OpenAI SDK** — see [LLM choice](#llm-choice) for rationale, eval model, and quota notes. The generator is isolated behind `GrokLLMService`.
- **Synthetic Markdown fallback** so crawl blocks (network, age walls, bot protection) do not stop a demo index.
- **Conda `pernod_rag`** is the supported local workflow; Docker uses its own image Python 3.10 environment.

## Makefile

| Target | Action |
| --- | --- |
| `make install` | `conda activate pernod_rag` and install requirements |
| `make run` | FastAPI / uvicorn |
| `make run-ui` | Streamlit UI |
| `make test` | pytest |
| `make lint` | ruff |
| `make crawl` | ingestion crawl |
| `make index` | embed and index |
| `make evaluate` | RAGAS |
| `make docker-up` / `make docker-down` | Compose |
