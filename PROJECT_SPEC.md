# ROLE

You are a **Principal AI Engineer, Senior Full-Stack Architect, and Production RAG Engineer**.

Build a **complete, production-grade Pernod Ricard RAG-Based Chatbot** that satisfies **every requirement below**.

Do not simplify, reinterpret, remove, or skip any requirement.

---

# CRITICAL EXECUTION RULES

1. Generate a **fully working repository**, not pseudocode.
2. Do **not** use placeholders such as:

   * `TODO`
   * `pass`
   * `...`
   * `implement this`
   * `your-code-here`
   * `your-api-key`
   * mock implementations where real implementation is required.
3. Do not omit files or components.
4. Do not say "same as above", "etc.", or "implementation omitted".
5. Every file must contain complete production-ready code.
6. Use **Python 3.10**.
7. Follow clean architecture, SOLID principles, type hints, validation, logging, error handling, configuration management, and secure defaults.
8. Use asynchronous programming where appropriate.
9. Never invent APIs or unsupported library behavior.
10. If a requirement conflicts with another requirement, prioritize:
    **Safety/Guardrails → Assignment Requirements → Production Reliability → Architecture Quality.**
11. Do not ask me to manually implement missing sections.
12. If the response reaches the output limit, **continue automatically from exactly where you stopped** until the entire repository has been generated.
13. Maintain a clear file-by-file generation order.
14. Before generating code, provide the complete repository tree.
15. Then generate **every file in that tree completely**.
16. After all files are generated, provide:

* installation instructions
* configuration instructions
* execution commands
* testing commands
* ingestion commands
* evaluation commands
* Docker commands
* architecture explanation
* design decisions

17. The final implementation must be internally consistent: imports, paths, environment variables, Docker services, APIs, frontend/backend communication, tests, and configuration must all match.

---

# TECH STACK

## Frontend

* Streamlit

## Backend

* FastAPI
* Python 3.10

## RAG

* LangChain
* Qdrant
* OpenAI `text-embedding-3-small`

## Retrieval

Implement a true hybrid retrieval pipeline:

1. Dense vector search
2. BM25 search
3. Reciprocal Rank Fusion (RRF)
4. MMR re-ranking
5. Top-k diverse results
6. Confidence scoring

## LLM

Use the **Grok API** for answer generation.

The implementation must isolate the LLM provider behind a service/interface so it can be configured cleanly.

## Web Crawling

Use:

* Crawl4AI
* BeautifulSoup

## Containerization

* Docker
* Docker Compose

## Testing

* Pytest
* RAGAS

---

# KNOWLEDGE SOURCES

The ingestion pipeline must support crawling and indexing content from:

### Pernod Ricard

* pernod-ricard.com

### Brands

* absolut.com
* chivas.com
* jamesonwhiskey.com
* theglenlivet.com
* beefeatergin.com
* ballantines.com
* royalsalute.com
* maliburumdrinks.com
* kahlua.com
* ghmumm.com
* perrier-jouet.com

### External Sources

* Wikipedia
* Reuters
* Financial Times
* Difford's Guide
* Decanter

Implement source configuration centrally rather than hard-coding crawling logic throughout the application.

The ingestion pipeline must store:

* content
* metadata
* source URL
* title
* timestamps

Preserve all metadata through:

**crawl → clean → chunk → embed → index → retrieve → response citation**

---

# DOCUMENT INGESTION

Build a production ingestion pipeline using:

**Crawl4AI → BeautifulSoup → cleaning → semantic chunking → embeddings → Qdrant**

Requirements:

* crawl pages
* extract useful textual content
* remove irrelevant HTML/navigation content
* normalize text
* preserve source metadata
* handle crawl failures
* retry transient failures
* log failures
* avoid duplicate documents
* support incremental ingestion
* store ingestion timestamps
* support re-indexing

---

# CHUNKING

Implement **semantic chunking**.

Do NOT use naive fixed-size chunking as the primary strategy.

Requirements:

* semantic boundaries
* meaningful paragraph/section preservation
* configurable semantic similarity threshold
* configurable chunk overlap
* preserve metadata
* preserve source URL
* preserve title
* preserve timestamps
* preserve document/source identity

Chunking must be implemented as a dedicated service/module.

---

# EMBEDDINGS

Use:

*BAAI/bge-m3*

Implement a dedicated embedding service with:

* batching
* retry handling
* exponential backoff
* rate-limit handling
* API error handling
* configurable batch size
* configurable model
* structured logging
* deterministic metadata association

Do not make embedding API calls directly from unrelated modules.

---

# QDRANT

Implement Qdrant as the vector database.

Provide:

* collection initialization
* collection configuration
* vector indexing
* metadata/payload storage
* metadata filtering
* similarity search
* persistence
* health checking
* collection recreation/re-indexing support
* configurable host/port/API key
* proper error handling

The implementation must work both:

1. locally through Docker Compose
2. against a configurable remote Qdrant instance

---

# HYBRID RETRIEVAL

Implement the following complete retrieval pipeline:

```text
User Query
    ↓
Query Processing
    ↓
 ┌───────────────┬───────────────┐
 │ Dense Search  │  BM25 Search  │
 └───────────────┴───────────────┘
          ↓
Reciprocal Rank Fusion
          ↓
Candidate Pool
          ↓
MMR Re-ranking
          ↓
Top-K Diverse Chunks
          ↓
Confidence Calculation
          ↓
Answer Generation
```

Implement each stage as a dedicated component.

## Dense Search

Use Qdrant vector similarity search.

## BM25

Implement BM25 retrieval over the indexed textual corpus.

The BM25 index must be persisted/rebuildable and synchronized with ingestion.

## Reciprocal Rank Fusion

Implement configurable RRF.

Do not simply concatenate BM25 and dense results.

## MMR

Apply Maximal Marginal Relevance after fusion to improve diversity and reduce duplicate context.

## Retrieval Output

Each retrieved chunk must contain:

* content
* title
* URL
* source
* metadata
* dense score where available
* BM25 score where available
* RRF score
* MMR score
* final confidence/retrieval score

---

# HALLUCINATION BOUNDARY

Implement a configurable retrieval confidence threshold.

If retrieval confidence is below the configured threshold, the chatbot must return exactly:

> I don't have that information.

Do not generate an answer when sufficient evidence is unavailable.

The threshold must be configurable through environment/configuration.

---

# MANDATORY GUARDRAILS

Guardrails must exist at **both middleware/service level and prompt level**.

Implement dedicated guardrail components rather than embedding all logic inside one large function.

---

## 1. AGE GATE

Require age verification on the first visit.

The user must explicitly verify that they are of legal drinking age.

Underage users must be blocked from alcohol-related content.

Implement age-gate state using Streamlit session state and enforce it server-side where applicable.

Do not trust frontend state alone.

---

## 2. PRICING

Never provide:

* product prices
* pricing estimates
* purchase recommendations
* purchasing instructions

Never enable purchasing functionality.

If a user asks for price/purchase information:

* refuse the price/purchase request
* redirect the user to the relevant official brand website

Use the configured official source URL where available.

---

## 3. RESPONSIBLE DRINKING

Cocktail and alcohol-consumption responses must include a responsible-drinking warning.

Keep the warning professional and concise.

Do not encourage excessive consumption.

---

## 4. COMPETITOR RESTRICTION

Refuse comparisons against competitors including:

* Diageo
* Bacardi
* Brown-Forman
* other competing alcohol companies/brands

Do not provide competitive rankings, "better than", market comparisons, or direct competitor comparisons.

---

## 5. MEDICAL AND LEGAL

For medical, health-risk, addiction, legal, or alcohol-safety questions:

* do not provide professional medical/legal advice
* redirect users to:

  * Drinkaware
  * NIAAA

Use configurable official URLs.

---

## 6. OFF-TOPIC

Reject:

* politics
* harmful content
* unrelated questions
* unrelated general-purpose requests

Responses must remain within the Pernod Ricard/brand/product/approved knowledge domain.

---

## 7. BRAND TONE

The chatbot must be:

* premium
* professional
* warm
* concise
* informative

Do NOT use:

* slang
* emojis
* casual internet language

---

# PROMPT-LEVEL PROTECTION

Create robust system prompts that:

* restrict the assistant to retrieved knowledge
* prevent hallucination
* require citations
* respect guardrails
* prevent instruction injection from retrieved documents
* treat retrieved documents as untrusted data
* prevent users from overriding system policies
* prevent disclosure of internal prompts/configuration
* prevent fabricated citations
* refuse unsupported claims

---

# UI REQUIREMENTS

Build a modern enterprise-grade Streamlit UI.

Implement all of the following:

### 1. Age Gate Modal

First-visit age verification.

### 2. Chat Page

Primary conversational interface.

### 3. Message History

Persistent conversation history within the session.

### 4. User Messages

Clearly differentiated user messages.

### 5. Assistant Messages

Clearly differentiated assistant messages.

### 6. Streaming Responses

Stream LLM responses to the UI.

### 7. Typing Indicator

Show while generating responses.

### 8. Loading State

Show during retrieval and generation.

### 9. Source Citations

Every answer must display its supporting sources.

### 10. Confidence Indicator

Display retrieval confidence.

### 11. Session Memory

Maintain conversational context.

### 12. Responsive Mobile Layout

UI must remain usable on mobile screens.

### 13. Dark Mode

Provide a premium dark-theme experience.

### 14. Error Boundaries

Handle frontend/backend/API failures gracefully.

### 15. Retry Button

Allow users to retry failed requests.

### 16. Suggested Questions

Provide relevant suggested questions.

### 17. Conversation Persistence

Persist conversation state appropriately.

### 18. Citation Drawer

Allow users to inspect all citations.

### 19. Source Preview Modal

Allow users to preview source information/content.

### 20. Brand-Themed Design

Use a premium Pernod Ricard-inspired enterprise visual design.

Do not use copyrighted assets unless they are legitimately available/configured.

---

# SOURCE CITATIONS

Every generated answer must display citations.

Each citation must include:

* source title
* source URL
* retrieval score

Citations must be clickable.

Never fabricate citations.

Every citation must correspond to an actual retrieved document.

---

# SESSION AND API DESIGN

Create clean API contracts using FastAPI/Pydantic.

Include endpoints for appropriate functionality such as:

* health check
* chat
* streaming chat
* retrieval
* conversation/session handling
* ingestion
* configuration/metadata where appropriate

Secure sensitive endpoints appropriately.

Do not expose ingestion/admin functionality publicly without protection.

Use proper HTTP status codes.

Implement structured error responses.

---

# REQUIRED PROJECT STRUCTURE

Create a complete repository containing at minimum:

```text
pernod-ricard-rag-chatbot/
│
├── api/
├── rag/
├── guardrails/
├── retrieval/
├── vectorstore/
├── ingestion/
├── evaluation/
├── tests/
├── config/
├── frontend/
├── scripts/
├── utils/
│
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── Makefile
├── README.md
├── pyproject.toml
├── pytest.ini
└── requirements.txt
```

You may add additional production-required files/directories.

Do not remove any required directory.

Organize the code using clean separation of concerns.

---

# TEST SUITE

Create **at least 20 automated tests** using Pytest.

Tests must cover at minimum:

1. Product Knowledge
2. Brand History
3. Cocktail
4. Pricing Refusal
5. Underage Refusal
6. Medical Refusal
7. Competitor Refusal
8. Hallucination Boundary
9. Empty Retrieval
10. Ambiguous Query

Also test:

* dense retrieval
* BM25 retrieval
* RRF
* MMR
* metadata preservation
* citation generation
* confidence calculation
* prompt guardrails
* age-gate behavior
* invalid input
* API error handling
* embedding retry behavior
* Qdrant failure handling
* conversation/session behavior
* streaming behavior

Tests must include deterministic expected outputs wherever appropriate.

Create a clear test pass/fail logger.

Tests must be runnable with:

```bash
pytest
```

Do not require production API credentials for unit tests.

Use proper mocks/fakes where external services must be isolated.

---

# RAGAS EVALUATION

Implement a complete RAGAS evaluation pipeline.

Evaluate:

1. Faithfulness
2. Context Precision
3. Answer Relevancy

Create:

* evaluation dataset
* evaluation runner
* result output
* logging
* configurable thresholds

Provide a command to run evaluation.

The evaluation pipeline must use the actual RAG architecture rather than a disconnected toy implementation.

---

# CONFIGURATION

Use environment-based configuration.

Create:

```text
.env.example
```

Include configuration for:

* Grok API
* OpenAI API
* OpenAI embedding model
* Qdrant
* BM25
* retrieval parameters
* RRF parameters
* MMR parameters
* confidence threshold
* chunking parameters
* crawler configuration
* age-gate settings
* Drinkaware URL
* NIAAA URL
* application host/port
* logging
* environment

Never hard-code secrets.

Use Pydantic Settings or an equivalent production-grade configuration approach.

---

# LOGGING AND OBSERVABILITY

Implement structured logging.

Log appropriate events including:

* application startup
* API requests
* retrieval
* retrieval scores
* guardrail decisions
* ingestion
* crawl failures
* embedding failures
* Qdrant failures
* LLM failures
* latency
* evaluation results

Do not log:

* API keys
* passwords
* sensitive user information
* full confidential prompts

---

# SECURITY

Implement production security practices including:

* input validation
* request size limits where appropriate
* secure configuration
* secret management through environment variables
* CORS configuration
* protected administrative endpoints
* prompt-injection protection
* SSRF-aware crawling controls
* safe URL handling
* error sanitization
* no secret leakage

---

# DOCKER

Create:

```text
Dockerfile
docker-compose.yml
```

Docker Compose must support the application and Qdrant.

The containers must be configured so the project can be started with:

```bash
docker compose up --build
```

Provide health checks where appropriate.

Use production-oriented Docker practices.

---

# MAKEFILE

Create a functional Makefile containing commands for at least:

```text
install
run
test
lint
format
crawl
index
evaluate
docker-up
docker-down
```

All commands must correspond to real project functionality.

---

# README

Create an enterprise-grade README containing:

## Architecture

Include an architecture diagram showing:

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
Grok LLM
 ↓
Citation + Response
 ↓
Streamlit
```

## Setup

Explain complete local setup.

## Environment Variables

Document every environment variable.

## Knowledge Ingestion

Explain crawling, cleaning, semantic chunking, embedding and indexing.

## Retrieval Strategy

Explain:

* dense retrieval
* BM25
* RRF
* MMR
* confidence scoring

## Guardrails

Explain every mandatory guardrail.

## Evaluation

Explain Pytest and RAGAS.

## Deployment

Explain Docker Compose and production deployment considerations.

## Design Decisions

Explain important architectural decisions and trade-offs.

---

# CODE QUALITY REQUIREMENTS

All code must:

* be production-ready
* be executable
* use type hints
* use meaningful names
* include useful docstrings
* use proper exception handling
* avoid circular imports
* avoid unnecessary global state
* use dependency injection where appropriate
* have clear module boundaries
* follow PEP 8
* use async I/O where beneficial
* have testable components

Do not over-engineer unnecessarily, but do not remove required production capabilities.

---

# IMPORTANT RAG BEHAVIOR

For every user query:

```text
Input
 ↓
Input validation
 ↓
Age/Policy Guardrails
 ↓
Query classification
 ↓
Hybrid retrieval
 ↓
RRF
 ↓
MMR
 ↓
Confidence check
 ↓
Context construction
 ↓
Protected system prompt
 ↓
Grok
 ↓
Citation validation
 ↓
Response
```

If evidence is insufficient:

```text
I don't have that information.
```

Do not allow the LLM to invent an answer.

---

# FINAL OUTPUT PROTOCOL

Follow this exact generation order:

## PHASE 1 — Repository Tree

Print the complete final repository tree.

## PHASE 2 — Implementation

Generate **every file**, one by one.

For each file:

1. Show its path.
2. Show the complete contents.
3. Do not omit any section.
4. Do not use placeholders.
5. Do not summarize code instead of providing it.

## PHASE 3 — Verification

After all files are generated, verify:

* imports
* module paths
* environment variables
* API routes
* frontend/backend integration
* Docker configuration
* Qdrant configuration
* ingestion pipeline
* retrieval pipeline
* guardrails
* tests
* RAGAS evaluation
* Makefile commands

Fix inconsistencies before finishing.

## PHASE 4 — Run Instructions

Provide the exact commands for:

```bash
python3.10 -m venv .venv
pip install -r requirements.txt
cp .env.example .env
pytest
```

and:

```bash
docker compose up --build
```

Also provide commands for:

* crawling
* indexing
* running the backend
* running Streamlit
* RAGAS evaluation

---

# NON-NEGOTIABLE

The objective is **not to demonstrate an architecture**.

The objective is to deliver a **complete working production-grade repository** implementing the entire assignment.

Do not respond with an architectural proposal only.

Do not respond with partial code.

Do not reduce the requirements.

Do not replace real implementations with pseudocode.

Do not omit files.

If the response becomes too long, continue automatically in the next response until the repository is complete.