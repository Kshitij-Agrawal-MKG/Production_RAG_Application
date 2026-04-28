# Ask My Docs — Production RAG System

> Domain-specific document Q&A with hybrid retrieval, cross-encoder reranking,
> citation enforcement, full observability via Langfuse, and CI-gated evaluation.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [CLI Reference](#cli-reference)
6. [Configuration](#configuration)
7. [Phase 1 — Fundamentals](#phase-1--fundamentals)
8. [Phase 2 — Hybrid Retrieval and Reranking](#phase-2--hybrid-retrieval-and-reranking)
9. [Phase 3 — Evaluation and CI Gate](#phase-3--evaluation-and-ci-gate)
10. [Monitoring and Observability](#monitoring-and-observability)
11. [Running Tests](#running-tests)
12. [Troubleshooting](#troubleshooting)
13. [Tech Stack](#tech-stack)
14. [Demo Video](#demo-video)

---

## Overview

Ask My Docs is a three-phase production RAG (Retrieval-Augmented Generation) system
that answers natural-language questions about your documents using grounded, cited responses.
It is designed to run entirely on a laptop with no cloud databases, no servers, and no GPU.

**What makes it production-grade:**

- **Hybrid retrieval** — BM25 keyword search and vector similarity are fused with
  Reciprocal Rank Fusion, covering both exact-match and semantic gaps.
- **Cross-encoder reranking** — a second model scores each `(query, chunk)` pair
  directly, producing more accurate top-k selection than cosine similarity alone.
- **Citation enforcement** — two independent layers prevent hallucination: a
  cross-encoder score threshold before the API call, and prompt-level `[chunk_N]`
  citation requirements inside the LLM call.
- **Full Langfuse tracing** — every pipeline query becomes a trace with spans for
  retrieval, reranking, and LLM generation. Token counts, latency, and quality scores
  are attached to each trace.
- **P50/P95 metrics** — a SQLite-backed MetricsStore tracks latency percentiles, cost
  per request, citation coverage, and failure rate over configurable time windows.
- **Prompt versioning** — every prompt file is SHA-256 hashed and registered in a
  manifest. CI fails if a prompt was silently modified without a version bump.
- **Offline evaluation** — a 29-entry golden dataset aligned to the sample knowledge
  base runs through the pipeline. A CI gate fails the build if quality drops below
  defined thresholds.

---

## Architecture

```
INDEXING PIPELINE
─────────────────
Documents (PDF / DOCX / MD / HTML / TXT)
    │
    ▼
Chunker  (500–800 tokens, 100 token overlap, tiktoken-accurate)
    │
    ├──────────────────────────────────────┐
    ▼                                      ▼
Embedder                             BM25 Index
(all-MiniLM-L6-v2)                   (rank_bm25 → .pkl file)
    │
    ▼
ChromaDB  (persistent local vector store, cosine similarity)


QUERY PIPELINE
──────────────
User question
    │
    ├────────────────────────┐
    ▼                        ▼
BM25 search             Vector search
(keyword, top-10)       (cosine sim, top-10)
    │                        │
    └────────────┬───────────┘
                 ▼
           RRF Merger                         ← Langfuse span: retrieval
    (Reciprocal Rank Fusion, k=60)
                 │
                 ▼
     Cross-Encoder Reranker                   ← Langfuse span: reranking
     (ms-marco-MiniLM-L-6-v2)
                 │
                 ▼
         Citation Guard
         (drop chunks below MIN_RELEVANCE_SCORE;
          no chunks → no answer, no API call)
                 │
                 ▼
     Gemini 1.5 Flash                         ← Langfuse generation
     (prompt + [chunk_N] citation rules)      (prompt, answer, token counts)
                 │
                 ▼
     Answer + [chunk_N] citations
                 │
                 ├──────────────────────────── MetricsStore (SQLite)
                 │                             (latency, cost, quality)
                 │
                 └──────────────────────────── QueryLogger (JSONL)
                                               (full query detail)


EVALUATION PIPELINE
───────────────────
data/golden_dataset.json (29 aligned Q&A pairs)
    │
    ▼
Evaluator  →  logs/eval_YYYYMMDD_HHMMSS.json
    │
    ▼
CI Gate Check  →  .github/workflows/ci.yml  →  fail build on regression


MONITORING STACK
────────────────
Langfuse Cloud  ←  PipelineTracer  (trace per query, spans per step)
MetricsStore    ←  RAGPipeline     (P50/P95 latency, cost, quality)
Dashboard       ←  cli.py          (terminal view of all metrics)
PromptVersioner ←  CI              (SHA-256 integrity + version manifest)
```

---

## Project Structure

```
ask_my_docs/
│
├── cli.py                              Entry point — all CLI commands
├── rag_pipeline.py                     Orchestrator: fully instrumented
├── ingest.py                           Document parsing, chunking, dual indexing
├── retriever.py                        Hybrid BM25 + vector search with RRF
├── reranker.py                         Cross-encoder reranking + citation gate
├── generator.py                        Gemini answer generation + token counting
├── config.py                           All settings in one place
│
├── prompts/
│   ├── answer_v1.txt                   Active LLM prompt template
│   └── versions.json                   Prompt version manifest (SHA-256 hashes)
│
├── monitoring/
│   ├── __init__.py
│   ├── tracer.py                       Langfuse pipeline tracing
│   ├── metrics_store.py                SQLite store: P50/P95, cost, quality
│   ├── token_counter.py                Token counting for cost tracking
│   ├── dashboard.py                    Rich terminal monitoring dashboard
│   └── prompt_versioner.py             Prompt integrity + version management
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                      7 evaluation metric functions (local)
│   ├── evaluator.py                    Golden dataset runner, JSON reports
│   ├── ci_gate.py                      Threshold enforcement, exit 0/1
│   └── logger.py                       Append-only JSONL query logger
│
├── data/
│   ├── docs/
│   │   └── sample_kb/                  Sample knowledge base (HelixDB product)
│   │       ├── 01_overview_and_pricing.md
│   │       ├── 02_authentication_security_api.md
│   │       └── 03_data_integrations_support.md
│   ├── indexes/                        Auto-created: chroma/ + bm25_index.pkl
│   └── golden_dataset.json             29 Q&A pairs aligned to sample KB
│
├── tests/
│   ├── test_chunker.py                 8 tests  — chunking logic
│   ├── test_retriever.py               4 tests  — RRF merge logic
│   ├── test_generator.py               7 tests  — citation parsing
│   ├── test_metrics.py                 38 tests — evaluation metrics
│   ├── test_ci_gate.py                 12 tests — threshold logic
│   ├── test_evaluator.py               18 tests — evaluator + logger
│   └── test_monitoring.py              42 tests — MetricsStore, tokens,
│                                                   PromptVersioner, Tracer
│
├── logs/
│   ├── queries.jsonl                   Auto-created: JSONL query log
│   ├── metrics.db                      Auto-created: SQLite metrics store
│   └── eval_YYYYMMDD_HHMMSS.json       Saved evaluation reports
│
├── .github/
│   └── workflows/
│       └── ci.yml                      4-job GitHub Actions pipeline
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── ARCHITECTURE.md
└── CONTRIBUTING.md
```

---

## Quick Start

### Prerequisites

- Windows 10/11, Python 3.10+
- Internet connection for first-time model download (~170 MB, one-time)
- Free Gemini API key: https://aistudio.google.com/app/apikey
- (Optional) Free Langfuse account: https://cloud.langfuse.com

### Step 1 — Virtual environment

```cmd
cd C:\Projects\ask_my_docs
python -m venv venv
venv\Scripts\activate
```

### Step 2 — Install dependencies

```cmd
pip install -r requirements.txt
```

Downloads two models on first run (cached after):
- `all-MiniLM-L6-v2` — embedding model, ~90 MB
- `ms-marco-MiniLM-L-6-v2` — cross-encoder reranker, ~80 MB

### Step 3 — Configure API keys

```cmd
copy .env.example .env
```

Open `.env` and fill in your keys:

```
GEMINI_API_KEY=AIzaSyYourActualKeyHere

# Optional — enables Langfuse tracing (pipeline works without it)
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Step 4 — Ingest the sample knowledge base

```cmd
python cli.py ingest data\docs\sample_kb\
```

### Step 5 — Ask your first question

```cmd
python cli.py ask "What is the HelixDB refund policy?"
```

The answer will show the cited source, token usage, cost, and latency.

---

## CLI Reference

Activate your virtual environment before every session: `venv\Scripts\activate`

### Core commands

```cmd
python cli.py ingest <path>              Ingest file or directory
python cli.py ask "your question"        Single-shot Q&A
python cli.py ask --verbose "..."        Q&A with retrieval scores shown
python cli.py ask                        Interactive Q&A loop
python cli.py stats                      Index chunk counts and paths
python cli.py clear                      Wipe all indexes (with confirmation)
```

### Evaluation commands

```cmd
python cli.py eval                       Full evaluation on golden dataset
python cli.py eval --limit 5             Quick smoke-test (5 questions only)
python cli.py eval --ci                  Exit 1 if any threshold fails
python cli.py eval --categories factual  Filter by category
python cli.py logs                       JSONL query log summary
python cli.py logs --tail 20             Last 20 queries as a table
python cli.py ci-check                   Gate check latest eval report
python cli.py ci-check logs\eval_X.json  Gate check specific report
```

### Monitoring commands

```cmd
python cli.py dashboard                  24-hour monitoring dashboard
python cli.py dashboard --window 6       Last 6 hours
python cli.py dashboard --window 168     Last 7 days

python cli.py prompt-version status      Current prompt version health
python cli.py prompt-version verify      Check on-disk hash vs manifest
python cli.py prompt-version register \
  --version answer_v2 \
  --description "Added chain-of-thought" \
  --changelog "CoT prefix before answer"
```

---

## Configuration

All settings live in `config.py`. Edit there — applies everywhere.

### Chunking

| Setting | Default | Effect |
|---------|---------|--------|
| `CHUNK_SIZE` | `600` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap tokens between adjacent chunks |
| `MIN_CHUNK_TOKENS` | `50` | Discard chunks shorter than this |

### Retrieval

| Setting | Default | Effect |
|---------|---------|--------|
| `VECTOR_TOP_K` | `10` | Candidates from ChromaDB |
| `BM25_TOP_K` | `10` | Candidates from BM25 index |
| `RERANK_TOP_K` | `5` | Chunks passed to LLM after reranking |
| `RRF_K` | `60` | RRF fusion constant |

### Quality and cost

| Setting | Default | Effect |
|---------|---------|--------|
| `MIN_RELEVANCE_SCORE` | `0.0` | Cross-encoder floor. Set `-3.0` to be stricter. |
| `GEMINI_TEMPERATURE` | `0.2` | Lower = more factual |
| `GEMINI_MAX_TOKENS` | `1024` | Max answer length in tokens |

---

## Phase 1 — Fundamentals

**Core files:** `ingest.py`, `retriever.py` (vector path), `generator.py`, `prompts/answer_v1.txt`

### Document parsing

| Format | Library |
|--------|---------|
| PDF | PyMuPDF (`fitz`) |
| DOCX | python-docx |
| HTML | BeautifulSoup + lxml |
| Markdown | markdown + BeautifulSoup |
| TXT | Built-in |

### Chunking strategy

Three-level recursive fallback: paragraph → sentence → word.
Token counting uses `tiktoken` cl100k_base. Chunk IDs are deterministic
MD5 hashes — re-ingesting an unchanged file adds zero new chunks.

### Prompt versioning

The active prompt lives in `prompts/answer_v1.txt`. Its SHA-256 is stored in
`prompts/versions.json`. CI fails if the file is modified without registering
a new version. See [Prompt versioning](#prompt-versioning) for details.

---

## Phase 2 — Hybrid Retrieval and Reranking

**Core files:** `retriever.py`, `reranker.py`

### Why hybrid retrieval?

Vector search misses exact-match queries (error codes, proper nouns).
BM25 misses paraphrases and synonyms. RRF fusion covers both failure modes
without requiring score normalisation — it only uses ranks.

`rank_bm25` replaces OpenSearch for laptop use. Swapping in OpenSearch is a
one-method change in `retriever._bm25_search()`. Everything above it is unchanged.

### Cross-encoder vs bi-encoder

The bi-encoder embeds query and chunk independently — fast but approximate.
The cross-encoder takes the full `(query, chunk)` pair — slow but accurate.
We run the bi-encoder over millions of chunks, then the cross-encoder over ≤20
merged candidates. Best of both: scalability + accuracy.

---

## Phase 3 — Evaluation and CI Gate

**Core files:** `evaluation/`, `data/golden_dataset.json`

### Sample knowledge base

`data/docs/sample_kb/` contains three Markdown documents for a fictional SaaS
product called **HelixDB** (a managed time-series database). Every entry in
`data/golden_dataset.json` has a `source_hint` pointing to the exact file and
section that contains the answer.

Documents:
- `01_overview_and_pricing.md` — product overview, features, pricing plans, Docker deployment
- `02_authentication_security_api.md` — API keys, password reset, MFA, encryption, rate limits, roles
- `03_data_integrations_support.md` — export, retention, integrations, SLAs, refunds, support contacts

### Golden dataset — 29 entries

25 answerable questions (all directly traceable to the sample KB) and 4
unanswerable questions to test the citation guard. Every answerable entry has:
- `expected_answer` — key phrase that must appear in the answer
- `expected_keywords` — list used for retrieval recall scoring
- `source_hint` — which KB file contains the answer

**Sample entries:**

| ID | Question | Source |
|----|----------|--------|
| gd_001 | What is the refund policy? | `03_data_integrations_support.md` |
| gd_008 | What are the API rate limits for the Growth Plan? | `02_authentication_security_api.md` |
| gd_023 | What query language does HelixDB use? | `01_overview_and_pricing.md` |
| gd_unanswerable_001 | What is the stock price of HelixDB? | *(not in KB)* |

### Evaluation metrics

All metrics are computed locally with no API calls.

| Metric | Threshold | Measures |
|--------|-----------|----------|
| `retrieval_recall` | ≥ 60% | Expected keywords in retrieved chunks |
| `retrieval_precision` | ≥ 50% | Relevant chunks in top-5 |
| `citation_coverage` | ≥ 60% | Answer sentences with `[chunk_N]` |
| `faithfulness_score` | ≥ 70% | Answer sentences with ≥35% chunk word overlap |
| `hallucination_rate` | ≤ 15% | Answers with < 20% chunk word overlap |
| `answer_accuracy` | ≥ 60% | Answerable questions answered correctly |
| `unanswerable_accuracy` | ≥ 80% | Unanswerable questions correctly abstained |

### Running evaluation

```cmd
python cli.py eval                  # run all 29 entries
python cli.py eval --limit 5        # quick smoke-test
python cli.py eval --ci             # exit 1 on regression

# Commit the report to activate the CI gate
git add logs\eval_*.json prompts\versions.json
git commit -m "eval: baseline evaluation report"
git push
```

---

## Monitoring and Observability

**Core files:** `monitoring/`

This layer answers: *"What caused the quality drop last Tuesday?"*

### Phase 1 — Langfuse tracing

Every pipeline query is a Langfuse **trace** containing:

```
trace: rag_query
  ├── span: retrieval
  │     input:  query text
  │     output: merged chunk count, top-5 with RRF scores
  │
  ├── span: reranking
  │     input:  candidate count
  │     output: approved chunks with cross-encoder scores
  │
  └── generation: llm_call
        input:  full prompt (versioned)
        output: answer text
        usage:  {input_tokens, output_tokens, total}
        model:  gemini-1.5-flash
```

Quality scores attached to each trace:
- `citation_coverage` — fraction of answer sentences with `[chunk_N]`
- `faithfulness` — word-overlap faithfulness proxy
- `was_answered` — 0 or 1

**Setup:**
1. Sign up free at https://cloud.langfuse.com
2. Create a project → Settings → API Keys → copy both keys
3. Add to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

Tracing degrades gracefully: if keys are not set, all tracing calls are
silent no-ops. The pipeline runs identically without Langfuse.

### Phase 2 — Quality metrics over time

`monitoring/metrics_store.py` writes one row per query to `logs/metrics.db`
(SQLite). Fields recorded per query:

| Field | Type | Description |
|-------|------|-------------|
| `latency_ms` | float | End-to-end pipeline time |
| `retrieval_latency_ms` | float | BM25 + vector + RRF time |
| `reranking_latency_ms` | float | Cross-encoder scoring time |
| `generation_latency_ms` | float | Gemini API time |
| `input_tokens` | int | Prompt token count |
| `output_tokens` | int | Completion token count |
| `cost_usd` | float | Estimated cost (Gemini 1.5 Flash rates) |
| `citation_coverage` | float | Fraction of cited answer sentences |
| `faithfulness_score` | float | Word-overlap faithfulness |
| `was_answered` | bool | Did the system produce an answer? |
| `retrieval_count` | int | Chunks before reranking |
| `approved_count` | int | Chunks after citation gate |
| `prompt_version` | str | Which prompt file was used |
| `error` | str | Error message if pipeline failed |

**Cost calculation** uses Gemini 1.5 Flash pricing:
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

**Terminal dashboard:**

```cmd
python cli.py dashboard              # last 24 hours
python cli.py dashboard --window 6   # last 6 hours
python cli.py dashboard --window 168 # last 7 days
```

The dashboard shows three side-by-side panels (Performance / Quality / Cost),
three trend tables (Latency / Quality / Cost, bucketed hourly), and an
anomaly detector that flags:
- P95 latency spikes above 10,000 ms
- Citation coverage drops below 50%
- Failure rate above 25%

Example output:
```
╭─────────────────────────╮ ╭─────────────────────────╮ ╭──────────────────────╮
│ Performance             │ │ Quality                 │ │ Cost                 │
│                         │ │                         │ │                      │
│ P50 latency  : 1,240 ms │ │ Failure rate  :   3.2%  │ │ Total cost : $0.0024 │
│ P95 latency  : 3,810 ms │ │ Citation cov. :  87.4%  │ │ Cost/query : $0.0003 │
│ Mean latency : 1,650 ms │ │ Faithfulness  :  79.1%  │ │ Tokens in  : 12,400  │
│ Total queries:      8   │ │ Error rate    :   0.0%  │ │ Tokens out :  1,850  │
│ Answered     :      8   │ │                         │ │                      │
╰─────────────────────────╯ ╰─────────────────────────╯ ╰──────────────────────╯
```

### Phase 3 — Prompt versioning and regression gating

Every prompt file has its SHA-256 stored in `prompts/versions.json`.
CI Job 3 (`prompt-integrity`) calls `PromptVersioner.check_for_changes()` and
fails the build if the file was modified without registering a new version.

**Workflow for changing a prompt:**

```cmd
# 1. Copy the current prompt to a new file
copy prompts\answer_v1.txt prompts\answer_v2.txt

# 2. Edit answer_v2.txt with your changes

# 3. Update generator.py to load the new version
#    Change: _PROMPT_VERSION = "answer_v1"
#    To:     _PROMPT_VERSION = "answer_v2"

# 4. Register the new version in the manifest
python cli.py prompt-version register \
  --version answer_v2 \
  --description "Added chain-of-thought prefix" \
  --changelog "Adds 'Think step by step:' before answer generation"

# 5. Run evaluation to validate the change
python cli.py eval --ci

# 6. Commit everything together
git add prompts\answer_v2.txt prompts\versions.json \
        generator.py logs\eval_*.json
git commit -m "feat: upgrade to answer_v2 prompt with CoT"
```

The CI pipeline (`ci.yml`) has four jobs:

| Job | Runs on | What it checks |
|-----|---------|----------------|
| Unit Tests | every push + PR | `pytest tests/` — 110 tests, no API keys |
| Monitoring Checks | every push + PR | Golden dataset schema, metric functions, MetricsStore, token counter, prompt versioner |
| Prompt Integrity | every push + PR | SHA-256 of prompt files vs manifest |
| Evaluation Gate | main branch only | Committed eval report vs CI thresholds |

---

## Running Tests

```cmd
venv\Scripts\activate

# Full suite — 110 tests
pytest tests/ -v

# By module
pytest tests/test_monitoring.py -v          # 42 monitoring tests
pytest tests/test_metrics.py -v             # 38 metric tests
pytest tests/test_ci_gate.py -v             # 12 CI gate tests
pytest tests/test_evaluator.py -v           # 18 evaluator tests
pytest tests/test_chunker.py tests/test_retriever.py tests/test_generator.py -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing
```

No API keys, no model downloads, no network access required.
All tests use mocks for pipeline components and temp directories.

### Test summary

| File | Tests | Covers |
|------|-------|--------|
| `test_chunker.py` | 8 | Chunking, overlap, min-token filter |
| `test_retriever.py` | 4 | RRF score, deduplication, multi-source merge |
| `test_generator.py` | 7 | Citation extraction, formatting |
| `test_metrics.py` | 38 | All 7 evaluation metric functions |
| `test_ci_gate.py` | 12 | Threshold enforcement and operator logic |
| `test_evaluator.py` | 18 | Aggregation, report saving, logger |
| `test_monitoring.py` | 42 | MetricsStore, token counter, PromptVersioner, Tracer no-op |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'fitz'`**
```cmd
pip install PyMuPDF
```

**`GEMINI_API_KEY not set`**
- Confirm `.env` exists (not just `.env.example`)
- Confirm it contains `GEMINI_API_KEY=AIzaSy...` with no quotes
- Confirm `(venv)` is visible in your terminal prompt

**`BM25 index not found` or `ChromaDB not found`**
```cmd
python cli.py ingest data\docs\sample_kb\
```

**Answer is always "I could not find a reliable answer"**
```cmd
python cli.py stats                              # confirm chunks are indexed
python cli.py ask --verbose "your question"      # see rerank scores
```
If all rerank scores are very negative, lower `MIN_RELEVANCE_SCORE` in
`config.py` (e.g. from `0.0` to `-5.0`).

**Dashboard shows "No data in this window"**
Ask at least one question first: `python cli.py ask "test"`.
Each query writes one row to `logs/metrics.db`.

**Langfuse traces not appearing**
```cmd
python cli.py prompt-version status       # check current config
```
Confirm `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are in `.env`.
The pipeline prints `Langfuse tracing: enabled` on startup when keys are present.

**CI prompt integrity failure**
```
[PROMPT INTEGRITY FAILURE]
  Version   : answer_v1
  Stored    : abc123...
  On-disk   : def456...
```
The prompt file was modified without registering a version bump. Run:
```cmd
python cli.py prompt-version register \
  --version answer_v1_fixed \
  --description "Describe what changed"
```
Or revert the prompt file to its original content.

**`sqlite3` error on Windows with ChromaDB**
- Requires Python 3.10+. Check with `python --version`.
- Ensure the virtual environment is active.

---

## Tech Stack

| Component | Tool | Reason |
|-----------|------|--------|
| Document parsing | PyMuPDF, python-docx, BeautifulSoup | Best-in-class per format |
| Token counting | tiktoken (lazy-loaded) | Accurate token counts, graceful fallback |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Fast, CPU-only, high quality |
| Vector store | ChromaDB (local persistent) | Zero server, cosine similarity |
| BM25 | rank_bm25 (pure Python) | No server, instant load |
| RRF fusion | Custom 10-line implementation | Transparent, no black boxes |
| Cross-encoder | cross-encoder/ms-marco-MiniLM-L-6-v2 | Best accuracy/speed tradeoff |
| LLM | Gemini 1.5 Flash | Free tier, generous rate limits |
| Tracing | Langfuse (optional) | Per-step spans, token counts, scores |
| Metrics store | SQLite via stdlib | Zero deps, persistent, P50/P95 |
| Evaluation | Custom local metrics | No API calls, deterministic |
| Prompt versioning | SHA-256 + JSON manifest | Version-controlled prompt integrity |
| CI/CD | GitHub Actions (4 jobs) | Unit tests, monitoring, integrity, eval gate |
| CLI | argparse + Rich | Beautiful terminal output |

---
### Demo Video
Watch how the project works in this demo:

[![Watch the Demo](https://img.youtube.com/vi/QEpdJMcr3vA/0.jpg)](https://www.youtube.com/watch?v=QEpdJMcr3vA)