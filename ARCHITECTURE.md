# Architecture Deep-Dive

This document explains the reasoning behind every significant design decision in the
Ask My Docs pipeline. Read this when you want to extend, replace, or scale a component.

---

## Table of Contents

1. [Why no LangChain or LlamaIndex?](#why-no-langchain-or-llamaindex)
2. [Chunking design](#chunking-design)
3. [Why rank_bm25 instead of OpenSearch](#why-rank_bm25-instead-of-opensearch)
4. [Reciprocal Rank Fusion explained](#reciprocal-rank-fusion-explained)
5. [Why a cross-encoder after a bi-encoder?](#why-a-cross-encoder-after-a-bi-encoder)
6. [ChromaDB persistence model](#chromadb-persistence-model)
7. [Citation enforcement layers](#citation-enforcement-layers)
8. [Evaluation metric design](#evaluation-metric-design)
9. [CI gate philosophy](#ci-gate-philosophy)
10. [Logger design](#logger-design)
11. [Scaling path](#scaling-path)

---

## Why no LangChain or LlamaIndex?

Both frameworks are excellent for prototyping. For a production system that you need to
understand, debug, and trust, they add significant friction:

- **Hidden behaviour.** Framework abstractions change between minor versions. A silent
  upgrade to a prompt template or retrieval strategy can break your system without any
  obvious error.
- **Debugging difficulty.** When an answer is wrong, you want to see exactly which chunks
  were retrieved, what scores they had, and what prompt was sent. Frameworks often make
  this opaque or require framework-specific debugging tools.
- **Coupling.** LangChain-style pipelines couple retrieval, reranking, and generation in
  ways that make it hard to swap one component without touching others.

This codebase is about 2,200 lines of fully transparent Python. Every step is explicit,
every variable is named, and every component can be unit-tested in isolation.

If you want to add LangChain later, the `RAGPipeline` class in `rag_pipeline.py` is the
only entry point — replace its internals without touching `cli.py` or the evaluation code.

---

## Chunking design

### Why 500–800 tokens?

This range is a well-validated empirical sweet spot:

- **Too small (< 200 tokens):** A chunk may be a single sentence with no surrounding
  context. The embedding captures the surface form rather than the meaning. Retrieval
  precision drops because the chunk does not represent a complete thought.
- **Too large (> 1000 tokens):** The embedding has to represent too many topics at once.
  When the user asks about topic A, a large chunk covering topics A, B, and C may rank
  lower than a focused smaller chunk. Reranking also becomes less effective.
- **500–800 tokens:** Covers 2–4 paragraphs — enough context for the LLM to generate
  a coherent sentence, but focused enough for the embedding to represent a single idea.

### Why token-based, not character-based?

The embedding model and the LLM both operate on tokens, not characters. A character limit
of, say, 2000 characters can produce chunks ranging from 300 to 700 tokens depending on
the language and vocabulary. Token-based chunking ensures consistent chunk density.

### Why three-level fallback?

Real documents are messy. Academic papers have multi-page paragraphs. Contract documents
have unnumbered lists treated as single blocks. Configuration files have no sentence
structure at all. The three-level fallback (paragraph → sentence → word) handles all cases
without requiring format-specific logic for each document type.

### Why 100-token overlap?

Facts stated at the end of one paragraph and facts stated at the start of the next are
semantically related. Without overlap, a query about a concept that bridges two chunks
may retrieve neither chunk at high rank. 100 tokens (~1 sentence) is enough to carry
the topic context forward without making the chunks redundant.

---

## Why rank_bm25 instead of OpenSearch?

The spec originally called for OpenSearch. `rank_bm25` was chosen for this implementation
for the following reasons:

| Dimension | rank_bm25 | OpenSearch |
|-----------|-----------|------------|
| Setup | `pip install rank_bm25` | Java JVM + Docker + cluster config |
| Startup time | < 1 second (pickle load) | 30–60 seconds |
| Memory | ~50 MB for 10k chunks | ~500 MB minimum |
| Windows compatibility | Perfect | Requires WSL2 or Docker Desktop |
| API complexity | 3 lines of Python | REST API + index mapping + query DSL |
| Production suitability | Single process only | Horizontally scalable |

For a laptop-based prototype that needs to run without errors and without hitting setup
issues, `rank_bm25` is the correct choice.

**The swap path to OpenSearch is intentionally simple.** The entire BM25 interaction is
isolated to two methods in `retriever.py`:

```python
def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
    # Replace this method's body with OpenSearch REST calls.
    # The return type (List[Dict] with text/source/chunk_index/score) stays the same.
    # Everything above this method — RRF merge, reranking, generation — is unchanged.
    ...
```

When you are ready to scale:
1. Stand up an OpenSearch cluster (locally or on AWS OpenSearch Service).
2. Replace `_load_bm25_index` with index creation logic.
3. Replace `_bm25_search` with an OpenSearch match query.
4. The rest of the pipeline is unaffected.

---

## Reciprocal Rank Fusion explained

RRF was introduced by Cormack et al. (2009) as a way to combine ranked lists from
multiple retrieval systems without requiring score normalisation.

**The problem with score fusion:** BM25 produces unbounded scores (e.g. 12.4, 8.1, 3.2).
Vector similarity scores are bounded between -1 and 1 for normalised embeddings.
Averaging them directly is meaningless — you'd be adding apples to oranges.

**RRF's solution:** Ignore the scores. Only use the ranks.

```
rrf_score(result, k=60) = 1 / (k + rank)
```

For a result at rank 0 (best): `1 / (60 + 0) = 0.0167`
For a result at rank 10: `1 / (60 + 10) = 0.0143`

When a result appears in multiple lists, its scores are summed. A result at rank 1
in both BM25 and vector search gets score `0.0164 + 0.0164 = 0.0328`, which beats
any result that only appears in one list.

**Why k=60?** The constant k dampens the advantage of very high ranks. Without it,
a rank-1 result would get an infinite score advantage. k=60 means the difference
between rank 1 and rank 2 is `1/61 - 1/62 = 0.00026` — small, as it should be.
The value 60 is empirically validated across many retrieval benchmarks.

---

## Why a cross-encoder after a bi-encoder?

This two-stage architecture is sometimes called a "retrieve and rerank" pipeline.

**Stage 1 — Bi-encoder (fast, approximate)**

The embedding model encodes the query and each chunk independently into a 384-dimensional
vector. Similarity is computed as cosine distance between these independent vectors.

- Scales to millions of chunks because embeddings are precomputed at index time.
- Approximate: the model cannot see the query and chunk together, so it cannot evaluate
  how well the chunk answers this specific question.

**Stage 2 — Cross-encoder (slow, accurate)**

The cross-encoder takes the concatenation `[CLS] query [SEP] chunk [SEP]` as a single
input and outputs a single relevance score. It can attend to both texts simultaneously.

- Cannot scale to millions of chunks (must run at query time for each pair).
- Accurate: the model explicitly models the relationship between query and chunk.

**The combination:** Use the bi-encoder to narrow from millions to ~20 candidates cheaply.
Then use the cross-encoder on those 20 to find the true top 5. You get the scalability
of approximate search and the accuracy of full cross-attention — at the cost of one
additional model inference on a small set.

**Model choice:** `ms-marco-MiniLM-L-6-v2` is trained on Microsoft's MS MARCO passage
ranking dataset — exactly the task of scoring passage relevance to a query. It is small
enough to run in < 100 ms on CPU for 20 pairs, which keeps end-to-end latency acceptable.

---

## ChromaDB persistence model

ChromaDB is initialised with `PersistentClient(path=str(CHROMA_DIR))`. This creates
a local SQLite-backed store at `data/indexes/chroma/`. The store persists across process
restarts — there is no in-memory mode in production use.

**Collection configuration:**
```python
self.collection = self.chroma_client.get_or_create_collection(
    name="ask_my_docs",
    metadata={"hnsw:space": "cosine"},
)
```

The `hnsw:space: cosine` setting tells ChromaDB to use cosine distance for approximate
nearest neighbour search via the HNSW (Hierarchical Navigable Small World) algorithm.
This matches the L2-normalised embeddings produced by `sentence-transformers` — if the
embeddings are normalised, cosine distance and dot product give identical rankings.

**Deduplication:** Chunk IDs are deterministic MD5 hashes. ChromaDB's `get()` call
checks which IDs already exist before adding. Re-ingesting an unchanged document adds
exactly zero new chunks and makes no database writes.

---

## Citation enforcement layers

The system has two independent layers. Either one alone is insufficient.

**Layer 1 — Score-based gate (reranker.py)**

This layer operates before any LLM call. The cross-encoder scores every candidate chunk
on a continuous scale. The `MIN_RELEVANCE_SCORE` threshold drops chunks that are simply
not relevant to the query — even if they are the best available results.

If all chunks are dropped, the `Generator.generate()` method receives an empty list
and immediately returns a "could not find" response. Zero API tokens are consumed.

This layer handles questions that are completely out of domain — "what is the weather
on Mars?" — where no chunk in the index is relevant.

**Layer 2 — Prompt-based enforcement (generator.py + prompts/answer_v1.txt)**

This layer operates inside the LLM call. The prompt template instructs the model with
explicit rules that are easy for a well-aligned LLM to follow:

1. Answer ONLY using the provided chunks.
2. Every claim must have a `[chunk_N]` citation.
3. If chunks are insufficient, say "I could not find".

The `_extract_cited_sources` function then parses the `[chunk_N]` markers from the
response and maps them back to source filenames. If the model produces an answer with
no citations, a warning is logged — this indicates the model may have drawn on
parametric knowledge outside the provided context.

**Why two layers?** Layer 1 cannot prevent hallucination if the model ignores the prompt.
Layer 2 cannot prevent the LLM from being called with irrelevant chunks. Together they
form a defence-in-depth approach to grounded generation.

---

## Evaluation metric design

All evaluation metrics are computed locally with no external API calls. This was a
deliberate choice over using RAGas or LLM-as-judge approaches.

**Trade-offs:**

| Approach | Accuracy | Cost | Determinism |
|----------|----------|------|-------------|
| LLM-as-judge (e.g. RAGas) | High — understands semantics | API cost per eval run | Non-deterministic |
| NLP-based (e.g. BERTScore) | Medium — contextual embeddings | Free, moderate compute | Deterministic |
| Word-overlap (our approach) | Lower — surface-level proxy | Free, instant | Deterministic |

The word-overlap approach was chosen because:
1. CI runs on every push — API costs add up quickly.
2. Deterministic metrics give reproducible pass/fail signals.
3. For the purpose of catching regressions (did quality drop?), a consistent proxy
   metric is more useful than a more accurate but noisy LLM score.

**Interpretation guidance:**

- `faithfulness_score` below 0.70 most commonly means the model is paraphrasing heavily.
  Check the actual answers in the JSON report before assuming hallucination.
- `hallucination_flag` is conservative — it only flags answers where a sentence has
  less than 20% word overlap with any approved chunk. False positives can occur when
  the model uses synonyms extensively.
- `unanswerable_accuracy` below 0.80 is the most actionable signal — it means the
  citation guard (`MIN_RELEVANCE_SCORE`) is too permissive and letting irrelevant chunks
  through to the LLM.

---

## CI gate philosophy

The CI gate is designed around a single principle: **regressions should be caught before
they reach production, not after users notice them.**

**What counts as a regression?**
- A prompt change that causes the model to stop citing sources (citation_coverage drops).
- A dependency upgrade that changes embedding behaviour (retrieval_recall drops).
- A config change that makes the system too permissive about unanswerable questions.
- Accidentally lowering `RERANK_TOP_K` such that the LLM receives fewer context chunks.

**How the gate works in practice:**

1. Developer changes something.
2. They run `python cli.py eval` locally to get a new report.
3. They commit the report to `logs/`.
4. On push to `main`, Job 3 in CI reads the committed report and checks it against thresholds.
5. If any metric regressed below threshold, the build fails.

**Why use a committed report rather than running eval in CI?**

Running the full evaluation in CI would require a Gemini API key as a GitHub Actions secret
and would incur API costs on every main branch push. Using a committed report means:
- CI is free (no API calls in CI)
- The report is version-controlled alongside the code change that produced it
- The developer sees the regression immediately before pushing, not 5 minutes later

**Adjusting thresholds:**

Thresholds live in `evaluation/ci_gate.py` in the `CI_THRESHOLDS` dict. When you add
more documents to your corpus or tune the pipeline, re-run eval, verify the new scores
are an improvement, then tighten the thresholds to match the new baseline.

---

## Logger design

The `QueryLogger` appends one JSON line per query to `logs/queries.jsonl`. Design choices:

**Append-only, not a database.** The log file is the source of truth. It is easy to
parse with standard tools (`jq`, Python `json.loads`), easy to tail with `cli.py logs`,
and easy to archive by date. A SQLite database would be more queryable but adds
infrastructure overhead and the risk of corruption on abrupt shutdown.

**Non-blocking by design.** The logger wraps every write in a `try/except` with a bare
`pass` on exception. A disk-full error or a permissions error on the log file must never
crash a production pipeline query. The logger is observability infrastructure — it must
never become a point of failure.

**Text truncation.** Chunk text is truncated to 300 characters in the log. Full chunk
text is already stored in ChromaDB. The log captures enough to understand what was
retrieved without bloating the JSONL file with kilobytes of text per entry.

**Session IDs.** Each `QueryLogger` instance generates a random 8-character hex session
ID on creation. Interactive sessions (`cli.py ask`) create one logger instance for the
duration of the session, so all queries from a single interactive run share the same
session ID. This makes it easy to group related queries when debugging.

---

## Scaling path

When your document corpus grows beyond what a single laptop can handle, here is the
recommended upgrade sequence. Each step is independent — you do not need to do all of them.

**Step 1 — Replace BM25 with OpenSearch**

Replace `_bm25_search` and `_load_bm25_index` in `retriever.py` with OpenSearch REST calls.
All other code is unchanged. OpenSearch handles 100M+ documents and supports shard replication.

**Step 2 — Replace ChromaDB with a managed vector store**

ChromaDB supports a server mode and cloud hosting (Chroma Cloud). Alternatively, swap
to Pinecone, Weaviate, or Qdrant by replacing `_vector_search` in `retriever.py`.
The return type signature stays the same.

**Step 3 — Upgrade the embedding model**

Replace `all-MiniLM-L6-v2` with `all-mpnet-base-v2` (higher quality, slower) or
`text-embedding-3-small` via OpenAI's API (no local compute needed). Change `EMBEDDING_MODEL`
in `config.py` and re-ingest all documents to rebuild the index with the new embeddings.

**Step 4 — Add a reranking API**

Replace `cross-encoder/ms-marco-MiniLM-L-6-v2` with Cohere Rerank or a hosted model for
higher accuracy on domain-specific queries. Replace the `Reranker` class body only.

**Step 5 — Wrap in a FastAPI service**

`RAGPipeline` is already a stateful class with a single `query()` method — it maps
directly onto a POST endpoint. Create a `serve.py` with `FastAPI`, instantiate
`RAGPipeline` at startup, and expose `POST /query`. The CLI remains for local use.

**Step 6 — Add a React/Next.js frontend**

The FastAPI service becomes the backend. The `cli.py` output format (answer + sources table)
maps directly to a chat-style UI with source cards below each answer.
