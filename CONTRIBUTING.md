# Contributing to Ask My Docs

This document explains how to extend the project — adding documents, growing the golden
dataset, writing tests, swapping components, and versioning prompts.

---

## Table of Contents

1. [Development setup](#development-setup)
2. [Adding documents](#adding-documents)
3. [Growing the golden dataset](#growing-the-golden-dataset)
4. [Writing tests](#writing-tests)
5. [Versioning prompts](#versioning-prompts)
6. [Swapping a component](#swapping-a-component)
7. [Tuning evaluation thresholds](#tuning-evaluation-thresholds)
8. [Code style](#code-style)
9. [Commit conventions](#commit-conventions)

---

## Development setup

```cmd
git clone <your-repo-url>
cd ask_my_docs
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Confirm everything works:

```cmd
pytest tests/ -v          # should show 87 passed
python cli.py stats       # should show 0 chunks (nothing ingested yet)
```

---

## Adding documents

1. Copy your files into `data\docs\` (or any subdirectory).
2. Run `python cli.py ingest data\docs\`.
3. Run `python cli.py stats` to confirm the chunk count increased.
4. Test a question: `python cli.py ask "your question here"`.

**Supported formats:** `.pdf`, `.docx`, `.md`, `.markdown`, `.html`, `.htm`, `.txt`

**Safe to re-run.** Already-indexed chunks are skipped. You can run `ingest` on the
same folder after adding new files — only the new files will be processed.

**To replace a document:** Run `python cli.py clear`, then `python cli.py ingest data\docs\`
to rebuild the entire index from scratch. There is no per-document delete at this time.

---

## Growing the golden dataset

`data/golden_dataset.json` is the primary quality signal for the system. The more
representative it is of real user questions, the more trustworthy the evaluation metrics are.

### Entry schema

```json
{
  "id":                "gd_001",
  "question":          "What is the cancellation policy?",
  "expected_answer":   "cancellation",
  "expected_keywords": ["cancel", "policy", "refund", "days"],
  "source_hint":       "terms_and_conditions.pdf",
  "category":          "factual",
  "difficulty":        "easy",
  "notes":             "High-frequency user question from support tickets"
}
```

### Field descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique string identifier. Use `gd_NNN` for general entries. |
| `question` | Yes | The natural-language question as a user would type it. |
| `expected_answer` | Yes | A short keyword that should appear in the answer. Use `null` for unanswerable questions. |
| `expected_keywords` | Yes | List of keywords expected in the retrieved chunks or answer. Can be empty `[]`. |
| `source_hint` | No | Filename hint for documentation purposes. Not used by the evaluator. |
| `category` | Yes | One of: `factual`, `technical`, `procedural`, `unanswerable` |
| `difficulty` | Yes | One of: `easy`, `medium`, `hard` |
| `notes` | No | Human-readable note about why this entry exists. |

### Categories

| Category | Use for |
|----------|---------|
| `factual` | Questions with a single correct answer ("What is the return window?") |
| `technical` | Questions requiring technical knowledge ("How does rate limiting work?") |
| `procedural` | Step-by-step questions ("How do I reset my password?") |
| `unanswerable` | Questions whose answers are not in any ingested document |

### How many entries to aim for

| Stage | Target | Goal |
|-------|--------|------|
| Initial baseline | 20–30 | Enough to catch obvious regressions |
| Alpha deployment | 50–75 | Covers main user question categories |
| Production | 100–200 | Statistically reliable metric estimates |

Include at least 10–15% unanswerable entries to test the citation guard.

### Unanswerable entries

These are questions whose answers do not exist in any of your documents. Examples:
- Questions about competitors
- Questions about future features not yet documented
- General knowledge questions unrelated to your domain
- Questions about internal data you chose not to ingest

For unanswerable entries, set `expected_answer` to `null` and `expected_keywords` to `[]`.

```json
{
  "id": "gd_unanswerable_004",
  "question": "What is the stock price of our parent company?",
  "expected_answer": null,
  "expected_keywords": [],
  "source_hint": null,
  "category": "unanswerable",
  "difficulty": "easy",
  "notes": "Financial data not in any ingested document — should abstain"
}
```

### After adding entries

```cmd
# Quick smoke-test on just the new entries
python cli.py eval --limit 5

# Full evaluation to update the report
python cli.py eval

# Commit the updated dataset and report
git add data\golden_dataset.json logs\eval_*.json
git commit -m "eval: add N golden dataset entries for <category>"
```

---

## Writing tests

All tests live in `tests/`. The naming convention is `test_<module>.py`.

**Ground rules:**
- No API keys in any test file.
- No real model loading in any test file.
- No ChromaDB or BM25 index required to run tests.
- Use mocks (`unittest.mock.MagicMock`) for anything that touches models or external services.

### Testing a pure function

```python
# tests/test_my_module.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from my_module import my_function

def test_basic_case():
    result = my_function("input")
    assert result == "expected output"

def test_edge_case():
    result = my_function("")
    assert result == []
```

### Testing with a mock pipeline

```python
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class MockRAGResult:
    question: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    retrieved_chunks: List[Dict] = field(default_factory=list)
    approved_chunks: List[Dict] = field(default_factory=list)
    was_answered: bool = True

def test_something_with_pipeline():
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = MockRAGResult(
        question="test",
        answer="The answer is here [chunk_1].",
        sources=[{"source": "doc.pdf", "chunk_index": 0, "rerank_score": 0.8}],
    )
    # Use mock_pipeline in your test
    result = mock_pipeline.query("test")
    assert "[chunk_1]" in result.answer
```

### Running tests

```cmd
pytest tests/ -v                          # all tests
pytest tests/test_my_module.py -v         # one file
pytest tests/test_my_module.py::MyClass -v  # one class
pytest tests/ --cov=. --cov-report=term-missing  # with coverage
```

---

## Versioning prompts

The LLM prompt template is stored as a plain text file in `prompts/`.
Current version: `prompts/answer_v1.txt`.

### To create a new prompt version

1. Copy the current prompt: `copy prompts\answer_v1.txt prompts\answer_v2.txt`
2. Edit `prompts/answer_v2.txt` with your changes.
3. Update the load path in `generator.py`:

```python
# generator.py, line ~20
_PROMPT_TEMPLATE = (config.PROMPTS_DIR / "answer_v2.txt").read_text(encoding="utf-8")
```

4. Run `python cli.py eval` to get a new evaluation report.
5. Compare the new report against the previous one. If metrics improved, commit both
   the new prompt file and the new report.

### Why keep old versions?

Old prompt files in `prompts/` serve as a record of what was tried. If a future change
causes a regression, you can diff the prompts to understand what changed. If you want
to run an A/B comparison, you can temporarily load both and compare eval reports.

### Prompt template variables

The current template uses two variables:

| Variable | Injected from |
|----------|---------------|
| `{question}` | The user's question string |
| `{chunks}` | The formatted approved chunks from `_format_chunks()` in `generator.py` |

If you add new variables to the template, update `generator.py`'s `prompt.format()` call accordingly.

---

## Swapping a component

Each major component has a defined interface. Swap the implementation without touching
callers as long as you preserve the interface.

### Swap the embedding model

1. Change `EMBEDDING_MODEL` in `config.py`.
2. Delete `data/indexes/` — the old embeddings are incompatible.
3. Re-run `python cli.py ingest data\docs\` to rebuild indexes.

```python
# config.py
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # higher quality
```

### Swap the LLM

The `Generator` class in `generator.py` wraps the Gemini SDK. To use a different LLM:

1. Replace the `__init__` method with your SDK's initialisation.
2. Replace the `generate()` method's API call block.
3. Preserve the return signature: `Tuple[str, List[Dict]]` — `(answer_text, cited_sources)`.

The prompt template in `prompts/answer_v1.txt` is model-agnostic. It may need minor
tuning for models with different instruction-following behaviour.

### Swap BM25 with OpenSearch

Replace these two methods in `retriever.py`:

```python
def _load_bm25_index(self):
    # Replace with: connect to OpenSearch, set self.os_client
    pass

def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
    # Replace with: OpenSearch match query
    # Must return: List[Dict] with keys: id, text, source, chunk_index, score
    pass
```

### Swap ChromaDB with another vector store

Replace `_vector_search` in `retriever.py`:

```python
def _vector_search(self, query: str, top_k: int) -> List[Dict]:
    # Replace with: Pinecone / Weaviate / Qdrant query
    # Must return: List[Dict] with keys: id, text, source, chunk_index, score
    pass
```

---

## Tuning evaluation thresholds

Thresholds live in `evaluation/ci_gate.py` in the `CI_THRESHOLDS` dict:

```python
CI_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    "retrieval_recall":       (">=", 0.60),
    "retrieval_precision":    (">=", 0.50),
    "citation_coverage":      (">=", 0.60),
    "faithfulness_score":     (">=", 0.70),
    "hallucination_rate":     ("<=", 0.15),
    "answer_accuracy":        (">=", 0.60),
    "unanswerable_accuracy":  (">=", 0.80),
}
```

**When to raise a threshold:**
- You have run several evaluations and scores are consistently above the current threshold.
- Raising the threshold locks in the improvement and prevents future regressions.

**When to lower a threshold:**
- You have made a deliberate change (e.g. switching to a more conservative LLM) that
  produces slightly lower scores but is correct behaviour.
- Document the reason in a commit message.

**Threshold update workflow:**
1. Run `python cli.py eval` to get current scores.
2. Update `CI_THRESHOLDS` to match or slightly below the current scores.
3. Run `python cli.py ci-check logs\eval_latest.json` to confirm the gate passes.
4. Commit `ci_gate.py` and the eval report together.

---

## Code style

- **Python 3.10+.** Use `match/case`, `X | Y` union types, `Path` objects, dataclasses.
- **Type hints everywhere.** Function signatures must have parameter and return types.
- **Docstrings on every class and public method.** Use the existing format: one-line summary,
  blank line, then parameter/return descriptions.
- **No print statements.** Use `rich.console.Console` for terminal output.
- **No bare `except`.** Catch specific exceptions. Exception: the logger's write path,
  which intentionally uses `except Exception: pass` to be non-blocking.
- **f-strings only.** No `%` formatting or `.format()` in new code (except prompt templates
  which use `.format()` for variable injection).

---

## Commit conventions

Use conventional commits format:

```
feat: add PDF table extraction support
fix: handle empty BM25 results when corpus is empty
docs: update ARCHITECTURE with OpenSearch swap instructions
test: add test for empty chunk list in reranker
refactor: extract _chunk_id into a standalone function
chore: update eval report after adding 10 golden dataset entries
eval: add unanswerable entries for out-of-domain questions
```

Prefix | Use for
`feat` | New functionality
`fix` | Bug fixes
`docs` | Documentation only (README, ARCHITECTURE, comments)
`test` | Adding or fixing tests
`refactor` | Code changes with no behaviour change
`chore` | Dependency updates, CI changes, report updates
`eval` | Golden dataset changes, threshold changes, eval report commits
