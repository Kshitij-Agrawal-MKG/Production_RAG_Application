"""
ingest.py
Document ingestion pipeline.

Flow:
  File(s) → Parse text → Chunk → Embed → ChromaDB
                                        → BM25 index (saved as pickle)

Usage:
  from ingest import Ingester
  ingester = Ingester()
  ingester.ingest_file("data/docs/manual.pdf")
  ingester.ingest_directory("data/docs/")
  ingester.save_bm25_index()
"""

import hashlib
import pickle
import re
from pathlib import Path
from typing import List, Dict, Tuple

import chromadb
import tiktoken
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rich.console import Console

import config

console = Console()


# ── Text extraction helpers ────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _extract_markdown(path: Path) -> str:
    import markdown
    from bs4 import BeautifulSoup
    md_text = path.read_text(encoding="utf-8", errors="replace")
    html = markdown.markdown(md_text)
    return BeautifulSoup(html, "lxml").get_text(separator="\n", strip=True)


def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text(path: Path) -> str:
    """Route file to the correct extractor based on extension."""
    ext = path.suffix.lower()
    extractors = {
        ".pdf":      _extract_pdf,
        ".docx":     _extract_docx,
        ".html":     _extract_html,
        ".htm":      _extract_html,
        ".md":       _extract_markdown,
        ".markdown": _extract_markdown,
        ".txt":      _extract_txt,
    }
    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}")
    text = extractors[ext](path)
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Chunker ────────────────────────────────────────────────────────────

class Chunker:
    """
    Token-aware recursive character splitter.
    Splits on paragraph → sentence → word boundaries to preserve semantics.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP,
        min_tokens: int = config.MIN_CHUNK_TOKENS,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_tokens = min_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _token_count(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _split(self, text: str) -> List[str]:
        """Split by paragraphs first, then sentences, then words."""
        # Try paragraph splits
        parts = re.split(r"\n\n+", text)
        if len(parts) == 1:
            # Fallback: sentence split
            parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk(self, text: str, source: str) -> List[Dict]:
        """
        Returns a list of chunk dicts:
          {text, source, chunk_index, token_count}
        """
        parts = self._split(text)
        chunks = []
        current_parts: List[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = self._token_count(part)

            # If a single part exceeds chunk size, force-split by words
            if part_tokens > self.chunk_size:
                words = part.split()
                sub_buf: List[str] = []
                sub_tok = 0
                for word in words:
                    wt = self._token_count(word)
                    if sub_tok + wt > self.chunk_size and sub_buf:
                        chunks.append(" ".join(sub_buf))
                        # keep overlap words
                        overlap_words: List[str] = []
                        ot = 0
                        for w in reversed(sub_buf):
                            wt2 = self._token_count(w)
                            if ot + wt2 > self.overlap:
                                break
                            overlap_words.insert(0, w)
                            ot += wt2
                        sub_buf = overlap_words + [word]
                        sub_tok = self._token_count(" ".join(sub_buf))
                    else:
                        sub_buf.append(word)
                        sub_tok += wt
                if sub_buf:
                    chunks.append(" ".join(sub_buf))
                continue

            if current_tokens + part_tokens > self.chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                # build overlap from trailing parts
                overlap_parts: List[str] = []
                ot = 0
                for p in reversed(current_parts):
                    pt = self._token_count(p)
                    if ot + pt > self.overlap:
                        break
                    overlap_parts.insert(0, p)
                    ot += pt
                current_parts = overlap_parts + [part]
                current_tokens = self._token_count("\n\n".join(current_parts))
            else:
                current_parts.append(part)
                current_tokens += part_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        result = []
        for i, chunk_text in enumerate(chunks):
            tc = self._token_count(chunk_text)
            if tc < self.min_tokens:
                continue
            result.append({
                "text": chunk_text,
                "source": source,
                "chunk_index": i,
                "token_count": tc,
            })
        return result


# ── Ingester ────────────────────────────────────────────────────────────

class Ingester:
    """
    Orchestrates: parse → chunk → embed → store in ChromaDB + BM25.
    """

    def __init__(self):
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

        # ChromaDB (persistent local)
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding model
        console.print("[dim]Loading embedding model…[/dim]")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # Chunker
        self.chunker = Chunker()

        # BM25 state (accumulates across calls)
        self._bm25_corpus: List[List[str]] = []   # tokenised chunks
        self._bm25_metadata: List[Dict] = []       # parallel metadata

        # Load existing BM25 index if present
        if config.BM25_INDEX_PATH.exists():
            self._load_bm25_index()

    # ── BM25 persistence ───────────────────────────────────────────────

    def _load_bm25_index(self):
        console.print("[dim]Loading existing BM25 index…[/dim]")
        with open(config.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._bm25_corpus   = data["corpus"]
        self._bm25_metadata = data["metadata"]

    def save_bm25_index(self):
        """Persist the BM25 corpus to disk. Call after ingestion."""
        with open(config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(
                {"corpus": self._bm25_corpus, "metadata": self._bm25_metadata},
                f,
            )
        console.print(
            f"[green]BM25 index saved:[/green] {len(self._bm25_corpus)} chunks "
            f"→ {config.BM25_INDEX_PATH}"
        )

    # ── Core ingestion ─────────────────────────────────────────────────

    def _chunk_id(self, source: str, chunk_index: int, text: str) -> str:
        """Stable, deterministic chunk ID."""
        h = hashlib.md5(f"{source}::{chunk_index}::{text[:80]}".encode()).hexdigest()[:12]
        return f"{Path(source).stem}_{chunk_index}_{h}"

    def ingest_file(self, path: str | Path) -> int:
        """
        Ingest a single file.
        Returns the number of chunks added.
        """
        path = Path(path)
        if not path.exists():
            console.print(f"[red]File not found:[/red] {path}")
            return 0
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            console.print(f"[yellow]Skipping unsupported file:[/yellow] {path.name}")
            return 0

        console.print(f"[cyan]Ingesting:[/cyan] {path.name}")

        # 1. Extract
        try:
            text = extract_text(path)
        except Exception as e:
            console.print(f"[red]Failed to extract {path.name}:[/red] {e}")
            return 0

        if not text.strip():
            console.print(f"[yellow]Empty content:[/yellow] {path.name}")
            return 0

        # 2. Chunk
        chunks = self.chunker.chunk(text, source=str(path))
        if not chunks:
            console.print(f"[yellow]No chunks produced for:[/yellow] {path.name}")
            return 0

        # 3. Embed (batch)
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts, show_progress_bar=False, batch_size=32, normalize_embeddings=True
        ).tolist()

        # 4. Add to ChromaDB (skip existing IDs)
        ids        = [self._chunk_id(c["source"], c["chunk_index"], c["text"]) for c in chunks]
        metadatas  = [{"source": c["source"], "chunk_index": c["chunk_index"], "token_count": c["token_count"]} for c in chunks]

        existing = set(self.collection.get(ids=ids)["ids"])
        new_ids, new_texts, new_embeddings, new_metas = [], [], [], []
        for i, cid in enumerate(ids):
            if cid not in existing:
                new_ids.append(cid)
                new_texts.append(texts[i])
                new_embeddings.append(embeddings[i])
                new_metas.append(metadatas[i])

        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_texts,
                embeddings=new_embeddings,
                metadatas=new_metas,
            )

        # 5. Add to BM25 corpus
        for c in chunks:
            tokens = c["text"].lower().split()
            self._bm25_corpus.append(tokens)
            self._bm25_metadata.append({"source": c["source"], "chunk_index": c["chunk_index"], "text": c["text"]})

        added = len(new_ids)
        console.print(
            f"  [green]✓[/green] {added} new chunks added "
            f"({len(chunks) - added} already indexed)"
        )
        return added

    def ingest_directory(self, directory: str | Path) -> int:
        """Recursively ingest all supported files in a directory."""
        directory = Path(directory)
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
        ]
        if not files:
            console.print(f"[yellow]No supported files found in:[/yellow] {directory}")
            return 0

        total = 0
        for f in tqdm(files, desc="Ingesting files"):
            total += self.ingest_file(f)

        self.save_bm25_index()
        console.print(f"\n[bold green]Done.[/bold green] Total new chunks indexed: {total}")
        return total

    def collection_stats(self) -> Dict:
        """Return basic stats about what's indexed."""
        count = self.collection.count()
        return {
            "chroma_chunks": count,
            "bm25_chunks": len(self._bm25_corpus),
        }
