"""
generator.py
Answer generation with Gemini (free-tier) + citation enforcement.
Extended with token counting for cost tracking and Langfuse tracing.

Flow:
  approved_chunks → format context → fill prompt template
                  → Gemini API → answer with [chunk_N] citations
                  → (input_tokens, output_tokens) returned alongside answer

Usage:
  from generator import Generator
  gen = Generator()
  answer, sources, prompt, in_tok, out_tok = gen.generate(query, approved_chunks)
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple

import google.generativeai as genai
from rich.console import Console

import config
from monitoring.token_counter import count_tokens, extract_token_usage, estimate_tokens

console = Console()

# Load versioned prompt template once
_PROMPT_VERSION  = "answer_v1"
_PROMPT_TEMPLATE = (config.PROMPTS_DIR / "answer_v1.txt").read_text(encoding="utf-8")


def _format_chunks(chunks: List[Dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source_name = Path(chunk["source"]).name
        lines.append(
            f"[chunk_{i}] source: {source_name} (chunk {chunk['chunk_index']})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(lines)


def _extract_cited_sources(answer: str, chunks: List[Dict]) -> List[Dict]:
    cited_indices = set(int(n) for n in re.findall(r"\[chunk_(\d+)\]", answer))
    cited_sources = []
    seen = set()
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(chunks):
            chunk = chunks[idx - 1]
            key = (chunk["source"], chunk["chunk_index"])
            if key not in seen:
                seen.add(key)
                cited_sources.append({
                    "source":       chunk["source"],
                    "chunk_index":  chunk["chunk_index"],
                    "rerank_score": chunk.get("rerank_score", 0.0),
                })
    return cited_sources


class Generator:
    """Wraps Gemini API for grounded answer generation with token tracking."""

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it to your .env file:\n"
                "  GEMINI_API_KEY=your_key_here\n"
                "Get a free key at: https://aistudio.google.com/app/apikey"
            )
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=config.GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=config.GEMINI_TEMPERATURE,
                max_output_tokens=config.GEMINI_MAX_TOKENS,
            ),
        )
        self.prompt_version = _PROMPT_VERSION
        self.model_name     = config.GEMINI_MODEL

    def generate(
        self,
        query: str,
        approved_chunks: List[Dict],
    ) -> Tuple[str, List[Dict], str, int, int]:
        """
        Generate a grounded answer.

        Returns:
          answer        (str)      — LLM response with [chunk_N] citations
          sources       (List)     — cited source dicts
          prompt        (str)      — full prompt sent to the LLM
          input_tokens  (int)      — prompt token count
          output_tokens (int)      — completion token count
        """
        # Citation guard: no chunks → no answer
        if not approved_chunks:
            no_answer = (
                "I could not find a reliable answer in the provided documents. "
                "The retrieved chunks did not meet the relevance threshold."
            )
            return no_answer, [], "", 0, 0

        # Build prompt
        chunks_text = _format_chunks(approved_chunks)
        prompt = _PROMPT_TEMPLATE.format(question=query, chunks=chunks_text)
        input_tokens_est = count_tokens(prompt)

        # Call Gemini
        try:
            response = self.model.generate_content(prompt)
            answer   = response.text.strip()

            # Try to get exact token counts from SDK metadata
            in_tok, out_tok = extract_token_usage(response)
            if in_tok == 0:
                in_tok, out_tok = estimate_tokens(prompt, answer)

        except Exception as e:
            console.print(f"[red]Gemini API error:[/red] {e}")
            return f"API error: {e}", [], prompt, input_tokens_est, 0

        sources = _extract_cited_sources(answer, approved_chunks)

        if "[chunk_" not in answer.lower():
            console.print(
                "[yellow]Warning:[/yellow] Model answered without any citations."
            )

        return answer, sources, prompt, in_tok, out_tok
