"""
monitoring/token_counter.py
Token counting for cost tracking and Langfuse usage metrics.

Uses tiktoken cl100k_base for accurate counting. Loads lazily on first call
so import never fails in restricted network environments (CI, air-gapped).
Falls back to a word-count heuristic if tiktoken data cannot be fetched.
"""

from typing import Tuple

_enc = None


def _get_enc():
    """Lazy-load tiktoken encoding. Returns None if unavailable."""
    global _enc
    if _enc is not None:
        return _enc
    try:
        import tiktoken
        _enc = tiktoken.get_encoding("cl100k_base")
        return _enc
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken cl100k_base.
    Falls back to word_count * 1.3 heuristic if tiktoken is unavailable.
    """
    if not text:
        return 0
    enc = _get_enc()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Heuristic fallback
    return max(1, int(len(text.split()) * 1.3))


def extract_token_usage(gemini_response) -> Tuple[int, int]:
    """
    Extract input/output token counts from a Gemini SDK response object.
    Returns (0, 0) if usage_metadata is unavailable.
    """
    try:
        usage = getattr(gemini_response, "usage_metadata", None)
        if usage is not None:
            inp = getattr(usage, "prompt_token_count",     0) or 0
            out = getattr(usage, "candidates_token_count", 0) or 0
            return int(inp), int(out)
    except Exception:
        pass
    return 0, 0


def estimate_tokens(prompt: str, answer: str) -> Tuple[int, int]:
    """Estimate token counts from strings when SDK metadata is unavailable."""
    return count_tokens(prompt), count_tokens(answer)
