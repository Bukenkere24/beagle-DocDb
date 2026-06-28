"""Document chunking strategies for DocDB/ETL Helper."""

from __future__ import annotations

import re
from typing import Literal

Strategy = Literal["recursive", "fixed", "semantic"]

_SPLIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\n+"),
    re.compile(r"\n"),
    re.compile(r"(?<=[.!?])\s+"),
    re.compile(r"\s+"),
)


def estimate_tokens(text: str) -> int:
    """Return approximate token count for *text*."""
    if not text:
        return 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _encode_tokens(text: str) -> list[int]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)
    except Exception:
        return list(text.encode("utf-8"))


def _decode_tokens(tokens: list[int]) -> str:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return enc.decode(tokens)
    except Exception:
        return bytes(tokens).decode("utf-8", errors="replace")


def chunk_fixed(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split *text* into fixed-size token windows with overlap."""
    if not text.strip():
        return []

    tokens = _encode_tokens(text)
    if estimate_tokens(text) <= max_tokens:
        return [text.strip()]

    overlap = min(overlap, max_tokens - 1)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        piece = _decode_tokens(tokens[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def _merge_small_parts(parts: list[str], max_tokens: int) -> list[str]:
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}\n\n{part}".strip() if buffer else part
        if estimate_tokens(candidate) <= max_tokens:
            buffer = candidate
        else:
            if buffer:
                merged.append(buffer)
            buffer = part
    if buffer:
        merged.append(buffer)
    return merged


def _split_recursive(text: str, max_tokens: int, depth: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    if depth >= len(_SPLIT_PATTERNS):
        return chunk_fixed(text, max_tokens, overlap=0)

    pattern = _SPLIT_PATTERNS[depth]
    parts = [p.strip() for p in pattern.split(text) if p.strip()]
    if len(parts) <= 1:
        return _split_recursive(text, max_tokens, depth + 1)

    merged = _merge_small_parts(parts, max_tokens)
    result: list[str] = []
    for part in merged:
        if estimate_tokens(part) <= max_tokens:
            result.append(part)
        else:
            result.extend(_split_recursive(part, max_tokens, depth + 1))
    return result


def chunk_recursive(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split *text* on natural boundaries, subdividing oversized sections."""
    base = _split_recursive(text, max_tokens)
    if overlap <= 0 or len(base) <= 1:
        return base

    overlapped: list[str] = [base[0]]
    for idx in range(1, len(base)):
        prev = base[idx - 1]
        curr = base[idx]
        prev_tokens = _encode_tokens(prev)
        tail = _decode_tokens(prev_tokens[-overlap:]).strip() if prev_tokens else ""
        combined = f"{tail}\n\n{curr}".strip() if tail else curr
        overlapped.append(combined if estimate_tokens(combined) <= max_tokens else curr)
    return overlapped


def chunk_semantic(text: str, max_tokens: int, overlap: int) -> list[str]:
    # TODO: semantic chunking — stretch goal (embedding-similarity boundaries).
    raise NotImplementedError("Semantic chunking is not yet implemented")


def chunk_text(
    text: str,
    strategy: Strategy,
    max_tokens: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Dispatch to the requested chunking strategy."""
    strategy = strategy.lower()  # type: ignore[assignment]
    if strategy == "fixed":
        return chunk_fixed(text, max_tokens, overlap)
    if strategy == "recursive":
        return chunk_recursive(text, max_tokens, overlap)
    if strategy == "semantic":
        return chunk_semantic(text, max_tokens, overlap)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")
