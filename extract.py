"""Metadata extraction, summarization, and embeddings-ready normalization."""

from __future__ import annotations

import json
import os
import re
from typing import Any

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from openai import OpenAI

        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:120]
    return "Untitled"


def _heuristic_tags(text: str, limit: int = 5) -> list[str]:
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    freq: dict[str, int] = {}
    stop = {"that", "this", "with", "from", "have", "will", "been", "they", "their", "about"}
    for word in words:
        if word in stop:
            continue
        freq[word] = freq.get(word, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:limit]]


def _heuristic_summary(text: str, max_sentences: int = 3) -> tuple[str, list[str]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        snippet = text.strip()[:240]
        return snippet, [snippet] if snippet else []
    summary = " ".join(sentences[:max_sentences])
    key_points = sentences[: min(5, len(sentences))]
    return summary, key_points


def _llm_json(prompt: str) -> dict[str, Any] | None:
    client = _openai_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. Do not wrap in markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=700,
        )
        raw = (response.choices[0].message.content or "").strip()
        return json.loads(raw)
    except Exception:
        return None


def extract_metadata(
    text: str,
    *,
    mime_type: str = "text/plain",
    source_uri: str = "",
) -> dict[str, Any]:
    """Extract chunk-level metadata, using BYOK LLM when available."""
    prompt = (
        "Extract document metadata as JSON with keys title (string), "
        "section (string), tags (array of strings).\n"
        f"mime_type: {mime_type}\n"
        f"source_uri: {source_uri}\n"
        f"text:\n{text[:4000]}"
    )
    payload = _llm_json(prompt)
    if payload:
        tags = payload.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        return {
            "title": str(payload.get("title") or _first_nonempty_line(text)),
            "section": str(payload.get("section") or "body"),
            "tags": [str(tag) for tag in tags[:8]],
        }

    return {
        "title": _first_nonempty_line(text),
        "section": "body",
        "tags": _heuristic_tags(text),
    }


def summarize_document(text: str) -> tuple[str, list[str]]:
    """Return document summary and key points."""
    prompt = (
        "Summarize the document as JSON with keys summary (string) and "
        "key_points (array of strings, 3-5 items).\n"
        f"text:\n{text[:6000]}"
    )
    payload = _llm_json(prompt)
    if payload:
        summary = str(payload.get("summary") or "").strip()
        key_points_raw = payload.get("key_points") or []
        if not isinstance(key_points_raw, list):
            key_points_raw = [str(key_points_raw)]
        key_points = [str(point).strip() for point in key_points_raw if str(point).strip()]
        if summary:
            return summary, key_points or [summary]

    return _heuristic_summary(text)


def summarize_chunk(text: str, doc_summary: str = "") -> str:
    """Return a short chunk summary."""
    if len(text) <= 240 and not doc_summary:
        return text.strip()

    prompt = (
        "Return JSON with key summary containing one concise sentence for this chunk.\n"
        f"document_context: {doc_summary[:500]}\n"
        f"chunk:\n{text[:2000]}"
    )
    payload = _llm_json(prompt)
    if payload and payload.get("summary"):
        return str(payload["summary"]).strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences[0] if sentences else text.strip()[:240]


def prepare_embeddings_ready(text: str) -> str:
    """Normalize chunk text for downstream embedding pipelines."""
    cleaned = _CONTROL_CHARS.sub("", text)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned
