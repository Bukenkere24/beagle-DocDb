"""Metadata extraction, summarization, and embeddings-ready normalization."""

from __future__ import annotations

import json
import os
import re
from typing import Any

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")
_BLOCK_TAGS = re.compile(r"</?(?:br|p|div|h[1-6]|li|tr|td|th|blockquote)[^>]*>", re.IGNORECASE)


def _normalize_mime(mime_type: str) -> str:
    return (mime_type or "text/plain").lower().split(";")[0].strip()


def _html_title(text: str) -> str | None:
    for pattern in (
        r"<h1[^>]*>(.*?)</h1>",
        r"<title[^>]*>(.*?)</title>",
    ):
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            title = _HTML_TAG.sub("", match.group(1)).strip()
            if title:
                return title[:120]
    return None


def _html_to_plain(text: str) -> str:
    with_breaks = _BLOCK_TAGS.sub("\n", text)
    plain = _HTML_TAG.sub(" ", with_breaks)
    return _WHITESPACE.sub(" ", plain).strip()


def _json_to_plain(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(data, dict):
        return text
    parts: list[str] = []
    title = data.get("title")
    if title:
        parts.append(str(title).strip())
    for key in ("body", "text", "content", "description", "summary"):
        value = data.get(key)
        if value:
            parts.append(str(value).strip())
            break
    return "\n\n".join(part for part in parts if part) or text


def _title_from_structured(text: str, mime_type: str) -> str | None:
    mime = _normalize_mime(mime_type)
    if mime in {"text/html", "application/xhtml+xml"}:
        return _html_title(text)
    if mime in {"application/json", "text/json"}:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("title"):
                return str(data["title"]).strip()[:120]
        except json.JSONDecodeError:
            return None
    return None


def normalize_text_by_mime(text: str, mime_type: str = "text/plain") -> str:
    """Convert HTML/JSON (and other structured mime types) to plain text."""
    mime = _normalize_mime(mime_type)
    if mime in {"text/html", "application/xhtml+xml"}:
        return _html_to_plain(text)
    if mime in {"application/json", "text/json"}:
        return _json_to_plain(text)
    return text


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
    plain = normalize_text_by_mime(text, mime_type)
    structured_title = _title_from_structured(text, mime_type)
    prompt = (
        "Extract document metadata as JSON with keys title (string), "
        "section (string), tags (array of strings).\n"
        f"mime_type: {mime_type}\n"
        f"source_uri: {source_uri}\n"
        f"text:\n{plain[:4000]}"
    )
    payload = _llm_json(prompt)
    if payload:
        tags = payload.get("tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        return {
            "title": str(
                payload.get("title") or structured_title or _first_nonempty_line(plain)
            ),
            "section": str(payload.get("section") or "body"),
            "tags": [str(tag) for tag in tags[:8]],
        }

    return {
        "title": structured_title or _first_nonempty_line(plain),
        "section": "body",
        "tags": _heuristic_tags(plain),
    }


def summarize_document(text: str, *, mime_type: str = "text/plain") -> tuple[str, list[str]]:
    """Return document summary and key points."""
    plain = normalize_text_by_mime(text, mime_type)
    prompt = (
        "Summarize the document as JSON with keys summary (string) and "
        "key_points (array of strings, 3-5 items).\n"
        f"mime_type: {mime_type}\n"
        f"text:\n{plain[:6000]}"
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

    return _heuristic_summary(plain)


def summarize_chunk(
    text: str,
    doc_summary: str = "",
    *,
    mime_type: str = "text/plain",
) -> str:
    """Return a short chunk summary."""
    plain = normalize_text_by_mime(text, mime_type)
    if len(plain) <= 240 and not doc_summary:
        return plain.strip()

    prompt = (
        "Return JSON with key summary containing one concise sentence for this chunk.\n"
        f"document_context: {doc_summary[:500]}\n"
        f"mime_type: {mime_type}\n"
        f"chunk:\n{plain[:2000]}"
    )
    payload = _llm_json(prompt)
    if payload and payload.get("summary"):
        return str(payload["summary"]).strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", plain) if s.strip()]
    return sentences[0] if sentences else plain.strip()[:240]


def prepare_embeddings_ready(text: str, mime_type: str = "text/plain") -> str:
    """Normalize chunk text for downstream embedding pipelines."""
    plain = normalize_text_by_mime(text, mime_type)
    cleaned = _CONTROL_CHARS.sub("", plain)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned
