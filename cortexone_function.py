"""CortexOne entry point for the DocDB/ETL Helper function."""

from __future__ import annotations

import json
import uuid
from typing import Any

from chunkers import chunk_text, estimate_tokens
from extract import (
    extract_metadata,
    prepare_embeddings_ready,
    summarize_chunk,
    summarize_document,
)

# Guardrails for 512 MB / 300 s ephemeral runs — never load unbounded input.
MAX_TOTAL_CHARS = 500_000
MAX_DOCS = 50
DEFAULT_CHUNKING = {"strategy": "recursive", "max_tokens": 800, "overlap": 100}
DEFAULT_EXTRACT = {"summary": True, "metadata": True, "embeddings_ready": True}
VALID_STRATEGIES = {"recursive", "fixed", "semantic"}


def _response(status_code: int, body: Any) -> dict[str, Any]:
    return {"statusCode": status_code, "body": body}


def _error(status_code: int, message: str) -> dict[str, Any]:
    return _response(status_code, {"error": message})


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def _as_int(value: Any, default: int, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _normalize_document(raw: Any, index: int) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    doc_id = str(raw.get("id") or f"doc-{index + 1}").strip() or f"doc-{index + 1}"
    text = raw.get("text")
    if text is None:
        text = ""
    if not isinstance(text, str):
        return None
    mime_type = str(raw.get("mime_type") or "text/plain").strip() or "text/plain"
    source_uri = str(raw.get("source_uri") or "").strip()
    return {
        "id": doc_id,
        "text": text,
        "mime_type": mime_type,
        "source_uri": source_uri,
    }


def _parse_event(event: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if event is None:
        return None, _error(400, "Missing event payload")
    if not isinstance(event, dict):
        return None, _error(400, "Event must be a JSON object")

    documents_raw = event.get("documents")
    if documents_raw is None:
        documents_raw = []
    if not isinstance(documents_raw, list):
        return None, _error(400, "documents must be an array")
    if not documents_raw:
        return None, _error(400, "documents must contain at least one record")

    documents: list[dict[str, str]] = []
    for index, raw_doc in enumerate(documents_raw[:MAX_DOCS]):
        normalized = _normalize_document(raw_doc, index)
        if normalized is None:
            return None, _error(400, f"Invalid document at index {index}")
        documents.append(normalized)

    chunking_raw = event.get("chunking") or {}
    if not isinstance(chunking_raw, dict):
        return None, _error(400, "chunking must be an object")

    strategy = str(chunking_raw.get("strategy") or DEFAULT_CHUNKING["strategy"]).strip().lower()
    if strategy not in VALID_STRATEGIES:
        return None, _error(400, f"Unsupported chunking strategy: {strategy}")

    max_tokens = _as_int(chunking_raw.get("max_tokens"), DEFAULT_CHUNKING["max_tokens"], minimum=64)
    overlap = _as_int(chunking_raw.get("overlap"), DEFAULT_CHUNKING["overlap"], minimum=0)
    if overlap >= max_tokens:
        overlap = max(0, max_tokens // 5)

    extract_raw = event.get("extract") or {}
    if not isinstance(extract_raw, dict):
        return None, _error(400, "extract must be an object")

    parsed = {
        "documents": documents,
        "chunking": {
            "strategy": strategy,
            "max_tokens": max_tokens,
            "overlap": overlap,
        },
        "extract": {
            "summary": _as_bool(extract_raw.get("summary"), DEFAULT_EXTRACT["summary"]),
            "metadata": _as_bool(extract_raw.get("metadata"), DEFAULT_EXTRACT["metadata"]),
            "embeddings_ready": _as_bool(
                extract_raw.get("embeddings_ready"), DEFAULT_EXTRACT["embeddings_ready"]
            ),
        },
        "schema_hint": str(event.get("schema_hint") or "").strip(),
        "truncated": len(documents_raw) > MAX_DOCS,
    }
    return parsed, None


def _apply_char_budget(documents: list[dict[str, str]]) -> tuple[list[dict[str, str]], bool]:
    kept: list[dict[str, str]] = []
    used = 0
    truncated = False
    for doc in documents:
        text = doc["text"]
        remaining = MAX_TOTAL_CHARS - used
        if remaining <= 0:
            truncated = True
            break
        if len(text) > remaining:
            kept.append({**doc, "text": text[:remaining]})
            truncated = True
            break
        kept.append(doc)
        used += len(text)
    return kept, truncated


def _process_documents(parsed: dict[str, Any]) -> dict[str, Any]:
    documents, char_truncated = _apply_char_budget(parsed["documents"])
    chunk_cfg = parsed["chunking"]
    extract_cfg = parsed["extract"]

    chunks_out: list[dict[str, Any]] = []
    doc_summaries: list[dict[str, Any]] = []

    for doc in documents:
        text = doc["text"]
        if not text.strip():
            continue

        doc_summary_text = ""
        doc_key_points: list[str] = []
        if extract_cfg["summary"]:
            doc_summary_text, doc_key_points = summarize_document(text)
            doc_summaries.append(
                {
                    "doc_id": doc["id"],
                    "summary": doc_summary_text,
                    "key_points": doc_key_points,
                }
            )

        try:
            raw_chunks = chunk_text(
                text,
                strategy=chunk_cfg["strategy"],
                max_tokens=chunk_cfg["max_tokens"],
                overlap=chunk_cfg["overlap"],
            )
        except NotImplementedError as exc:
            raise exc

        for index, chunk_body in enumerate(raw_chunks):
            body = chunk_body
            if extract_cfg["embeddings_ready"]:
                body = prepare_embeddings_ready(body)

            metadata = (
                extract_metadata(
                    chunk_body,
                    mime_type=doc["mime_type"],
                    source_uri=doc["source_uri"],
                )
                if extract_cfg["metadata"]
                else {"title": doc["id"], "section": "body", "tags": []}
            )

            chunk_summary = ""
            if extract_cfg["summary"]:
                chunk_summary = summarize_chunk(chunk_body, doc_summary_text)

            chunks_out.append(
                {
                    "doc_id": doc["id"],
                    "chunk_id": f"{doc['id']}-chunk-{index + 1}-{uuid.uuid4().hex[:8]}",
                    "text": body,
                    "token_count": estimate_tokens(body),
                    "metadata": metadata,
                    "summary": chunk_summary,
                }
            )

    body: dict[str, Any] = {
        "chunks": chunks_out,
        "doc_summaries": doc_summaries,
        "stats": {
            "docs_in": len(parsed["documents"]),
            "chunks_out": len(chunks_out),
            "truncated": parsed["truncated"] or char_truncated,
        },
    }
    if parsed["schema_hint"]:
        body["schema_hint"] = parsed["schema_hint"]
    return body


def cortexone_handler(event: dict[str, Any] | None, context: Any = None) -> dict[str, Any]:
    """Rival/CortexOne function entry point."""
    _ = context
    try:
        parsed, error = _parse_event(event)
        if error:
            return error

        assert parsed is not None
        body = _process_documents(parsed)
        return _response(200, body)
    except NotImplementedError as exc:
        return _error(400, str(exc))
    except Exception:
        return _error(500, "Internal server error")


def _run_test_case(name: str, event: dict[str, Any]) -> None:
    result = cortexone_handler(event)
    status = result["statusCode"]
    body = result["body"]
    print(f"\n=== {name} ===")
    print(f"statusCode: {status}")
    if isinstance(body, dict):
        print(json.dumps(body, indent=2)[:2000])
    else:
        print(body)


if __name__ == "__main__":
    sample = (
        "DocDB ETL Helper\n\n"
        "This tool converts raw records into chunked payloads. "
        "It respects runtime limits and never assumes persistence."
    )

    _run_test_case(
        "1) single short doc",
        {
            "documents": [
                {
                    "id": "doc-1",
                    "text": sample,
                    "mime_type": "text/plain",
                    "source_uri": "file://sample.txt",
                }
            ],
            "chunking": {"strategy": "recursive", "max_tokens": 120, "overlap": 20},
            "extract": {"summary": True, "metadata": True, "embeddings_ready": True},
        },
    )

    _run_test_case(
        "2) multi-doc",
        {
            "documents": [
                {
                    "id": "alpha",
                    "text": "Alpha document about ingestion pipelines.",
                    "mime_type": "text/plain",
                    "source_uri": "s3://bucket/alpha.txt",
                },
                {
                    "id": "beta",
                    "text": "Beta document about chunking strategies and overlap windows.",
                    "mime_type": "text/markdown",
                    "source_uri": "s3://bucket/beta.md",
                },
            ],
            "chunking": {"strategy": "fixed", "max_tokens": 64, "overlap": 8},
        },
    )

    oversized = "A" * (MAX_TOTAL_CHARS + 10_000)
    _run_test_case(
        "3) oversized input -> truncation",
        {
            "documents": [{"id": "big", "text": oversized, "mime_type": "text/plain", "source_uri": ""}],
            "chunking": {"strategy": "fixed", "max_tokens": 256, "overlap": 32},
            "extract": {"summary": False, "metadata": False, "embeddings_ready": True},
        },
    )

    _run_test_case(
        "4) malformed record",
        {"documents": [{"id": "bad", "text": 123}]},
    )

    _run_test_case(
        "5) mixed mime types",
        {
            "documents": [
                {
                    "id": "html-doc",
                    "text": "<html><body><h1>Title</h1><p>HTML content block.</p></body></html>",
                    "mime_type": "text/html",
                    "source_uri": "https://example.com/page",
                },
                {
                    "id": "json-doc",
                    "text": '{"title": "Record", "body": "JSON-backed content."}',
                    "mime_type": "application/json",
                    "source_uri": "file://record.json",
                },
            ],
            "chunking": {"strategy": "recursive", "max_tokens": 100, "overlap": 10},
        },
    )
