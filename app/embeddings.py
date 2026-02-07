from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import httpx

from .config import settings


class EmbeddingClientError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: List[List[float]]
    model: str


def embeddings_enabled() -> bool:
    return bool(settings.embeddings_base_url.strip())


def _normalize_base_url(raw: str) -> str:
    return raw.rstrip("/")


def _validate_texts(texts: Sequence[str]) -> List[str]:
    cleaned = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not cleaned:
        raise EmbeddingClientError("embedding request requires at least one non-empty text")
    return cleaned


def _validate_vectors(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    expected_dim = settings.embeddings_dim
    normalized: List[List[float]] = []
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dim:
            raise EmbeddingClientError(
                f"embedding {index} has dim {len(vector)}; expected {expected_dim}"
            )
        normalized.append([float(value) for value in vector])
    return normalized


def embed_texts(texts: Sequence[str]) -> EmbeddingResult:
    if not embeddings_enabled():
        raise EmbeddingClientError("EMBEDDINGS_BASE_URL is not configured")

    cleaned = _validate_texts(texts)
    payload = {"texts": cleaned, "model": settings.embeddings_model_id}
    url = f"{_normalize_base_url(settings.embeddings_base_url)}/embed"
    timeout = httpx.Timeout(settings.embeddings_timeout_s)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        raise EmbeddingClientError(f"embedding HTTP request failed: {exc}") from exc

    if response.status_code != 200:
        detail = response.text.strip()
        if len(detail) > 400:
            detail = detail[:400]
        raise EmbeddingClientError(
            f"embedding service returned {response.status_code}: {detail}"
        )

    body = response.json()
    raw_vectors = body.get("embeddings")
    if not isinstance(raw_vectors, list):
        raise EmbeddingClientError("embedding response missing 'embeddings' list")
    if len(raw_vectors) != len(cleaned):
        raise EmbeddingClientError(
            f"embedding response count mismatch: got {len(raw_vectors)}, expected {len(cleaned)}"
        )

    vectors = _validate_vectors(raw_vectors)
    model = str(body.get("model") or settings.embeddings_model_id)
    return EmbeddingResult(vectors=vectors, model=model)


def embed_texts_batched(
    texts: Sequence[str], batch_size: int | None = None
) -> EmbeddingResult:
    cleaned = _validate_texts(texts)
    size = batch_size or settings.embeddings_batch_size
    if size <= 0:
        raise EmbeddingClientError("batch size must be > 0")

    all_vectors: List[List[float]] = []
    model_used = settings.embeddings_model_id
    for start in range(0, len(cleaned), size):
        chunk = cleaned[start : start + size]
        result = embed_texts(chunk)
        all_vectors.extend(result.vectors)
        model_used = result.model
    return EmbeddingResult(vectors=all_vectors, model=model_used)
