from __future__ import annotations

from typing import List

import pytest

import app.embedding_pipeline as embedding_pipeline
from app.embeddings import EmbeddingClientError, EmbeddingResult


def test_infer_batch_size_limit_from_triton_error() -> None:
    message = (
        "Triton infer failed: [400] inference request batch-size "
        "must be <= 8 for 'qwen3_embed_4b_onnx'"
    )
    assert embedding_pipeline.infer_batch_size_limit(message) == 8


def test_embed_texts_adaptive_reduces_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: List[int] = []

    def _mock_embed_texts(texts):
        calls.append(len(texts))
        if len(texts) > 2:
            raise EmbeddingClientError("inference request batch-size must be <= 2")
        return EmbeddingResult(
            vectors=[[0.1] * embedding_pipeline.settings.embeddings_dim for _ in texts],
            model="mock-embed",
        )

    monkeypatch.setattr(embedding_pipeline, "embed_texts", _mock_embed_texts)

    result = embedding_pipeline._embed_texts_adaptive(
        ["a", "b", "c", "d", "e"],
        batch_size=5,
    )
    assert calls == [5, 2, 2, 1]
    assert len(result.vectors) == 5
    assert result.model == "mock-embed"


def test_embed_texts_adaptive_raises_when_single_row_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _mock_embed_texts(texts):
        raise EmbeddingClientError("upstream unavailable")

    monkeypatch.setattr(embedding_pipeline, "embed_texts", _mock_embed_texts)

    with pytest.raises(EmbeddingClientError):
        embedding_pipeline._embed_texts_adaptive(["only-one"], batch_size=4)


def test_run_embedding_backfill_requires_embeddings_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding_pipeline, "embeddings_enabled", lambda: False)
    with pytest.raises(RuntimeError, match="EMBEDDINGS_BASE_URL"):
        embedding_pipeline.run_embedding_backfill(batch_size=8)


def test_run_embedding_backfill_rejects_non_positive_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(embedding_pipeline, "embeddings_enabled", lambda: True)
    with pytest.raises(RuntimeError, match="EMBEDDINGS_BATCH_SIZE must be > 0"):
        embedding_pipeline.run_embedding_backfill(batch_size=0)
