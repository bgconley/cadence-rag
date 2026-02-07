from __future__ import annotations

from typing import Any

import pytest

from app.config import settings
from app.embeddings import EmbeddingClientError, embed_texts, embed_texts_batched


class _FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self) -> dict[str, Any]:
        return self._body


class _FakeClient:
    def __init__(self, response: _FakeResponse, recorder: list[dict[str, Any]]) -> None:
        self._response = response
        self._recorder = recorder

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
        self._recorder.append({"url": url, "payload": json})
        return self._response


def test_embed_texts_requires_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "embeddings_base_url", "")
    with pytest.raises(EmbeddingClientError):
        embed_texts(["hello"])


def test_embed_texts_validates_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "embeddings_base_url", "http://embed.local")
    monkeypatch.setattr(settings, "embeddings_model_id", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(settings, "embeddings_dim", 4)

    def fake_client(*args, **kwargs):
        return _FakeClient(
            _FakeResponse(
                200,
                {
                    "embeddings": [[0.1, 0.2, 0.3, 0.4]],
                    "model": "Qwen/Qwen3-Embedding-4B",
                },
            ),
            calls,
        )

    monkeypatch.setattr("app.embeddings.httpx.Client", fake_client)
    result = embed_texts(["hello"])
    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == 4
    assert calls[0]["url"] == "http://embed.local/embed"
    assert calls[0]["payload"]["texts"] == ["hello"]


def test_embed_texts_batched_splits_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(settings, "embeddings_base_url", "http://embed.local")
    monkeypatch.setattr(settings, "embeddings_model_id", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(settings, "embeddings_dim", 3)

    class _BatchClient(_FakeClient):
        def post(self, url: str, json: dict[str, Any]) -> _FakeResponse:
            super().post(url, json)
            vectors = [[0.1, 0.2, 0.3] for _ in json["texts"]]
            return _FakeResponse(200, {"embeddings": vectors, "model": json["model"]})

    def fake_client(*args, **kwargs):
        return _BatchClient(_FakeResponse(200, {}), calls)

    monkeypatch.setattr("app.embeddings.httpx.Client", fake_client)
    result = embed_texts_batched(["a", "b", "c", "d", "e"], batch_size=2)
    assert len(calls) == 3
    assert [len(call["payload"]["texts"]) for call in calls] == [2, 2, 1]
    assert len(result.vectors) == 5
