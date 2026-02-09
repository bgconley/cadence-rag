from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

import app.ingest_fs as ingest_fs
from app.embeddings import EmbeddingClientError
from app.ingest_fs import (
    _auto_embed_call_if_configured,
    _build_auto_manifest,
    _build_retry_policy,
    validate_bundle_directory,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_bundle_directory_transcript_and_analysis(tmp_path: Path) -> None:
    bundle = tmp_path / "demo-call-001"
    bundle.mkdir(parents=True)

    _write(
        bundle / "transcript.json",
        json.dumps(
            [
                {
                    "speaker": "Alice",
                    "start_ts_ms": 0,
                    "end_ts_ms": 1000,
                    "text": "Discuss BOM and tiering options.",
                }
            ]
        ),
    )
    _write(bundle / "analysis" / "summary.md", "Summary content")
    _write(bundle / "_READY", "")
    _write(
        bundle / "manifest.json",
        json.dumps(
            {
                "bundle_id": "demo-call-001",
                "call_ref": {"external_source": "test", "external_id": "demo-call-001"},
                "transcript": {"path": "transcript.json", "format": "json_turns"},
                "analysis": [{"kind": "summary", "path": "analysis/summary.md"}],
            }
        ),
    )

    validated = validate_bundle_directory(bundle)
    assert validated.bundle_id == "demo-call-001"
    assert len(validated.files) == 3
    assert {item.kind for item in validated.files} == {
        "manifest",
        "transcript",
        "analysis:summary",
    }


def test_validate_bundle_directory_rejects_path_escape(tmp_path: Path) -> None:
    bundle = tmp_path / "demo-call-002"
    bundle.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("[]", encoding="utf-8")
    _write(bundle / "_READY", "")
    _write(
        bundle / "manifest.json",
        json.dumps(
            {
                "bundle_id": "demo-call-002",
                "call_ref": {"external_source": "test", "external_id": "demo-call-002"},
                "transcript": {"path": "../outside.json", "format": "json_turns"},
            }
        ),
    )

    with pytest.raises(ValueError, match="path escapes bundle root"):
        validate_bundle_directory(bundle)


def test_build_retry_policy_backoff_intervals() -> None:
    assert _build_retry_policy(max_attempts=1, base_backoff_s=10) is None

    retry = _build_retry_policy(max_attempts=4, base_backoff_s=5)
    assert retry is not None
    assert retry.max == 3
    assert list(retry.intervals) == [5, 10, 20]


def test_build_auto_manifest_infers_transcript_and_analysis(tmp_path: Path) -> None:
    bundle = tmp_path / "starcluster-call-001"
    bundle.mkdir(parents=True)
    _write(
        bundle / "Starcluster call.md",
        "**Speaker A**: Opening\n*00:00*\n",
    )
    _write(bundle / "analysis" / "action_items.csv", "owner,item\nAlice,Follow up\n")

    manifest = _build_auto_manifest(bundle)
    assert manifest.bundle_id == "starcluster-call-001"
    assert manifest.call_ref.external_source == "filesystem"
    assert manifest.call_ref.external_id == "starcluster-call-001"
    assert manifest.transcript is not None
    assert manifest.transcript.path == "Starcluster call.md"
    assert manifest.transcript.format == "markdown_turns"
    assert len(manifest.analysis) == 1
    assert manifest.analysis[0].kind == "action_items"
    assert manifest.analysis[0].format == "csv"


def test_build_auto_manifest_sanitizes_bundle_id(tmp_path: Path) -> None:
    bundle = tmp_path / "Starcluster Four Way Call 20251126"
    bundle.mkdir(parents=True)
    _write(
        bundle / "call.md",
        "**Speaker A**: Opening\n*00:00*\n",
    )

    manifest = _build_auto_manifest(bundle)
    assert manifest.bundle_id == "Starcluster-Four-Way-Call-20251126"
    assert manifest.call_ref.external_id == "Starcluster-Four-Way-Call-20251126"


def test_auto_embed_call_skips_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_on_success", False)
    result = _auto_embed_call_if_configured(uuid4())
    assert result["status"] == "skipped"
    assert result["reason"] == "disabled"


def test_auto_embed_call_skips_when_embeddings_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_on_success", True)
    monkeypatch.setattr(ingest_fs, "embeddings_enabled", lambda: False)
    result = _auto_embed_call_if_configured(uuid4())
    assert result["status"] == "skipped"
    assert result["reason"] == "embeddings_not_configured"


def test_auto_embed_call_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_on_success", True)
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_fail_on_error", False)
    monkeypatch.setattr(ingest_fs, "embeddings_enabled", lambda: True)

    summary = SimpleNamespace(
        rows_updated=7,
        calls_touched=1,
        ingestion_runs_inserted=1,
        model_used="Qwen/Qwen3-Embedding-4B",
        per_table={"chunks": 4, "artifact_chunks": 3},
    )
    monkeypatch.setattr(ingest_fs, "run_embedding_backfill", lambda **kwargs: summary)

    result = _auto_embed_call_if_configured(uuid4())
    assert result["status"] == "ok"
    assert result["rows_updated"] == 7
    assert result["model_used"] == "Qwen/Qwen3-Embedding-4B"


def test_auto_embed_call_returns_error_when_fail_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_on_success", True)
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_fail_on_error", False)
    monkeypatch.setattr(ingest_fs, "embeddings_enabled", lambda: True)

    def _raise(**kwargs):
        raise EmbeddingClientError("service timeout")

    monkeypatch.setattr(ingest_fs, "run_embedding_backfill", _raise)
    result = _auto_embed_call_if_configured(uuid4())
    assert result["status"] == "error"
    assert "service timeout" in result["error"]


def test_auto_embed_call_raises_when_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_on_success", True)
    monkeypatch.setattr(ingest_fs.settings, "ingest_auto_embed_fail_on_error", True)
    monkeypatch.setattr(ingest_fs, "embeddings_enabled", lambda: True)

    def _raise(**kwargs):
        raise EmbeddingClientError("service timeout")

    monkeypatch.setattr(ingest_fs, "run_embedding_backfill", _raise)
    with pytest.raises(EmbeddingClientError):
        _auto_embed_call_if_configured(uuid4())
