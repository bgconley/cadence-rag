from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingest_fs import validate_bundle_directory


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
