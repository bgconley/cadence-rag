from __future__ import annotations

import json
from pathlib import Path

import app.ingest_adapters as ingest_adapters
import pytest
from app.ingest_adapters import load_analysis_content, load_transcript_payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_transcript_payload_json_turns_list(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.json"
    _write_json(
        transcript_path,
        [
            {
                "speaker": "SE",
                "start_ts_ms": 0,
                "end_ts_ms": 1000,
                "text": "Review architecture options.",
            }
        ],
    )

    payload = load_transcript_payload(transcript_path)
    assert payload.format == "json_turns"
    assert len(payload.content) == 1
    assert payload.content[0].speaker == "SE"
    assert payload.content[0].start_ts_ms == 0
    assert payload.content[0].end_ts_ms == 1000


def test_load_transcript_payload_common_alt_keys(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript_alt.json"
    _write_json(
        transcript_path,
        {
            "segments": [
                {
                    "speakerName": "Customer",
                    "start": 1.5,
                    "end": 3.0,
                    "transcript": "Need object store with tiering.",
                },
                {
                    "participant": "SE",
                    "start_time": "00:00:04.000",
                    "duration_s": 2.5,
                    "content": "We support replication and policy tiers.",
                },
            ]
        },
    )

    payload = load_transcript_payload(transcript_path, format_hint="auto")
    assert len(payload.content) == 2
    assert payload.content[0].speaker == "Customer"
    assert payload.content[0].start_ts_ms == 1500
    assert payload.content[0].end_ts_ms == 3000
    assert payload.content[1].speaker == "SE"
    assert payload.content[1].start_ts_ms == 4000
    assert payload.content[1].end_ts_ms == 6500


def test_load_analysis_content_csv_to_markdown(tmp_path: Path) -> None:
    csv_path = tmp_path / "action_items.csv"
    csv_path.write_text(
        "owner,item,status\nAlice,Validate BOM,open\nBob,Review SSD tiering,in_progress\n",
        encoding="utf-8",
    )

    content = load_analysis_content(csv_path, format_hint="auto")
    assert "| owner | item | status |" in content
    assert "Validate BOM" in content
    assert "Review SSD tiering" in content


def test_load_analysis_content_json_records_to_markdown(tmp_path: Path) -> None:
    json_path = tmp_path / "decisions.json"
    _write_json(
        json_path,
        [
            {"decision": "Use S3 API", "owner": "SE"},
            {"decision": "Pilot on Dell", "owner": "Customer"},
        ],
    )

    content = load_analysis_content(json_path, format_hint="auto")
    assert "| decision | owner |" in content
    assert "Use S3 API" in content
    assert "Pilot on Dell" in content


def test_load_transcript_payload_markdown_turns(tmp_path: Path) -> None:
    md_path = tmp_path / "call.md"
    md_path.write_text(
        "\n".join(
            [
                "**Paul Tran (SMC)**: So far we have folks from Super Micro,",
                "*00:00*",
                "",
                "**Noel A**: Fine, thank you.",
                "*00:44*",
            ]
        ),
        encoding="utf-8",
    )

    payload = load_transcript_payload(md_path, format_hint="auto")
    assert len(payload.content) == 2
    assert payload.content[0].speaker == "Paul Tran (SMC)"
    assert payload.content[0].start_ts_ms == 0
    assert payload.content[0].end_ts_ms == 44000
    assert payload.content[1].speaker == "Noel A"
    assert payload.content[1].start_ts_ms == 44000


def test_load_analysis_content_docx_uses_docx_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docx_path = tmp_path / "analysis.docx"
    docx_path.write_bytes(b"dummy")
    monkeypatch.setattr(ingest_adapters, "_docx_to_text", lambda path: "docx-text")

    content = load_analysis_content(docx_path, format_hint="auto")
    assert content == "docx-text"


def test_load_analysis_content_pdf_uses_pdf_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "analysis.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(ingest_adapters, "_pdf_to_text", lambda path: "pdf-text")

    content = load_analysis_content(pdf_path, format_hint="auto")
    assert content == "pdf-text"


def test_pdf_ocr_fallback_used_for_low_quality_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        ingest_adapters,
        "_extract_pdf_text_with_pypdf",
        lambda path: ("", 2),
    )
    monkeypatch.setattr(
        ingest_adapters,
        "_pdf_to_text_via_ocrmypdf",
        lambda path: "OCR recovered text",
    )
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_enabled", True)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_min_chars", 20)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_min_alpha_ratio", 0.5)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_max_pages", 10)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_force", False)

    content = load_analysis_content(pdf_path, format_hint="pdf")
    assert content == "OCR recovered text"


def test_pdf_ocr_fallback_skipped_for_high_quality_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "text.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    baseline = "This is a normal page with readable extracted text."
    monkeypatch.setattr(
        ingest_adapters,
        "_extract_pdf_text_with_pypdf",
        lambda path: (baseline, 1),
    )

    called = {"value": False}

    def _mock_ocr(path: Path) -> str:
        called["value"] = True
        return "SHOULD NOT BE USED"

    monkeypatch.setattr(ingest_adapters, "_pdf_to_text_via_ocrmypdf", _mock_ocr)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_enabled", True)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_min_chars", 10)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_min_alpha_ratio", 0.2)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_max_pages", 10)
    monkeypatch.setattr(ingest_adapters.settings, "analysis_pdf_ocr_force", False)

    content = load_analysis_content(pdf_path, format_hint="pdf")
    assert content == baseline
    assert called["value"] is False
