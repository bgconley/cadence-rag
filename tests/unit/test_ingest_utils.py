from __future__ import annotations

from app.ingest import (
    ChunkingOptions,
    UtteranceRecord,
    build_artifact_chunks,
    build_chunks,
    extract_tech_tokens,
)


def test_extract_tech_tokens() -> None:
    text = (
        "ECONNRESET at https://example.com for ticket ABC-123 "
        "from 192.168.1.2 HTTP 502 v1.2.3 /var/log/syslog deadbeef"
    )
    tokens = extract_tech_tokens(text)
    expected = {
        "ECONNRESET",
        "https://example.com",
        "ABC-123",
        "192.168.1.2",
        "HTTP 502",
        "v1.2.3",
        "/var/log/syslog",
        "deadbeef",
    }
    assert expected.issubset(set(tokens))


def test_extract_tech_tokens_sales_se_lexicon() -> None:
    text = (
        "The build BOM uses SSD with object store tiering on Lenovo and SMC. "
        "We are in a competitive bake-off head-to-head vs incumbent options: "
        "AWS (Amazon), Azure (Microsoft), GCP (Google Cloud), OCI (Oracle)."
    )
    tokens = set(extract_tech_tokens(text))
    expected = {
        "build",
        "BOM",
        "SSD",
        "object store",
        "tiering",
        "Lenovo",
        "Supermicro",
        "competitive",
        "bake-off",
        "head-to-head",
        "vs",
        "incumbent",
        "AWS",
        "Azure",
        "GCP",
        "OCI",
    }
    assert expected.issubset(tokens)


def test_build_chunks_respects_max_tokens() -> None:
    utterances = [
        UtteranceRecord(
            utterance_id=1,
            speaker="Alice",
            speaker_id=None,
            start_ts_ms=0,
            end_ts_ms=1000,
            confidence=None,
            text="one two three",
            token_count=3,
        ),
        UtteranceRecord(
            utterance_id=2,
            speaker="Bob",
            speaker_id=None,
            start_ts_ms=1000,
            end_ts_ms=2000,
            confidence=None,
            text="four five six",
            token_count=3,
        ),
        UtteranceRecord(
            utterance_id=3,
            speaker="Alice",
            speaker_id=None,
            start_ts_ms=2000,
            end_ts_ms=3000,
            confidence=None,
            text="seven eight nine",
            token_count=3,
        ),
    ]
    options = ChunkingOptions(target_tokens=4, max_tokens=5, overlap_tokens=2)
    chunks = build_chunks(utterances, options)
    assert chunks, "chunks should be created"
    assert all(chunk.token_count <= options.max_tokens for chunk in chunks)


def test_build_artifact_chunks_itemizes_action_items() -> None:
    content = "- File ticket ABC-123.\n- Roll back api-gateway."
    chunks = build_artifact_chunks("action_items", content)
    assert len(chunks) == 2
    assert chunks[0].ordinal == 0
    assert chunks[1].ordinal == 1
    assert "ABC-123" in chunks[0].content
    assert "api-gateway" in chunks[1].content
    assert chunks[0].start_char is not None and chunks[0].end_char is not None


def test_build_artifact_chunks_is_deterministic() -> None:
    content = (
        "Decision summary:\n\n"
        "- Keep Triton as the serving plane.\n"
        "- Use artifact chunks for analysis retrieval."
    )
    first = build_artifact_chunks("decisions", content)
    second = build_artifact_chunks("decisions", content)
    assert [c.ordinal for c in first] == [c.ordinal for c in second]
    assert [c.content for c in first] == [c.content for c in second]
    assert [c.start_char for c in first] == [c.start_char for c in second]
    assert [c.end_char for c in first] == [c.end_char for c in second]
