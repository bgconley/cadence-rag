from __future__ import annotations

from app.ingest import ChunkingOptions, UtteranceRecord, build_chunks, extract_tech_tokens


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
