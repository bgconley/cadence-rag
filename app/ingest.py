from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import text

from .config import settings
from .db import engine
from .schemas import AnalysisArtifactIn, CallRef, ChunkingOptions, UtteranceIn

PIPELINE_VERSION = "v2"
EMBEDDING_CONFIG_DISABLED = {"enabled": False, "model_id": None, "dim": 1024}
NER_CONFIG_DISABLED = {"enabled": False}

TECH_TOKEN_PATTERNS = [
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),  # IPv4
    re.compile(r"\b[A-Z]{2,10}-\d+\b"),  # ticket IDs
    re.compile(r"\bE[A-Z0-9_]{2,}\b"),  # ECONNRESET, EAI_AGAIN, etc
    re.compile(r"\bHTTP\s?\d{3}\b", re.IGNORECASE),
    re.compile(r"\bORA-\d{4,}\b", re.IGNORECASE),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),  # versions
    re.compile(r"\b[a-f0-9]{7,40}\b", re.IGNORECASE),  # commit hashes
    re.compile(r"(?:/[\w.\-]+)+"),  # file paths
]

# Domain lexicon keeps the exact-token lane relevant for sales/SE call retrieval.
DOMAIN_TECH_TOKEN_RULES = [
    (re.compile(r"\bbill of materials\b", re.IGNORECASE), "BOM"),
    (re.compile(r"\bbom\b", re.IGNORECASE), "BOM"),
    (re.compile(r"\bbuild(?:s|ing)?\b", re.IGNORECASE), "build"),
    (re.compile(r"\bssd\b", re.IGNORECASE), "SSD"),
    (
        re.compile(r"\bobject\s+(?:store|storage)\b", re.IGNORECASE),
        "object store",
    ),
    (re.compile(r"\bobject\b", re.IGNORECASE), "object"),
    (re.compile(r"\btiering\b", re.IGNORECASE), "tiering"),
    (re.compile(r"\blenovo\b", re.IGNORECASE), "Lenovo"),
    (re.compile(r"\bdell\b", re.IGNORECASE), "Dell"),
    (re.compile(r"\bsuper[\s-]?micro\b|\bsmc\b", re.IGNORECASE), "Supermicro"),
    (
        re.compile(r"\baws\b|\bamazon web services\b", re.IGNORECASE),
        "AWS",
    ),
    (re.compile(r"\bamazon\b", re.IGNORECASE), "Amazon"),
    (re.compile(r"\bazure\b", re.IGNORECASE), "Azure"),
    (re.compile(r"\bmicrosoft\b", re.IGNORECASE), "Microsoft"),
    (
        re.compile(r"\bgcp\b|\bgoogle cloud(?: platform)?\b", re.IGNORECASE),
        "GCP",
    ),
    (re.compile(r"\bgoogle\b", re.IGNORECASE), "Google"),
    (
        re.compile(r"\boci\b|\boracle cloud(?: infrastructure)?\b", re.IGNORECASE),
        "OCI",
    ),
    (re.compile(r"\boracle\b", re.IGNORECASE), "Oracle"),
    (re.compile(r"\bcompet(?:e|es|ing|ition|itive|itor|itors)\b", re.IGNORECASE), "competitive"),
    (re.compile(r"\bincumbent\b", re.IGNORECASE), "incumbent"),
    (re.compile(r"\bbake[\s-]?off\b", re.IGNORECASE), "bake-off"),
    (re.compile(r"\bhead[\s-]?to[\s-]?head\b", re.IGNORECASE), "head-to-head"),
    (re.compile(r"\bvs\.?(?=\s|$)|\bversus\b", re.IGNORECASE), "vs"),
]

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|(?:\d+[\.\)]))\s+\S")
KIND_ITEMIZED = {"action_items", "decisions"}


@dataclass
class UtteranceRecord:
    utterance_id: int
    speaker: Optional[str]
    speaker_id: Optional[str]
    start_ts_ms: int
    end_ts_ms: int
    confidence: Optional[float]
    text: str
    token_count: int


@dataclass
class ChunkRecord:
    speaker: str
    start_ts_ms: int
    end_ts_ms: int
    token_count: int
    text: str
    utterance_ids: List[int]


@dataclass
class ArtifactChunkRecord:
    ordinal: int
    content: str
    token_count: int
    start_char: Optional[int]
    end_char: Optional[int]
    tech_tokens: List[str]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def extract_tech_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for pattern in TECH_TOKEN_PATTERNS:
        tokens.extend(pattern.findall(text))
    for pattern, canonical in DOMAIN_TECH_TOKEN_RULES:
        if pattern.search(text):
            tokens.append(canonical)
    # normalize and dedupe while preserving order
    seen = set()
    result = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(token)
    return result


def _format_utterance_text(utterance: UtteranceRecord) -> str:
    if utterance.speaker:
        return f"{utterance.speaker}: {utterance.text}"
    return utterance.text


def _normalize_span(
    content: str, start: int, end: int
) -> Optional[Tuple[str, int, int]]:
    if start >= end:
        return None
    raw = content[start:end]
    left_trim = len(raw) - len(raw.lstrip())
    right_trim = len(raw) - len(raw.rstrip())
    span_start = start + left_trim
    span_end = end - right_trim
    if span_start >= span_end:
        return None
    return content[span_start:span_end], span_start, span_end


def _split_paragraph_spans(content: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    start: Optional[int] = None
    cursor = 0

    for line in content.splitlines(keepends=True):
        line_start = cursor
        cursor += len(line)
        if line.strip():
            if start is None:
                start = line_start
        elif start is not None:
            span = _normalize_span(content, start, line_start)
            if span:
                spans.append(span)
            start = None

    if start is not None:
        span = _normalize_span(content, start, len(content))
        if span:
            spans.append(span)

    if not spans and content.strip():
        span = _normalize_span(content, 0, len(content))
        if span:
            spans.append(span)

    return spans


def _split_bullet_spans(
    segment_text: str, base_start: int
) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    saw_bullet = False
    current_start: Optional[int] = None
    cursor = 0

    for line in segment_text.splitlines(keepends=True):
        line_start = cursor
        cursor += len(line)
        if BULLET_LINE_RE.match(line):
            saw_bullet = True
            if current_start is not None:
                span = _normalize_span(segment_text, current_start, line_start)
                if span:
                    text_val, rel_start, rel_end = span
                    spans.append(
                        (text_val, base_start + rel_start, base_start + rel_end)
                    )
            current_start = line_start
            continue

        if current_start is None and line.strip():
            current_start = line_start

    if current_start is not None:
        span = _normalize_span(segment_text, current_start, len(segment_text))
        if span:
            text_val, rel_start, rel_end = span
            spans.append((text_val, base_start + rel_start, base_start + rel_end))

    return spans if saw_bullet else []


def build_artifact_chunks(kind: str, content: str) -> List[ArtifactChunkRecord]:
    chunks: List[ArtifactChunkRecord] = []
    kind_key = kind.strip().lower()
    itemize_kind = kind_key in KIND_ITEMIZED

    ordinal = 0
    for segment_text, segment_start, segment_end in _split_paragraph_spans(content):
        unit_spans: List[Tuple[str, int, int]]
        if itemize_kind:
            bullet_units = _split_bullet_spans(segment_text, segment_start)
            unit_spans = bullet_units if bullet_units else [
                (segment_text, segment_start, segment_end)
            ]
        else:
            unit_spans = [(segment_text, segment_start, segment_end)]

        for unit_text, unit_start, unit_end in unit_spans:
            text_val = unit_text.strip()
            if not text_val:
                continue
            chunks.append(
                ArtifactChunkRecord(
                    ordinal=ordinal,
                    content=text_val,
                    token_count=count_tokens(text_val),
                    start_char=unit_start,
                    end_char=unit_end,
                    tech_tokens=extract_tech_tokens(text_val),
                )
            )
            ordinal += 1

    if chunks:
        return chunks

    fallback = content.strip()
    if not fallback:
        return []
    return [
        ArtifactChunkRecord(
            ordinal=0,
            content=fallback,
            token_count=count_tokens(fallback),
            start_char=0,
            end_char=len(fallback),
            tech_tokens=extract_tech_tokens(fallback),
        )
    ]


def build_chunks(
    utterances: Sequence[UtteranceRecord], options: ChunkingOptions
) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []
    i = 0
    n = len(utterances)
    target = options.target_tokens
    max_tokens = options.max_tokens
    overlap_tokens = options.overlap_tokens

    while i < n:
        current: List[UtteranceRecord] = []
        token_count = 0
        start_index = i

        while i < n:
            u = utterances[i]
            if current and token_count + u.token_count > max_tokens:
                break
            current.append(u)
            token_count += u.token_count
            i += 1
            if token_count >= target:
                break

        if not current:
            u = utterances[i]
            current = [u]
            token_count = u.token_count
            i += 1

        # Determine overlap utterances
        overlap: List[UtteranceRecord] = []
        if overlap_tokens > 0 and current:
            overlap_count = 0
            for u in reversed(current):
                overlap_count += u.token_count
                overlap.append(u)
                if overlap_count >= overlap_tokens:
                    break
            overlap = list(reversed(overlap))

        # Ensure progress
        overlap_len = min(len(overlap), max(len(current) - 1, 0))
        if overlap_len > 0:
            i = max(start_index + 1, i - overlap_len)

        speakers = {u.speaker for u in current if u.speaker}
        speaker = speakers.pop() if len(speakers) == 1 else "MULTI"
        start_ts_ms = current[0].start_ts_ms
        end_ts_ms = current[-1].end_ts_ms
        chunk_text = "\n".join(_format_utterance_text(u) for u in current)

        chunks.append(
            ChunkRecord(
                speaker=speaker or "MULTI",
                start_ts_ms=start_ts_ms,
                end_ts_ms=end_ts_ms,
                token_count=token_count,
                text=chunk_text,
                utterance_ids=[u.utterance_id for u in current],
            )
        )

    return chunks


def _resolve_call_by_id(conn, call_id: UUID):
    row = conn.execute(
        text("SELECT call_id, started_at FROM calls WHERE call_id = :call_id"),
        {"call_id": str(call_id)},
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="call_id not found")
    return row


def _resolve_by_external_id(conn, call_ref: CallRef):
    rows = conn.execute(
        text(
            """
            SELECT call_id, started_at
            FROM calls
            WHERE external_id = :external_id
              AND (external_source IS NOT DISTINCT FROM :external_source)
            """
        ),
        {
            "external_id": call_ref.external_id,
            "external_source": call_ref.external_source,
        },
    ).fetchall()
    if len(rows) > 1:
        raise HTTPException(status_code=409, detail="ambiguous external_id match")
    return rows[0] if rows else None


def _resolve_by_source_hash(conn, call_ref: CallRef):
    rows = conn.execute(
        text(
            """
            SELECT call_id, started_at
            FROM calls
            WHERE source_uri = :source_uri
              AND source_hash = :source_hash
            """
        ),
        {
            "source_uri": call_ref.source_uri,
            "source_hash": call_ref.source_hash,
        },
    ).fetchall()
    if len(rows) > 1:
        raise HTTPException(status_code=409, detail="ambiguous source match")
    return rows[0] if rows else None


def _upsert_call(conn, call_ref: CallRef) -> Tuple[UUID, datetime, bool]:
    row = None
    if call_ref.call_id:
        row = _resolve_call_by_id(conn, call_ref.call_id)
    elif call_ref.external_id:
        row = _resolve_by_external_id(conn, call_ref)
    elif call_ref.source_uri and call_ref.source_hash:
        row = _resolve_by_source_hash(conn, call_ref)

    if row:
        # Update call with any provided fields
        participants_json = (
            json.dumps(call_ref.participants)
            if call_ref.participants is not None
            else None
        )
        metadata_json = (
            json.dumps(call_ref.metadata) if call_ref.metadata is not None else None
        )
        conn.execute(
            text(
                """
                UPDATE calls SET
                  external_id = COALESCE(:external_id, external_id),
                  external_source = COALESCE(:external_source, external_source),
                  started_at = COALESCE(:started_at, started_at),
                  ended_at = COALESCE(:ended_at, ended_at),
                  title = COALESCE(:title, title),
                  source_uri = COALESCE(:source_uri, source_uri),
                  source_hash = COALESCE(:source_hash, source_hash),
                  participants = COALESCE(CAST(:participants AS jsonb), participants),
                  tags = COALESCE(:tags, tags),
                  metadata = COALESCE(CAST(:metadata AS jsonb), metadata)
                WHERE call_id = :call_id
                """
            ),
            {
                "call_id": row[0],
                "external_id": call_ref.external_id,
                "external_source": call_ref.external_source,
                "started_at": call_ref.started_at,
                "ended_at": call_ref.ended_at,
                "title": call_ref.title,
                "source_uri": call_ref.source_uri,
                "source_hash": call_ref.source_hash,
                "participants": participants_json,
                "tags": call_ref.tags,
                "metadata": metadata_json,
            },
        )
        return row[0], row[1], False

    # Create new call
    started_at = call_ref.started_at or _now_utc()
    insert_metadata = call_ref.metadata or {}
    participants_json = (
        json.dumps(call_ref.participants)
        if call_ref.participants is not None
        else None
    )
    row = conn.execute(
        text(
            """
            INSERT INTO calls
              (corpus_id, external_id, external_source, started_at, ended_at, title,
               source_uri, source_hash, participants, tags, metadata)
            VALUES
              (:corpus_id, :external_id, :external_source, :started_at, :ended_at, :title,
               :source_uri, :source_hash, CAST(:participants AS jsonb), :tags, CAST(:metadata AS jsonb))
            RETURNING call_id, started_at
            """
        ),
        {
            "corpus_id": None,
            "external_id": call_ref.external_id,
            "external_source": call_ref.external_source,
            "started_at": started_at,
            "ended_at": call_ref.ended_at,
            "title": call_ref.title,
            "source_uri": call_ref.source_uri,
            "source_hash": call_ref.source_hash,
            "participants": participants_json,
            "tags": call_ref.tags,
            "metadata": json.dumps(insert_metadata),
        },
    ).fetchone()
    return row[0], row[1], True


def resolve_call(call_ref: Optional[CallRef]) -> Tuple[UUID, datetime, bool]:
    call_ref = call_ref or CallRef()
    with engine.begin() as conn:
        call_id, started_at, created = _upsert_call(conn, call_ref)
        return call_id, started_at, created


def ingest_call(call_ref: CallRef) -> Tuple[UUID, bool]:
    call_id, _started_at, created = resolve_call(call_ref)
    return call_id, created


def _record_ingestion_run(
    conn,
    call_id: UUID,
    chunking_config: dict,
    embedding_config: dict,
    ner_config: dict,
) -> None:
    conn.execute(
        text(
            """
            INSERT INTO ingestion_runs
              (call_id, pipeline_version, chunking_config, embedding_config, ner_config)
            VALUES
              (:call_id, :pipeline_version, CAST(:chunking_config AS jsonb),
               CAST(:embedding_config AS jsonb), CAST(:ner_config AS jsonb))
            """
        ),
        {
            "call_id": call_id,
            "pipeline_version": PIPELINE_VERSION,
            "chunking_config": json.dumps(chunking_config),
            "embedding_config": json.dumps(embedding_config),
            "ner_config": json.dumps(ner_config),
        },
    )


def ingest_transcript(
    call_ref: Optional[CallRef],
    utterances_in: Sequence[UtteranceIn],
    options: ChunkingOptions,
) -> Tuple[UUID, int, int]:
    call_id, call_started_at, _created = resolve_call(call_ref)

    utterance_records: List[UtteranceRecord] = []
    with engine.begin() as conn:
        for u in utterances_in:
            text_val = u.text.strip()
            token_count = count_tokens(text_val)
            row = conn.execute(
                text(
                    """
                    INSERT INTO utterances
                      (call_id, speaker, speaker_id, start_ts_ms, end_ts_ms, confidence, text)
                    VALUES
                      (:call_id, :speaker, :speaker_id, :start_ts_ms, :end_ts_ms, :confidence, :text)
                    RETURNING utterance_id
                    """
                ),
                {
                    "call_id": call_id,
                    "speaker": u.speaker,
                    "speaker_id": u.speaker_id,
                    "start_ts_ms": u.start_ts_ms,
                    "end_ts_ms": u.end_ts_ms,
                    "confidence": u.confidence,
                    "text": text_val,
                },
            ).fetchone()
            utterance_records.append(
                UtteranceRecord(
                    utterance_id=row[0],
                    speaker=u.speaker,
                    speaker_id=u.speaker_id,
                    start_ts_ms=u.start_ts_ms,
                    end_ts_ms=u.end_ts_ms,
                    confidence=u.confidence,
                    text=text_val,
                    token_count=token_count,
                )
            )

        chunks = build_chunks(utterance_records, options)
        for chunk in chunks:
            tech_tokens = extract_tech_tokens(chunk.text)
            row = conn.execute(
                text(
                    """
                    INSERT INTO chunks
                      (call_id, call_started_at, speaker, start_ts_ms, end_ts_ms,
                       token_count, text, tech_tokens)
                    VALUES
                      (:call_id, :call_started_at, :speaker, :start_ts_ms, :end_ts_ms,
                       :token_count, :text, :tech_tokens)
                    RETURNING chunk_id
                    """
                ),
                {
                    "call_id": call_id,
                    "call_started_at": call_started_at,
                    "speaker": chunk.speaker,
                    "start_ts_ms": chunk.start_ts_ms,
                    "end_ts_ms": chunk.end_ts_ms,
                    "token_count": chunk.token_count,
                    "text": chunk.text,
                    "tech_tokens": tech_tokens,
                },
            ).fetchone()
            chunk_id = row[0]
            for ordinal, utterance_id in enumerate(chunk.utterance_ids):
                conn.execute(
                    text(
                        """
                        INSERT INTO chunk_utterances (chunk_id, utterance_id, ordinal)
                        VALUES (:chunk_id, :utterance_id, :ordinal)
                        """
                    ),
                    {
                        "chunk_id": chunk_id,
                        "utterance_id": utterance_id,
                        "ordinal": ordinal,
                    },
                )

        _record_ingestion_run(
            conn,
            call_id=call_id,
            chunking_config=options.model_dump(),
            embedding_config=EMBEDDING_CONFIG_DISABLED,
            ner_config=NER_CONFIG_DISABLED,
        )

    return call_id, len(utterance_records), len(chunks)


def ingest_analysis(
    call_ref: CallRef, artifacts: Sequence[AnalysisArtifactIn]
) -> Tuple[UUID, int]:
    call_id, call_started_at, _created = resolve_call(call_ref)
    with engine.begin() as conn:
        for artifact in artifacts:
            content = artifact.content.strip()
            token_count = count_tokens(content)
            tech_tokens = extract_tech_tokens(content)
            row = conn.execute(
                text(
                    """
                    INSERT INTO analysis_artifacts
                      (call_id, call_started_at, kind, content, token_count, tech_tokens, metadata)
                    VALUES
                      (:call_id, :call_started_at, :kind, :content, :token_count, :tech_tokens, CAST(:metadata AS jsonb))
                    RETURNING artifact_id
                    """
                ),
                {
                    "call_id": call_id,
                    "call_started_at": call_started_at,
                    "kind": artifact.kind,
                    "content": content,
                    "token_count": token_count,
                    "tech_tokens": tech_tokens,
                    "metadata": json.dumps(artifact.metadata or {}),
                },
            ).fetchone()
            artifact_id = row[0]

            for chunk in build_artifact_chunks(artifact.kind, content):
                conn.execute(
                    text(
                        """
                        INSERT INTO artifact_chunks
                          (artifact_id, call_id, call_started_at, kind, ordinal, content,
                           token_count, start_char, end_char, tech_tokens, metadata)
                        VALUES
                          (:artifact_id, :call_id, :call_started_at, :kind, :ordinal, :content,
                           :token_count, :start_char, :end_char, :tech_tokens, CAST(:metadata AS jsonb))
                        """
                    ),
                    {
                        "artifact_id": artifact_id,
                        "call_id": call_id,
                        "call_started_at": call_started_at,
                        "kind": artifact.kind,
                        "ordinal": chunk.ordinal,
                        "content": chunk.content,
                        "token_count": chunk.token_count,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "tech_tokens": chunk.tech_tokens,
                        "metadata": json.dumps(artifact.metadata or {}),
                    },
                )
        _record_ingestion_run(
            conn,
            call_id=call_id,
            chunking_config={
                "enabled": True,
                "mode": "analysis_artifact_chunks_v1",
                "itemized_kinds": sorted(KIND_ITEMIZED),
            },
            embedding_config=EMBEDDING_CONFIG_DISABLED,
            ner_config=NER_CONFIG_DISABLED,
        )
    return call_id, len(artifacts)
