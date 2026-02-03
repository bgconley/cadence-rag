from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CallRef(BaseModel):
    call_id: Optional[UUID] = None
    external_id: Optional[str] = None
    external_source: Optional[str] = None
    source_uri: Optional[str] = None
    source_hash: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    title: Optional[str] = None
    participants: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class UtteranceIn(BaseModel):
    speaker: Optional[str] = None
    speaker_id: Optional[str] = None
    start_ts_ms: int
    end_ts_ms: int
    confidence: Optional[float] = None
    text: str


class TranscriptPayload(BaseModel):
    format: Literal["json_turns"] = "json_turns"
    content: List[UtteranceIn]


class ChunkingOptions(BaseModel):
    target_tokens: int = 350
    max_tokens: int = 600
    overlap_tokens: int = 50


class TranscriptIngestRequest(BaseModel):
    call_ref: Optional[CallRef] = None
    transcript: TranscriptPayload
    options: Optional[ChunkingOptions] = None


class AnalysisArtifactIn(BaseModel):
    kind: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class AnalysisIngestRequest(BaseModel):
    call_ref: CallRef
    artifacts: List[AnalysisArtifactIn] = Field(default_factory=list)


class CallIngestRequest(BaseModel):
    call_ref: CallRef


class Budget(BaseModel):
    max_evidence_items: int = 8
    max_total_chars: int = 6000


class RetrieveFilters(BaseModel):
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    call_ids: Optional[List[UUID]] = None
    external_id: Optional[str] = None
    external_source: Optional[str] = None
    call_tags: Optional[List[str]] = None


class RetrieveRequest(BaseModel):
    query: str
    intent: Literal[
        "auto", "decision", "action_items", "who_said", "troubleshooting", "status"
    ] = "auto"
    filters: Optional[RetrieveFilters] = None
    budget: Budget = Field(default_factory=Budget)
    return_style: Literal["evidence_pack_json", "ids_only"] = "evidence_pack_json"
    debug: bool = False


class ExpandRequest(BaseModel):
    evidence_id: str
    window_ms: Optional[int] = None
    max_chars: int = 2000
