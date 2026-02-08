from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


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
    target_tokens: int = Field(default=350, ge=1)
    max_tokens: int = Field(default=600, ge=1)
    overlap_tokens: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def validate_relationships(self) -> "ChunkingOptions":
        if self.max_tokens < self.target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("overlap_tokens must be < target_tokens")
        return self


class TranscriptIngestRequest(BaseModel):
    call_ref: Optional[CallRef] = None
    transcript: TranscriptPayload
    options: Optional[ChunkingOptions] = None


class AnalysisArtifactIn(BaseModel):
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
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
    window_ms: Optional[int] = Field(default=None, ge=0)
    max_chars: int = Field(default=2000, ge=1, le=20000)
