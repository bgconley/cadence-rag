from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import AnalysisArtifactIn, ChunkingOptions, ExpandRequest


def test_chunking_options_relationship_validation() -> None:
    ChunkingOptions(target_tokens=350, max_tokens=600, overlap_tokens=50)

    with pytest.raises(ValidationError):
        ChunkingOptions(target_tokens=400, max_tokens=399, overlap_tokens=50)

    with pytest.raises(ValidationError):
        ChunkingOptions(target_tokens=200, max_tokens=300, overlap_tokens=200)


def test_analysis_artifact_kind_pattern() -> None:
    AnalysisArtifactIn(kind="action_items", content="x")

    with pytest.raises(ValidationError):
        AnalysisArtifactIn(kind="Action Items", content="x")


def test_expand_request_bounds() -> None:
    ExpandRequest(evidence_id="Q-1", max_chars=1000)

    with pytest.raises(ValidationError):
        ExpandRequest(evidence_id="Q-1", max_chars=50000)
