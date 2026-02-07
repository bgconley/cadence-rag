from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.config import settings
from app.retrieve import _choose_dense_mode
from app.schemas import RetrieveFilters


def test_choose_dense_mode_exact_for_small_scoped_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embeddings_exact_scan_threshold", 2000)
    filters = RetrieveFilters(call_ids=[uuid4()])
    mode = _choose_dense_mode(
        estimated_rows=200, filters=filters, call_ids=filters.call_ids
    )
    assert mode == "exact"


def test_choose_dense_mode_ann_for_large_scoped_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embeddings_exact_scan_threshold", 2000)
    filters = RetrieveFilters(
        date_from=datetime.now(timezone.utc),
        date_to=datetime.now(timezone.utc),
    )
    mode = _choose_dense_mode(estimated_rows=5000, filters=filters, call_ids=None)
    assert mode == "ann"


def test_choose_dense_mode_ann_for_unscoped_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embeddings_exact_scan_threshold", 5000)
    mode = _choose_dense_mode(estimated_rows=100, filters=None, call_ids=None)
    assert mode == "ann"


def test_choose_dense_mode_exact_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "embeddings_exact_scan_threshold", 10)
    mode = _choose_dense_mode(estimated_rows=0, filters=None, call_ids=None)
    assert mode == "exact"
