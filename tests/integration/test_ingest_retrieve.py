from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient


def test_ingest_and_retrieve_roundtrip(client: TestClient) -> None:
    external_id = f"call-{uuid4().hex[:8]}"
    call_payload = {
        "call_ref": {
            "external_source": "test",
            "external_id": external_id,
            "started_at": "2026-02-03T00:00:00Z",
            "title": "Smoke Test Call",
        }
    }
    call_resp = client.post("/ingest/call", json=call_payload)
    assert call_resp.status_code == 200
    assert call_resp.json()["created"] is True

    transcript_payload = {
        "call_ref": {
            "external_source": "test",
            "external_id": external_id,
            "started_at": "2026-02-03T00:00:00Z",
            "title": "Smoke Test Call",
        },
        "transcript": {
            "format": "json_turns",
            "content": [
                {
                    "speaker": "Alice",
                    "start_ts_ms": 0,
                    "end_ts_ms": 5000,
                    "text": "We saw ECONNRESET in api-gateway.",
                },
                {
                    "speaker": "Bob",
                    "start_ts_ms": 5000,
                    "end_ts_ms": 10000,
                    "text": "Let's roll back version 1.2.3.",
                },
                {
                    "speaker": "Alice",
                    "start_ts_ms": 10000,
                    "end_ts_ms": 15000,
                    "text": "Action item: file ticket ABC-123.",
                },
            ],
        },
    }
    transcript_resp = client.post("/ingest/transcript", json=transcript_payload)
    assert transcript_resp.status_code == 200
    call_id = transcript_resp.json()["call_id"]

    analysis_payload = {
        "call_ref": {"external_source": "test", "external_id": external_id},
        "artifacts": [
            {
                "kind": "summary",
                "content": "We saw ECONNRESET in api-gateway and planned rollback.",
            }
        ],
    }
    analysis_resp = client.post("/ingest/analysis", json=analysis_payload)
    assert analysis_resp.status_code == 200
    assert analysis_resp.json()["call_id"] == call_id

    retrieve_payload = {
        "query": "Where did we discuss ECONNRESET in api-gateway?",
        "intent": "troubleshooting",
        "budget": {"max_evidence_items": 6, "max_total_chars": 2000},
    }
    retrieve_resp = client.post("/retrieve", json=retrieve_payload)
    assert retrieve_resp.status_code == 200
    body = retrieve_resp.json()
    assert body["artifacts"], "expected at least one analysis artifact chunk"
    assert all(
        item["evidence_id"].startswith("A-") for item in body["artifacts"]
    )
    assert all("artifact_chunk_id" in item for item in body["artifacts"])
    assert body["quotes"], "expected at least one quote"
    assert any("ECONNRESET" in quote["snippet"] for quote in body["quotes"])


def test_retrieve_with_call_ref_filter(client: TestClient) -> None:
    external_id = f"call-{uuid4().hex[:8]}"
    transcript_payload = {
        "call_ref": {
            "external_source": "test",
            "external_id": external_id,
            "started_at": "2026-02-03T00:00:00Z",
            "title": "Filter Test Call",
        },
        "transcript": {
            "format": "json_turns",
            "content": [
                {
                    "speaker": "Alice",
                    "start_ts_ms": 0,
                    "end_ts_ms": 1000,
                    "text": "ECONNRESET happened once.",
                }
            ],
        },
    }
    transcript_resp = client.post("/ingest/transcript", json=transcript_payload)
    assert transcript_resp.status_code == 200

    retrieve_payload = {
        "query": "ECONNRESET",
        "filters": {"external_source": "test", "external_id": external_id},
    }
    retrieve_resp = client.post("/retrieve", json=retrieve_payload)
    assert retrieve_resp.status_code == 200
    body = retrieve_resp.json()
    assert body["quotes"], "expected quotes for filtered retrieval"
    assert all(
        quote["call_id"] == body["quotes"][0]["call_id"] for quote in body["quotes"]
    )


def test_expand_and_browse_endpoints(client: TestClient) -> None:
    external_id = f"call-{uuid4().hex[:8]}"
    transcript_payload = {
        "call_ref": {
            "external_source": "test",
            "external_id": external_id,
            "started_at": "2026-02-03T00:00:00Z",
            "title": "Browse Test Call",
        },
        "transcript": {
            "format": "json_turns",
            "content": [
                {
                    "speaker": "Alice",
                    "start_ts_ms": 0,
                    "end_ts_ms": 1000,
                    "text": "We saw ECONNRESET in api-gateway.",
                },
                {
                    "speaker": "Bob",
                    "start_ts_ms": 1000,
                    "end_ts_ms": 2000,
                    "text": "Action item: file ticket ABC-123.",
                },
            ],
        },
    }
    transcript_resp = client.post("/ingest/transcript", json=transcript_payload)
    assert transcript_resp.status_code == 200
    call_id = transcript_resp.json()["call_id"]

    calls_resp = client.get("/calls", params={"external_id": external_id})
    assert calls_resp.status_code == 200
    items = calls_resp.json()["items"]
    assert any(item["call_id"] == call_id for item in items)

    call_resp = client.get(f"/calls/{call_id}")
    assert call_resp.status_code == 200
    assert call_resp.json()["counts"]["chunks"] >= 1

    retrieve_resp = client.post(
        "/retrieve",
        json={"query": "ECONNRESET", "filters": {"external_id": external_id}},
    )
    assert retrieve_resp.status_code == 200
    evidence_id = retrieve_resp.json()["quotes"][0]["evidence_id"]

    expand_resp = client.post(
        "/expand",
        json={"evidence_id": evidence_id, "window_ms": 5000, "max_chars": 2000},
    )
    assert expand_resp.status_code == 200
    assert "ECONNRESET" in expand_resp.json()["snippet"]


def test_retrieve_ids_only_stable(client: TestClient) -> None:
    external_id = f"call-{uuid4().hex[:8]}"
    transcript_payload = {
        "call_ref": {
            "external_source": "test",
            "external_id": external_id,
            "started_at": "2026-02-03T00:00:00Z",
            "title": "IDs Only Call",
        },
        "transcript": {
            "format": "json_turns",
            "content": [
                {
                    "speaker": "Alice",
                    "start_ts_ms": 0,
                    "end_ts_ms": 1000,
                    "text": "We saw ECONNRESET in api-gateway.",
                }
            ],
        },
    }
    assert client.post("/ingest/transcript", json=transcript_payload).status_code == 200

    analysis_payload = {
        "call_ref": {"external_source": "test", "external_id": external_id},
        "artifacts": [
            {
                "kind": "summary",
                "content": "ECONNRESET appeared in the gateway logs.",
            }
        ],
    }
    assert client.post("/ingest/analysis", json=analysis_payload).status_code == 200

    retrieve_payload = {"query": "ECONNRESET", "return_style": "ids_only"}
    resp_one = client.post("/retrieve", json=retrieve_payload)
    resp_two = client.post("/retrieve", json=retrieve_payload)
    assert resp_one.status_code == 200
    assert resp_two.status_code == 200
    ids_one = resp_one.json()["retrieved_ids"]
    ids_two = resp_two.json()["retrieved_ids"]
    assert ids_one == ids_two
    assert any(item.startswith("chunk:") for item in ids_one)
    assert any(item.startswith("artifact_chunk:") for item in ids_one)


def test_analysis_evidence_roundtrips_via_expand(client: TestClient) -> None:
    external_id = f"call-{uuid4().hex[:8]}"
    analysis_payload = {
        "call_ref": {
            "external_source": "test",
            "external_id": external_id,
            "started_at": "2026-02-03T00:00:00Z",
            "title": "Artifact Expand Call",
        },
        "artifacts": [
            {
                "kind": "action_items",
                "content": "- File ticket ABC-123 for ECONNRESET.\n- Roll back api-gateway to v1.2.3.",
            }
        ],
    }
    assert client.post("/ingest/analysis", json=analysis_payload).status_code == 200

    retrieve_resp = client.post(
        "/retrieve",
        json={
            "query": "ABC-123",
            "filters": {"external_source": "test", "external_id": external_id},
        },
    )
    assert retrieve_resp.status_code == 200
    artifacts = retrieve_resp.json()["artifacts"]
    assert artifacts, "expected analysis evidence for ticket query"
    evidence_id = artifacts[0]["evidence_id"]
    assert evidence_id.startswith("A-")

    expand_resp = client.post(
        "/expand",
        json={"evidence_id": evidence_id, "max_chars": 2000},
    )
    assert expand_resp.status_code == 200
    expanded = expand_resp.json()
    assert expanded["artifact_chunk_id"] > 0
    assert "ABC-123" in expanded["snippet"]


def test_retrieve_respects_budget(client: TestClient) -> None:
    external_id = f"call-{uuid4().hex[:8]}"
    transcript_payload = {
        "call_ref": {
            "external_source": "test",
            "external_id": external_id,
            "started_at": "2026-02-03T00:00:00Z",
            "title": "Budget Test Call",
        },
        "transcript": {
            "format": "json_turns",
            "content": [
                {
                    "speaker": "Alice",
                    "start_ts_ms": 0,
                    "end_ts_ms": 1000,
                    "text": "We saw ECONNRESET in api-gateway.",
                }
            ],
        },
    }
    assert client.post("/ingest/transcript", json=transcript_payload).status_code == 200

    analysis_payload = {
        "call_ref": {"external_source": "test", "external_id": external_id},
        "artifacts": [
            {
                "kind": "summary",
                "content": "ECONNRESET appeared in the gateway logs.",
            }
        ],
    }
    assert client.post("/ingest/analysis", json=analysis_payload).status_code == 200

    retrieve_payload = {
        "query": "ECONNRESET",
        "budget": {"max_evidence_items": 1, "max_total_chars": 20},
    }
    retrieve_resp = client.post("/retrieve", json=retrieve_payload)
    assert retrieve_resp.status_code == 200
    body = retrieve_resp.json()
    total_items = len(body["artifacts"]) + len(body["quotes"])
    assert total_items <= 1
    for item in body["artifacts"] + body["quotes"]:
        assert len(item["snippet"]) <= 20
