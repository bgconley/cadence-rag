from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text


def test_ingest_jobs_endpoints_return_job_and_files(client: TestClient) -> None:
    import app.db as db_module

    bundle_id = f"bundle-{uuid4().hex[:8]}"
    with db_module.engine.begin() as conn:
        job_id = conn.execute(
            text(
                """
                INSERT INTO ingest_jobs
                  (bundle_id, status, queue_name, source_path, manifest_path, call_ref)
                VALUES
                  (:bundle_id, 'queued', 'ingest', '/tmp/source', '/tmp/source/manifest.json',
                   CAST(:call_ref AS jsonb))
                RETURNING ingest_job_id
                """
            ),
            {"bundle_id": bundle_id, "call_ref": json.dumps({"external_id": bundle_id})},
        ).scalar_one()
        conn.execute(
            text(
                """
                INSERT INTO ingest_job_files
                  (ingest_job_id, kind, relative_path, file_sha256, file_size_bytes)
                VALUES
                  (:ingest_job_id, 'manifest', 'manifest.json', 'abc123', 42)
                """
            ),
            {"ingest_job_id": job_id},
        )

    list_resp = client.get("/ingest/jobs")
    assert list_resp.status_code == 200
    items = list_resp.json()["items"]
    assert any(item["ingest_job_id"] == str(job_id) for item in items)

    job_resp = client.get(f"/ingest/jobs/{job_id}")
    assert job_resp.status_code == 200
    payload = job_resp.json()
    assert payload["bundle_id"] == bundle_id
    assert payload["files"][0]["relative_path"] == "manifest.json"


def test_scan_inbox_once_enqueues_and_tracks_job(
    client: TestClient, tmp_path: Path, monkeypatch
) -> None:
    import app.ingest_fs as ingest_fs

    root = tmp_path / "ingest"
    bundle = root / "inbox" / "scan-call-001"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "_READY").write_text("", encoding="utf-8")
    (bundle / "transcript.json").write_text(
        json.dumps(
            [
                {
                    "speaker": "Alice",
                    "start_ts_ms": 0,
                    "end_ts_ms": 1000,
                    "text": "Discussed Lenovo vs Dell for object store tiering.",
                }
            ]
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_id": "scan-call-001",
                "call_ref": {"external_source": "test", "external_id": "scan-call-001"},
                "transcript": {"path": "transcript.json", "format": "json_turns"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ingest_fs.settings, "ingest_root_dir", str(root))
    monkeypatch.setattr(ingest_fs, "_enqueue_job", lambda ingest_job_id: str(ingest_job_id))

    summary = ingest_fs.scan_inbox_once()
    assert summary["discovered"] == 1
    assert summary["queued"] == 1
    assert summary["invalid"] == 0

    jobs_resp = client.get("/ingest/jobs", params={"status": "queued"})
    assert jobs_resp.status_code == 200
    assert any(item["bundle_id"] == "scan-call-001" for item in jobs_resp.json()["items"])
