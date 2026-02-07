from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence
from uuid import UUID

from pydantic import BaseModel, Field
from redis import Redis
from rq import Queue
from sqlalchemy import text

from .config import settings
from .db import engine
from .ingest import ingest_analysis, ingest_call, ingest_transcript
from .schemas import AnalysisArtifactIn, CallRef, ChunkingOptions, TranscriptPayload

INGEST_JOB_STATUS = Literal["queued", "running", "succeeded", "failed", "invalid"]
STATUS_QUEUED: INGEST_JOB_STATUS = "queued"
STATUS_RUNNING: INGEST_JOB_STATUS = "running"
STATUS_SUCCEEDED: INGEST_JOB_STATUS = "succeeded"
STATUS_FAILED: INGEST_JOB_STATUS = "failed"
STATUS_INVALID: INGEST_JOB_STATUS = "invalid"

BUNDLE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]{1,120}$")


class TranscriptFileRef(BaseModel):
    path: str = "transcript.json"
    format: Literal["json_turns"] = "json_turns"
    sha256: Optional[str] = None
    options: Optional[ChunkingOptions] = None


class AnalysisFileRef(BaseModel):
    kind: str
    path: str
    sha256: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BundleManifest(BaseModel):
    bundle_id: Optional[str] = None
    call_ref: CallRef
    transcript: Optional[TranscriptFileRef] = None
    analysis: List[AnalysisFileRef] = Field(default_factory=list)


@dataclass(frozen=True)
class BundleFileRecord:
    kind: str
    relative_path: str
    absolute_path: Path
    file_sha256: str
    file_size_bytes: int


@dataclass(frozen=True)
class ValidatedBundle:
    bundle_id: str
    bundle_path: Path
    manifest_path: Path
    manifest: BundleManifest
    files: List[BundleFileRecord]


def _resolve_ingest_root() -> Path:
    return Path(settings.ingest_root_dir).expanduser().resolve()


def _ensure_ingest_dirs() -> Dict[str, Path]:
    root = _resolve_ingest_root()
    inbox = root / "inbox"
    processing = root / "processing"
    done = root / "done"
    failed = root / "failed"
    for path in (root, inbox, processing, done, failed):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "inbox": inbox,
        "processing": processing,
        "done": done,
        "failed": failed,
    }


def _safe_join(bundle_path: Path, relative_path: str) -> Path:
    bundle_resolved = bundle_path.resolve()
    candidate = (bundle_path / relative_path).resolve()
    if candidate != bundle_resolved and bundle_resolved not in candidate.parents:
        raise ValueError(f"path escapes bundle root: {relative_path}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path: Path) -> BundleManifest:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return BundleManifest.model_validate(raw)


def _normalize_bundle_id(bundle_path: Path, manifest: BundleManifest) -> str:
    bundle_id = manifest.bundle_id or bundle_path.name
    if not BUNDLE_ID_RE.fullmatch(bundle_id):
        raise ValueError(
            "bundle_id must match [a-zA-Z0-9._-]{1,120} "
            f"(got {bundle_id!r})"
        )
    return bundle_id


def _validate_file_ref(
    bundle_path: Path,
    kind: str,
    relative_path: str,
    expected_sha256: Optional[str],
) -> BundleFileRecord:
    absolute_path = _safe_join(bundle_path, relative_path)
    if not absolute_path.exists():
        raise ValueError(f"missing file: {relative_path}")
    if not absolute_path.is_file():
        raise ValueError(f"not a regular file: {relative_path}")
    observed_hash = _sha256_file(absolute_path)
    if expected_sha256 and observed_hash.lower() != expected_sha256.lower():
        raise ValueError(
            f"sha256 mismatch for {relative_path}: "
            f"expected {expected_sha256}, got {observed_hash}"
        )
    return BundleFileRecord(
        kind=kind,
        relative_path=relative_path,
        absolute_path=absolute_path,
        file_sha256=observed_hash,
        file_size_bytes=absolute_path.stat().st_size,
    )


def validate_bundle_directory(bundle_path: Path) -> ValidatedBundle:
    bundle_path = bundle_path.resolve()
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("manifest.json is required")
    manifest = _load_manifest(manifest_path)
    bundle_id = _normalize_bundle_id(bundle_path, manifest)
    if manifest.transcript is None and not manifest.analysis:
        raise ValueError("manifest must include transcript and/or analysis entries")

    files: List[BundleFileRecord] = []
    files.append(
        _validate_file_ref(
            bundle_path=bundle_path,
            kind="manifest",
            relative_path="manifest.json",
            expected_sha256=None,
        )
    )
    if manifest.transcript is not None:
        files.append(
            _validate_file_ref(
                bundle_path=bundle_path,
                kind="transcript",
                relative_path=manifest.transcript.path,
                expected_sha256=manifest.transcript.sha256,
            )
        )
    for analysis in manifest.analysis:
        files.append(
            _validate_file_ref(
                bundle_path=bundle_path,
                kind=f"analysis:{analysis.kind}",
                relative_path=analysis.path,
                expected_sha256=analysis.sha256,
            )
        )
    return ValidatedBundle(
        bundle_id=bundle_id,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        manifest=manifest,
        files=files,
    )


def _move_bundle(src: Path, dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / src.name
    if target.exists():
        target = dest_root / f"{src.name}-{int(time.time())}"
    shutil.move(str(src), str(target))
    return target.resolve()


def _redis() -> Redis:
    return Redis.from_url(settings.redis_url)


def _serialize_job(row: Dict[str, Any], files: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ingest_job_id": str(row["ingest_job_id"]),
        "bundle_id": row["bundle_id"],
        "status": row["status"],
        "queue_name": row["queue_name"],
        "source_path": row["source_path"],
        "manifest_path": row["manifest_path"],
        "call_ref": row["call_ref"] or {},
        "call_id": str(row["call_id"]) if row["call_id"] else None,
        "error": row["error"],
        "attempts": row["attempts"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        "started_at": row["started_at"].isoformat() if row["started_at"] else None,
        "completed_at": row["completed_at"].isoformat()
        if row["completed_at"]
        else None,
        "files": [
            {
                "kind": file_row["kind"],
                "relative_path": file_row["relative_path"],
                "file_sha256": file_row["file_sha256"],
                "file_size_bytes": file_row["file_size_bytes"],
            }
            for file_row in files
        ],
    }


def _fetch_job_files(conn, ingest_job_id: UUID) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text(
            """
            SELECT kind, relative_path, file_sha256, file_size_bytes
            FROM ingest_job_files
            WHERE ingest_job_id = :ingest_job_id
            ORDER BY ingest_job_file_id ASC
            """
        ),
        {"ingest_job_id": ingest_job_id},
    ).mappings()
    return [dict(row) for row in rows]


def _create_or_get_job(
    bundle_id: str,
    source_path: Path,
    manifest_path: Path,
    call_ref: Dict[str, Any],
    status: INGEST_JOB_STATUS = STATUS_QUEUED,
    error: Optional[str] = None,
) -> tuple[UUID, bool]:
    with engine.begin() as conn:
        created = conn.execute(
            text(
                """
                INSERT INTO ingest_jobs
                  (bundle_id, status, queue_name, source_path, manifest_path, call_ref, error)
                VALUES
                  (:bundle_id, :status, :queue_name, :source_path, :manifest_path,
                   CAST(:call_ref AS jsonb), :error)
                ON CONFLICT (bundle_id) DO NOTHING
                RETURNING ingest_job_id
                """
            ),
            {
                "bundle_id": bundle_id,
                "status": status,
                "queue_name": settings.ingest_queue_name,
                "source_path": str(source_path),
                "manifest_path": str(manifest_path),
                "call_ref": json.dumps(call_ref),
                "error": error,
            },
        ).fetchone()
        if created:
            return created[0], True

        row = conn.execute(
            text(
                """
                SELECT ingest_job_id
                FROM ingest_jobs
                WHERE bundle_id = :bundle_id
                """
            ),
            {"bundle_id": bundle_id},
        ).fetchone()
        if row is None:
            raise RuntimeError(f"failed to create or fetch ingest job for {bundle_id}")
        return row[0], False


def _upsert_job_files(ingest_job_id: UUID, files: Sequence[BundleFileRecord]) -> None:
    with engine.begin() as conn:
        for file_rec in files:
            conn.execute(
                text(
                    """
                    INSERT INTO ingest_job_files
                      (ingest_job_id, kind, relative_path, file_sha256, file_size_bytes)
                    VALUES
                      (:ingest_job_id, :kind, :relative_path, :file_sha256, :file_size_bytes)
                    ON CONFLICT (ingest_job_id, relative_path)
                    DO UPDATE SET
                      kind = EXCLUDED.kind,
                      file_sha256 = EXCLUDED.file_sha256,
                      file_size_bytes = EXCLUDED.file_size_bytes
                    """
                ),
                {
                    "ingest_job_id": ingest_job_id,
                    "kind": file_rec.kind,
                    "relative_path": file_rec.relative_path,
                    "file_sha256": file_rec.file_sha256,
                    "file_size_bytes": file_rec.file_size_bytes,
                },
            )


def update_ingest_job_status(
    ingest_job_id: UUID,
    status: INGEST_JOB_STATUS,
    *,
    call_id: Optional[UUID] = None,
    error: Optional[str] = None,
    started: bool = False,
    completed: bool = False,
    increment_attempts: bool = False,
) -> None:
    set_clauses = ["status = :status", "updated_at = now()"]
    params: Dict[str, Any] = {"ingest_job_id": ingest_job_id, "status": status}
    if call_id is not None:
        set_clauses.append("call_id = :call_id")
        params["call_id"] = call_id
    if error is not None:
        set_clauses.append("error = :error")
        params["error"] = error
    if started:
        set_clauses.append("started_at = now()")
    if completed:
        set_clauses.append("completed_at = now()")
    if increment_attempts:
        set_clauses.append("attempts = attempts + 1")

    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                UPDATE ingest_jobs
                SET {", ".join(set_clauses)}
                WHERE ingest_job_id = :ingest_job_id
                """
            ),
            params,
        )


def get_ingest_job(ingest_job_id: UUID) -> Dict[str, Any]:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT ingest_job_id, bundle_id, status, queue_name, source_path,
                       manifest_path, call_ref, call_id, error, attempts,
                       created_at, updated_at, started_at, completed_at
                FROM ingest_jobs
                WHERE ingest_job_id = :ingest_job_id
                """
            ),
            {"ingest_job_id": ingest_job_id},
        ).mappings().first()
        if row is None:
            raise KeyError(f"ingest job not found: {ingest_job_id}")
        files = _fetch_job_files(conn, ingest_job_id)
        return _serialize_job(dict(row), files)


def list_ingest_jobs(
    *,
    status: Optional[INGEST_JOB_STATUS] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    where_clause = ""
    params: Dict[str, Any] = {"limit": limit}
    if status is not None:
        where_clause = "WHERE status = :status"
        params["status"] = status

    with engine.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT ingest_job_id, bundle_id, status, queue_name, source_path,
                       manifest_path, call_ref, call_id, error, attempts,
                       created_at, updated_at, started_at, completed_at
                FROM ingest_jobs
                {where_clause}
                ORDER BY created_at DESC, ingest_job_id DESC
                LIMIT :limit
                """
            ),
            params,
        ).mappings()
        items: List[Dict[str, Any]] = []
        for row in rows:
            ingest_job_id = row["ingest_job_id"]
            files = _fetch_job_files(conn, ingest_job_id)
            items.append(_serialize_job(dict(row), files))
    return {"items": items}


def _enqueue_job(ingest_job_id: UUID) -> str:
    queue = Queue(settings.ingest_queue_name, connection=_redis())
    rq_job = queue.enqueue(
        "app.ingest_fs.process_ingest_job",
        str(ingest_job_id),
        job_id=str(ingest_job_id),
    )
    return rq_job.id


def _record_invalid_bundle(bundle_path: Path, error: str) -> None:
    bundle_id = bundle_path.name
    manifest_path = bundle_path / "manifest.json"
    _create_or_get_job(
        bundle_id=bundle_id,
        source_path=bundle_path,
        manifest_path=manifest_path,
        call_ref={},
        status=STATUS_INVALID,
        error=error,
    )


def scan_inbox_once() -> Dict[str, Any]:
    paths = _ensure_ingest_dirs()
    discovered = 0
    queued = 0
    duplicates = 0
    invalid = 0

    for candidate in sorted(paths["inbox"].iterdir()):
        if not candidate.is_dir():
            continue
        if not (candidate / "_READY").exists():
            continue
        discovered += 1

        try:
            validated = validate_bundle_directory(candidate)
        except Exception as exc:
            invalid += 1
            _record_invalid_bundle(candidate, str(exc))
            _move_bundle(candidate, paths["failed"])
            continue

        processing_path = _move_bundle(candidate, paths["processing"])
        processing_manifest_path = processing_path / "manifest.json"
        job_id, created = _create_or_get_job(
            bundle_id=validated.bundle_id,
            source_path=processing_path,
            manifest_path=processing_manifest_path,
            call_ref=validated.manifest.call_ref.model_dump(
                mode="json", exclude_none=True
            ),
            status=STATUS_QUEUED,
        )
        if not created:
            duplicates += 1
            update_ingest_job_status(
                job_id,
                STATUS_INVALID,
                error=f"duplicate bundle_id={validated.bundle_id}",
                completed=True,
            )
            _move_bundle(processing_path, paths["failed"])
            continue

        rel_files: List[BundleFileRecord] = []
        for file_rec in validated.files:
            relative = str(file_rec.absolute_path.relative_to(validated.bundle_path))
            relocated = processing_path / relative
            rel_files.append(
                BundleFileRecord(
                    kind=file_rec.kind,
                    relative_path=relative,
                    absolute_path=relocated,
                    file_sha256=file_rec.file_sha256,
                    file_size_bytes=file_rec.file_size_bytes,
                )
            )
        _upsert_job_files(job_id, rel_files)
        _enqueue_job(job_id)
        queued += 1

    return {
        "discovered": discovered,
        "queued": queued,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def _load_transcript_file(path: Path) -> TranscriptPayload:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {"format": "json_turns", "content": raw}
    return TranscriptPayload.model_validate(raw)


def process_ingest_job(ingest_job_id: str) -> Dict[str, Any]:
    job_uuid = UUID(ingest_job_id)
    job = get_ingest_job(job_uuid)
    source_path = Path(job["source_path"]).resolve()
    manifest_path = Path(job["manifest_path"]).resolve()
    paths = _ensure_ingest_dirs()

    update_ingest_job_status(
        job_uuid,
        STATUS_RUNNING,
        error=None,
        started=True,
        increment_attempts=True,
    )

    try:
        manifest = _load_manifest(manifest_path)
        validated = validate_bundle_directory(source_path)
        call_ref = manifest.call_ref

        call_id, _created = ingest_call(call_ref)

        if manifest.transcript is not None:
            transcript_file = _safe_join(source_path, manifest.transcript.path)
            transcript_payload = _load_transcript_file(transcript_file)
            options = manifest.transcript.options or ChunkingOptions()
            ingest_transcript(call_ref, transcript_payload.content, options)

        if manifest.analysis:
            artifacts: List[AnalysisArtifactIn] = []
            for analysis_ref in manifest.analysis:
                analysis_file = _safe_join(source_path, analysis_ref.path)
                content = analysis_file.read_text(encoding="utf-8").strip()
                artifacts.append(
                    AnalysisArtifactIn(
                        kind=analysis_ref.kind,
                        content=content,
                        metadata=analysis_ref.metadata,
                    )
                )
            ingest_analysis(call_ref, artifacts)

        update_ingest_job_status(
            job_uuid,
            STATUS_SUCCEEDED,
            call_id=call_id,
            completed=True,
            error=None,
        )
        done_path = _move_bundle(validated.bundle_path, paths["done"])
        return {
            "ingest_job_id": ingest_job_id,
            "status": STATUS_SUCCEEDED,
            "call_id": str(call_id),
            "done_path": str(done_path),
        }
    except Exception as exc:
        update_ingest_job_status(
            job_uuid,
            STATUS_FAILED,
            error=str(exc),
            completed=True,
        )
        if source_path.exists():
            _move_bundle(source_path, paths["failed"])
        raise
