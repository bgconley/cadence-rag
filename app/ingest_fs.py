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
from rq import Queue, Retry
from sqlalchemy import text

from .config import settings
from .db import engine
from .embedding_pipeline import run_embedding_backfill
from .embeddings import EmbeddingClientError, embeddings_enabled
from .ingest_adapters import load_analysis_content, load_transcript_payload
from .ingest import ingest_analysis, ingest_call, ingest_transcript
from .logging_utils import get_logger
from .schemas import AnalysisArtifactIn, CallRef, ChunkingOptions, TranscriptPayload

INGEST_JOB_STATUS = Literal["queued", "running", "succeeded", "failed", "invalid"]
STATUS_QUEUED: INGEST_JOB_STATUS = "queued"
STATUS_RUNNING: INGEST_JOB_STATUS = "running"
STATUS_SUCCEEDED: INGEST_JOB_STATUS = "succeeded"
STATUS_FAILED: INGEST_JOB_STATUS = "failed"
STATUS_INVALID: INGEST_JOB_STATUS = "invalid"

BUNDLE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]{1,120}$")
MANIFEST_FILENAME = "manifest.json"
READY_FILENAME = "_READY"
TRANSCRIPT_EXTS = {".json", ".md", ".markdown", ".txt"}
ANALYSIS_EXTS = {
    ".md",
    ".markdown",
    ".txt",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".html",
    ".htm",
    ".docx",
    ".pdf",
}
DIRECT_INBOX_FILE_EXTS = TRANSCRIPT_EXTS | ANALYSIS_EXTS
INCOMPLETE_FILE_SUFFIXES = (".part", ".partial", ".tmp", ".download")
logger = get_logger(__name__)


class TranscriptFileRef(BaseModel):
    path: str = "transcript.json"
    format: Literal["json_turns", "markdown_turns", "auto"] = "json_turns"
    sha256: Optional[str] = None
    options: Optional[ChunkingOptions] = None


class AnalysisFileRef(BaseModel):
    kind: str
    path: str
    format: Literal[
        "auto", "text", "markdown", "csv", "tsv", "json", "html", "docx", "pdf"
    ] = "auto"
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
    manifest_path = bundle_path / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise ValueError(f"{MANIFEST_FILENAME} is required")
    manifest = _load_manifest(manifest_path)
    bundle_id = _normalize_bundle_id(bundle_path, manifest)
    if manifest.transcript is None and not manifest.analysis:
        raise ValueError("manifest must include transcript and/or analysis entries")

    files: List[BundleFileRecord] = []
    files.append(
        _validate_file_ref(
            bundle_path=bundle_path,
            kind="manifest",
            relative_path=MANIFEST_FILENAME,
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


def _infer_transcript_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return "markdown_turns"
    return "auto"


def _infer_analysis_format(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".log": "text",
        ".csv": "csv",
        ".tsv": "tsv",
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".docx": "docx",
        ".pdf": "pdf",
    }
    return mapping.get(suffix, "auto")


def _infer_analysis_kind(path: Path) -> str:
    stem = path.stem.lower()
    if "action" in stem or "todo" in stem or "next_step" in stem:
        return "action_items"
    if "decision" in stem:
        return "decisions"
    if "note" in stem or "tech" in stem:
        return "tech_notes"
    return "summary"


def _list_bundle_files(bundle_path: Path) -> List[Path]:
    files = [
        path
        for path in bundle_path.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]
    files.sort(key=lambda p: str(p.relative_to(bundle_path)).lower())
    return files


def _supports_direct_inbox_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in {MANIFEST_FILENAME, READY_FILENAME}:
        return False
    lower_name = path.name.lower()
    if lower_name.endswith(INCOMPLETE_FILE_SUFFIXES):
        return False
    return path.suffix.lower() in DIRECT_INBOX_FILE_EXTS


def _is_direct_inbox_file_ready(path: Path) -> bool:
    if not _supports_direct_inbox_file(path):
        return False
    age_s = time.time() - path.stat().st_mtime
    return age_s >= max(0, int(settings.ingest_single_file_min_age_s))


def _sanitize_bundle_seed(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("._-")
    if not cleaned:
        cleaned = "bundle"
    return cleaned[:100]


def _build_bundle_id_from_file(path: Path) -> str:
    seed = _sanitize_bundle_seed(path.stem)
    digest = _sha256_file(path)[:12]
    bundle_id = f"{seed}-{digest}"
    if not BUNDLE_ID_RE.fullmatch(bundle_id):
        bundle_id = _sanitize_bundle_seed(bundle_id)
    return bundle_id


def _move_file(src: Path, dest_root: Path) -> Path:
    dest_root.mkdir(parents=True, exist_ok=True)
    target = dest_root / src.name
    if target.exists():
        target = dest_root / f"{src.stem}-{int(time.time())}{src.suffix}"
    shutil.move(str(src), str(target))
    return target.resolve()


def _wrap_single_file_candidate(path: Path, processing_root: Path) -> Path:
    bundle_id = _build_bundle_id_from_file(path)
    bundle_path = processing_root / bundle_id
    if bundle_path.exists():
        bundle_path = processing_root / f"{bundle_id}-{int(time.time())}"
    bundle_path.mkdir(parents=True, exist_ok=False)
    _move_file(path, bundle_path)
    return bundle_path.resolve()


def _pick_transcript_candidate(bundle_path: Path, files: Sequence[Path]) -> Optional[Path]:
    candidates = [
        path
        for path in files
        if path.name not in {MANIFEST_FILENAME, READY_FILENAME}
        and path.suffix.lower() in TRANSCRIPT_EXTS
    ]
    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, str]:
        rel = str(path.relative_to(bundle_path)).lower()
        score = 100
        if "transcript" in rel:
            score -= 80
        if "call" in rel:
            score -= 10
        if path.suffix.lower() == ".json":
            score -= 10
        return score, rel

    return min(candidates, key=rank)


def _title_from_bundle_id(bundle_id: str) -> str:
    words = re.sub(r"[_\-]+", " ", bundle_id).strip().split()
    if not words:
        return bundle_id
    return " ".join(word.capitalize() for word in words)


def _build_auto_manifest(bundle_path: Path) -> BundleManifest:
    files = _list_bundle_files(bundle_path)
    transcript_path = _pick_transcript_candidate(bundle_path, files)

    analysis_refs: List[AnalysisFileRef] = []
    for path in files:
        if path.name in {MANIFEST_FILENAME, READY_FILENAME}:
            continue
        if transcript_path is not None and path == transcript_path:
            continue
        if path.suffix.lower() not in ANALYSIS_EXTS:
            continue

        rel = str(path.relative_to(bundle_path))
        analysis_refs.append(
            AnalysisFileRef(
                kind=_infer_analysis_kind(path),
                path=rel,
                format=_infer_analysis_format(path),
            )
        )

    if transcript_path is None and not analysis_refs:
        raise ValueError("manifest missing and no transcript/analysis files detected")

    bundle_id = bundle_path.name
    if not BUNDLE_ID_RE.fullmatch(bundle_id):
        bundle_id = _sanitize_bundle_seed(bundle_id)
    transcript_ref = None
    if transcript_path is not None:
        transcript_ref = TranscriptFileRef(
            path=str(transcript_path.relative_to(bundle_path)),
            format=_infer_transcript_format(transcript_path),
        )

    manifest = BundleManifest(
        bundle_id=bundle_id,
        call_ref=CallRef(
            external_source="filesystem",
            external_id=bundle_id,
            title=_title_from_bundle_id(bundle_id),
        ),
        transcript=transcript_ref,
        analysis=analysis_refs,
    )
    return manifest


def _ensure_manifest(bundle_path: Path) -> Path:
    manifest_path = bundle_path / MANIFEST_FILENAME
    if manifest_path.exists():
        return manifest_path
    if not settings.ingest_auto_manifest:
        raise ValueError(f"{MANIFEST_FILENAME} is required")

    manifest = _build_auto_manifest(bundle_path)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json", exclude_none=True), indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "ingest_bundle.manifest_generated bundle_id=%s manifest_path=%s",
        manifest.bundle_id or bundle_path.name,
        manifest_path,
    )
    return manifest_path


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
    max_attempts = max(1, int(settings.ingest_job_max_attempts))
    retry = _build_retry_policy(max_attempts, settings.ingest_job_retry_backoff_s)
    rq_job = queue.enqueue(
        "app.ingest_fs.process_ingest_job",
        str(ingest_job_id),
        job_id=str(ingest_job_id),
        retry=retry,
    )
    logger.info(
        "ingest_job.enqueued ingest_job_id=%s queue=%s max_attempts=%s",
        ingest_job_id,
        settings.ingest_queue_name,
        max_attempts,
    )
    return rq_job.id


def _build_retry_policy(max_attempts: int, base_backoff_s: int) -> Optional[Retry]:
    normalized_attempts = max(1, int(max_attempts))
    max_retries = max(0, normalized_attempts - 1)
    if max_retries == 0:
        return None
    base = max(1, int(base_backoff_s))
    intervals = [base * (2 ** idx) for idx in range(max_retries)]
    return Retry(max=max_retries, interval=intervals)


def _record_invalid_bundle(bundle_path: Path, error: str) -> None:
    bundle_id = bundle_path.name
    manifest_path = bundle_path / MANIFEST_FILENAME
    _create_or_get_job(
        bundle_id=bundle_id,
        source_path=bundle_path,
        manifest_path=manifest_path,
        call_ref={},
        status=STATUS_INVALID,
        error=error,
    )


def _record_invalid_path(path: Path, error: str) -> None:
    if path.is_dir():
        _record_invalid_bundle(path, error)
        return

    bundle_id = _sanitize_bundle_seed(path.stem)
    manifest_path = path.parent / f"{path.name}.manifest.json"
    _create_or_get_job(
        bundle_id=bundle_id,
        source_path=path,
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
        candidate_is_bundle_dir = candidate.is_dir() and (candidate / READY_FILENAME).exists()
        candidate_is_single_file = candidate.is_file() and _is_direct_inbox_file_ready(candidate)
        if not candidate_is_bundle_dir and not candidate_is_single_file:
            continue
        discovered += 1

        processing_path: Optional[Path] = None
        try:
            if candidate_is_bundle_dir:
                _ensure_manifest(candidate)
                validated = validate_bundle_directory(candidate)
                processing_path = _move_bundle(candidate, paths["processing"])
            else:
                processing_path = _wrap_single_file_candidate(
                    candidate, paths["processing"]
                )
                _ensure_manifest(processing_path)
                validated = validate_bundle_directory(processing_path)
        except Exception as exc:
            invalid += 1
            logger.warning(
                "ingest_bundle.invalid path=%s error=%s", candidate, str(exc)
            )
            if processing_path and processing_path.exists():
                _record_invalid_path(processing_path, str(exc))
                _move_bundle(processing_path, paths["failed"])
            elif candidate.exists():
                _record_invalid_path(candidate, str(exc))
                if candidate.is_dir():
                    _move_bundle(candidate, paths["failed"])
                elif candidate.is_file():
                    _move_file(candidate, paths["failed"])
            continue

        processing_manifest_path = processing_path / MANIFEST_FILENAME
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
            logger.warning(
                "ingest_bundle.duplicate bundle_id=%s", validated.bundle_id
            )
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
        logger.info(
            "ingest_bundle.queued bundle_id=%s ingest_job_id=%s files=%s",
            validated.bundle_id,
            job_id,
            len(rel_files),
        )

    return {
        "discovered": discovered,
        "queued": queued,
        "duplicates": duplicates,
        "invalid": invalid,
    }


def _load_transcript_file(path: Path, *, format_hint: str) -> TranscriptPayload:
    return load_transcript_payload(path, format_hint=format_hint)


def _auto_embed_call_if_configured(call_id: UUID) -> Dict[str, Any]:
    if not settings.ingest_auto_embed_on_success:
        return {"status": "skipped", "reason": "disabled"}
    if not embeddings_enabled():
        return {"status": "skipped", "reason": "embeddings_not_configured"}

    try:
        summary = run_embedding_backfill(
            batch_size=max(1, int(settings.embeddings_batch_size)),
            call_id=call_id,
            source="ingest_auto_embed",
        )
    except EmbeddingClientError as exc:
        if settings.ingest_auto_embed_fail_on_error:
            raise
        return {"status": "error", "error": str(exc)}
    except Exception as exc:
        if settings.ingest_auto_embed_fail_on_error:
            raise
        logger.exception("ingest_job.auto_embed_failed call_id=%s error=%s", call_id, str(exc))
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "rows_updated": summary.rows_updated,
        "calls_touched": summary.calls_touched,
        "model_used": summary.model_used,
        "ingestion_runs_inserted": summary.ingestion_runs_inserted,
    }


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
    attempt_no = int(job["attempts"]) + 1
    max_attempts = max(1, int(settings.ingest_job_max_attempts))
    logger.info(
        "ingest_job.start ingest_job_id=%s attempt=%s max_attempts=%s",
        job_uuid,
        attempt_no,
        max_attempts,
    )

    try:
        manifest = _load_manifest(manifest_path)
        validated = validate_bundle_directory(source_path)
        call_ref = manifest.call_ref

        call_id, _created = ingest_call(call_ref)

        if manifest.transcript is not None:
            transcript_file = _safe_join(source_path, manifest.transcript.path)
            transcript_payload = _load_transcript_file(
                transcript_file, format_hint=manifest.transcript.format
            )
            options = manifest.transcript.options or ChunkingOptions()
            ingest_transcript(call_ref, transcript_payload.content, options)

        if manifest.analysis:
            artifacts: List[AnalysisArtifactIn] = []
            for analysis_ref in manifest.analysis:
                analysis_file = _safe_join(source_path, analysis_ref.path)
                content = load_analysis_content(
                    analysis_file, format_hint=analysis_ref.format
                ).strip()
                artifacts.append(
                    AnalysisArtifactIn(
                        kind=analysis_ref.kind,
                        content=content,
                        metadata=analysis_ref.metadata,
                    )
                )
            ingest_analysis(call_ref, artifacts)

        embed_result = _auto_embed_call_if_configured(call_id)
        if embed_result.get("status") == "ok":
            logger.info(
                "ingest_job.auto_embed_complete ingest_job_id=%s call_id=%s rows_updated=%s model=%s",
                ingest_job_id,
                call_id,
                embed_result.get("rows_updated"),
                embed_result.get("model_used"),
            )
        elif embed_result.get("status") == "error":
            logger.warning(
                "ingest_job.auto_embed_error ingest_job_id=%s call_id=%s error=%s",
                ingest_job_id,
                call_id,
                embed_result.get("error"),
            )

        update_ingest_job_status(
            job_uuid,
            STATUS_SUCCEEDED,
            call_id=call_id,
            completed=True,
            error=None,
        )
        done_path = _move_bundle(validated.bundle_path, paths["done"])
        logger.info(
            "ingest_job.complete ingest_job_id=%s status=%s call_id=%s done_path=%s",
            ingest_job_id,
            STATUS_SUCCEEDED,
            call_id,
            done_path,
        )
        return {
            "ingest_job_id": ingest_job_id,
            "status": STATUS_SUCCEEDED,
            "call_id": str(call_id),
            "done_path": str(done_path),
            "embedding": embed_result,
        }
    except Exception as exc:
        error = str(exc)
        if attempt_no >= max_attempts:
            update_ingest_job_status(
                job_uuid,
                STATUS_FAILED,
                error=error,
                completed=True,
            )
            if source_path.exists():
                _move_bundle(source_path, paths["failed"])
            logger.exception(
                "ingest_job.failed ingest_job_id=%s attempt=%s error=%s",
                ingest_job_id,
                attempt_no,
                error,
            )
        else:
            update_ingest_job_status(
                job_uuid,
                STATUS_QUEUED,
                error=error,
                completed=False,
            )
            logger.warning(
                "ingest_job.retry_scheduled ingest_job_id=%s attempt=%s error=%s",
                ingest_job_id,
                attempt_no,
                error,
            )
        raise
