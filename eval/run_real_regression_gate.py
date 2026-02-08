from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence
from urllib.parse import quote
from uuid import UUID, uuid4

import psycopg
from alembic import command
from alembic.config import Config
from sqlalchemy import text


DEFAULT_BASE_URL = "postgresql+psycopg://rag:rag@10.25.0.50:5432/rag"


def _admin_url(url: str) -> str:
    if url.startswith("postgresql+psycopg://"):
        url = "postgresql://" + url.split("postgresql+psycopg://", 1)[1]
    return url.split("?", 1)[0]


def _schema_url(base_url: str, schema: str) -> str:
    sep = "&" if "?" in base_url else "?"
    options = quote(f"-c search_path={schema},public")
    return f"{base_url}{sep}options={options}"


def _dcg(relevances: Sequence[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel > 0:
            score += rel / math.log2(idx + 1)
    return score


def _compute_metrics(
    gold: Dict[str, List[str]], results: Dict[str, List[str]], ks: List[int]
) -> Dict[str, float]:
    totals = {f"recall@{k}": 0.0 for k in ks}
    totals["mrr"] = 0.0
    for k in ks:
        totals[f"ndcg@{k}"] = 0.0

    count = 0
    for query_id, relevant_ids in gold.items():
        retrieved = results.get(query_id, [])
        relevant_set = set(relevant_ids)
        if not relevant_ids:
            continue

        count += 1
        rr = 0.0
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                rr = 1.0 / idx
                break
        totals["mrr"] += rr

        for k in ks:
            topk = retrieved[:k]
            hit_count = sum(1 for doc_id in topk if doc_id in relevant_set)
            totals[f"recall@{k}"] += hit_count / max(len(relevant_ids), 1)
            relevances = [1 if doc_id in relevant_set else 0 for doc_id in topk]
            ideal = [1] * min(len(relevant_ids), k)
            totals[f"ndcg@{k}"] += _dcg(relevances) / (_dcg(ideal) or 1.0)

    if count == 0:
        return {key: 0.0 for key in totals}
    return {key: value / count for key, value in totals.items()}


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _lookup_single_id(conn, sql_query: str, params: Dict[str, object]) -> int:
    row = conn.execute(text(sql_query), params).fetchone()
    if not row:
        raise RuntimeError(f"expected row not found for params={params}")
    return int(row[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real retrieval regression gate in an isolated DB schema."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", DEFAULT_BASE_URL),
        help="Target database URL (default: env DATABASE_URL or P620 default).",
    )
    parser.add_argument(
        "--schema-prefix",
        default="rag_test_eval",
        help="Temporary schema prefix.",
    )
    parser.add_argument(
        "--keep-schema",
        action="store_true",
        help="Do not drop the temporary schema after the run.",
    )
    parser.add_argument(
        "--gold-out",
        default="/tmp/regression_real_gold.jsonl",
        help="Path for generated gold JSONL.",
    )
    parser.add_argument(
        "--results-out",
        default="/tmp/regression_real_results.jsonl",
        help="Path for generated results JSONL.",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--min-mrr", type=float, default=0.60)
    parser.add_argument("--min-recall-at", type=int, default=20)
    parser.add_argument("--min-recall", type=float, default=0.80)
    parser.add_argument("--min-ndcg-at", type=int, default=10)
    parser.add_argument("--min-ndcg", type=float, default=0.70)
    args = parser.parse_args()

    schema = f"{args.schema_prefix}_{uuid4().hex[:8]}"
    admin_url = _admin_url(args.database_url)
    schema_url = _schema_url(args.database_url, schema)

    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.execute(f"CREATE SCHEMA {schema}")

    try:
        os.environ["DATABASE_URL"] = schema_url
        os.environ["SKIP_VERSION_CHECK"] = "true"
        os.environ["EMBEDDINGS_BASE_URL"] = ""
        os.environ["ALEMBIC_VERSION_TABLE_SCHEMA"] = schema

        command.upgrade(Config("alembic.ini"), "head")

        import app.config as config_module
        import app.db as db_module
        import app.ingest as ingest_module
        import app.retrieve as retrieve_module
        import app.schemas as schemas_module

        importlib.reload(config_module)
        importlib.reload(db_module)
        importlib.reload(ingest_module)
        importlib.reload(retrieve_module)
        importlib.reload(schemas_module)

        from app.db import engine
        from app.ingest import ingest_analysis, ingest_transcript
        from app.retrieve import retrieve_evidence
        from app.schemas import (
            AnalysisArtifactIn,
            CallRef,
            ChunkingOptions,
            RetrieveRequest,
            UtteranceIn,
        )

        def ingest_fixture_call(
            external_id: str,
            utterances: List[UtteranceIn],
            artifacts: List[AnalysisArtifactIn],
        ) -> UUID:
            call_ref = CallRef(external_source="eval", external_id=external_id)
            call_id, _, _ = ingest_transcript(call_ref, utterances, ChunkingOptions())
            ingest_analysis(call_ref, artifacts)
            return call_id

        call1 = ingest_fixture_call(
            "eval-call-1",
            [
                UtteranceIn(
                    speaker="SE",
                    start_ts_ms=0,
                    end_ts_ms=1000,
                    text="We should finalize the BOM for the Lenovo build with SSD capacity details.",
                ),
                UtteranceIn(
                    speaker="AE",
                    start_ts_ms=1000,
                    end_ts_ms=2000,
                    text="Object store tiering and cost profile are key for this customer.",
                ),
            ],
            [
                AnalysisArtifactIn(
                    kind="action_items",
                    content="- Finalize BOM for Lenovo build by Friday.\n- Produce SSD sizing worksheet.",
                )
            ],
        )
        call2 = ingest_fixture_call(
            "eval-call-2",
            [
                UtteranceIn(
                    speaker="SE",
                    start_ts_ms=0,
                    end_ts_ms=1000,
                    text="Competitive bake-off is head-to-head versus incumbent AWS and Azure options.",
                ),
                UtteranceIn(
                    speaker="SE",
                    start_ts_ms=1000,
                    end_ts_ms=2000,
                    text="We need OCI and GCP comparison notes as well.",
                ),
            ],
            [
                AnalysisArtifactIn(
                    kind="decisions",
                    content="- Proceed with competitive bake-off.\n- Position against incumbent cloud footprint.",
                )
            ],
        )
        call3 = ingest_fixture_call(
            "eval-call-3",
            [
                UtteranceIn(
                    speaker="Engineer",
                    start_ts_ms=0,
                    end_ts_ms=1000,
                    text="Ticket ABC-123 was opened after ECONNRESET errors in api-gateway.",
                ),
                UtteranceIn(
                    speaker="Engineer",
                    start_ts_ms=1000,
                    end_ts_ms=2000,
                    text="Roll back build v1.2.3 if ECONNRESET persists.",
                ),
            ],
            [
                AnalysisArtifactIn(
                    kind="summary",
                    content="ECONNRESET issue tracked under ABC-123 with rollback contingency.",
                )
            ],
        )

        with engine.connect() as conn:
            q1_chunk = _lookup_single_id(
                conn,
                """
                SELECT chunk_id FROM chunks
                WHERE call_id = :call_id AND text ILIKE :pattern
                ORDER BY chunk_id ASC LIMIT 1
                """,
                {"call_id": call1, "pattern": "%BOM for the Lenovo build%"},
            )
            q1_art = _lookup_single_id(
                conn,
                """
                SELECT artifact_chunk_id FROM artifact_chunks
                WHERE call_id = :call_id AND content ILIKE :pattern
                ORDER BY artifact_chunk_id ASC LIMIT 1
                """,
                {"call_id": call1, "pattern": "%Finalize BOM for Lenovo build%"},
            )
            q2_chunk = _lookup_single_id(
                conn,
                """
                SELECT chunk_id FROM chunks
                WHERE call_id = :call_id AND text ILIKE :pattern
                ORDER BY chunk_id ASC LIMIT 1
                """,
                {"call_id": call2, "pattern": "%Competitive bake-off%"},
            )
            q2_art = _lookup_single_id(
                conn,
                """
                SELECT artifact_chunk_id FROM artifact_chunks
                WHERE call_id = :call_id AND content ILIKE :pattern
                ORDER BY artifact_chunk_id ASC LIMIT 1
                """,
                {"call_id": call2, "pattern": "%Proceed with competitive bake-off%"},
            )
            q3_chunk = _lookup_single_id(
                conn,
                """
                SELECT chunk_id FROM chunks
                WHERE call_id = :call_id AND text ILIKE :pattern
                ORDER BY chunk_id ASC LIMIT 1
                """,
                {"call_id": call3, "pattern": "%Ticket ABC-123%"},
            )
            q3_art = _lookup_single_id(
                conn,
                """
                SELECT artifact_chunk_id FROM artifact_chunks
                WHERE call_id = :call_id AND content ILIKE :pattern
                ORDER BY artifact_chunk_id ASC LIMIT 1
                """,
                {"call_id": call3, "pattern": "%ABC-123%"},
            )

        gold_rows = [
            {
                "query_id": "q1",
                "query": "What did we commit to for the Lenovo BOM build?",
                "relevant_ids": [f"chunk:{q1_chunk}", f"artifact_chunk:{q1_art}"],
            },
            {
                "query_id": "q2",
                "query": "What was the competitive bake-off decision versus incumbent cloud vendors?",
                "relevant_ids": [f"chunk:{q2_chunk}", f"artifact_chunk:{q2_art}"],
            },
            {
                "query_id": "q3",
                "query": "Which ticket tracked the ECONNRESET issue?",
                "relevant_ids": [f"chunk:{q3_chunk}", f"artifact_chunk:{q3_art}"],
            },
        ]

        result_rows = []
        for row in gold_rows:
            response = retrieve_evidence(
                RetrieveRequest(query=row["query"], return_style="ids_only")
            )
            result_rows.append(
                {
                    "query_id": row["query_id"],
                    "retrieved_ids": response["retrieved_ids"],
                }
            )
            print(
                f"{row['query_id']} relevant={row['relevant_ids']} "
                f"top5={response['retrieved_ids'][:5]}"
            )

        gold_out = Path(args.gold_out)
        results_out = Path(args.results_out)
        _write_jsonl(
            gold_out,
            [{"query_id": row["query_id"], "relevant_ids": row["relevant_ids"]} for row in gold_rows],
        )
        _write_jsonl(results_out, result_rows)
        print(f"GOLD={gold_out}")
        print(f"RESULTS={results_out}")

        ks = sorted(set(args.k + [args.min_recall_at, args.min_ndcg_at]))
        metrics = _compute_metrics(
            {row["query_id"]: row["relevant_ids"] for row in gold_rows},
            {row["query_id"]: row["retrieved_ids"] for row in result_rows},
            ks,
        )
        print(json.dumps(metrics, indent=2))

        failures = []
        recall_key = f"recall@{args.min_recall_at}"
        ndcg_key = f"ndcg@{args.min_ndcg_at}"
        if metrics.get("mrr", 0.0) < args.min_mrr:
            failures.append(f"mrr {metrics.get('mrr', 0.0):.4f} < {args.min_mrr:.4f}")
        if metrics.get(recall_key, 0.0) < args.min_recall:
            failures.append(
                f"{recall_key} {metrics.get(recall_key, 0.0):.4f} < {args.min_recall:.4f}"
            )
        if metrics.get(ndcg_key, 0.0) < args.min_ndcg:
            failures.append(
                f"{ndcg_key} {metrics.get(ndcg_key, 0.0):.4f} < {args.min_ndcg:.4f}"
            )

        if failures:
            print("[real_regression_gate] FAIL")
            for failure in failures:
                print(f" - {failure}")
            raise SystemExit(1)

        print("[real_regression_gate] PASS")
    finally:
        if args.keep_schema:
            print(f"SCHEMA_KEPT={schema}")
        else:
            with psycopg.connect(admin_url, autocommit=True) as conn:
                conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            print(f"SCHEMA_DROPPED={schema}")


if __name__ == "__main__":
    main()
