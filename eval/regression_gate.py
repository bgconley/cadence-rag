from __future__ import annotations

import argparse
import json
import sys

from run_eval import compute_metrics, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail when retrieval metrics fall below configured thresholds."
    )
    parser.add_argument("--gold", required=True, help="Gold set JSONL")
    parser.add_argument("--results", required=True, help="Results JSONL")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--min-mrr", type=float, default=0.0)
    parser.add_argument("--min-recall-at", type=int, default=20)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--min-ndcg-at", type=int, default=10)
    parser.add_argument("--min-ndcg", type=float, default=0.0)
    args = parser.parse_args()

    ks = sorted(set(args.k + [args.min_recall_at, args.min_ndcg_at]))
    gold_rows = load_jsonl(args.gold)
    result_rows = load_jsonl(args.results)

    gold = {row["query_id"]: row.get("relevant_ids", []) for row in gold_rows}
    results = {
        row["query_id"]: row.get("retrieved_ids", row.get("retrieved", []))
        for row in result_rows
    }
    metrics = compute_metrics(gold, results, ks)

    recall_key = f"recall@{args.min_recall_at}"
    ndcg_key = f"ndcg@{args.min_ndcg_at}"
    failures = []

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

    print(json.dumps(metrics, indent=2))
    if failures:
        print("[regression_gate] FAIL")
        for failure in failures:
            print(f" - {failure}")
        raise SystemExit(1)

    print("[regression_gate] PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
