import argparse
import json
import math
from typing import Dict, List, Sequence


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dcg(relevances: Sequence[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel > 0:
            score += rel / math.log2(idx + 1)
    return score


def compute_metrics(
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

        # MRR
        rr = 0.0
        for idx, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                rr = 1.0 / idx
                break
        totals["mrr"] += rr

        # Recall and nDCG
        for k in ks:
            topk = retrieved[:k]
            hit_count = sum(1 for doc_id in topk if doc_id in relevant_set)
            totals[f"recall@{k}"] += hit_count / max(len(relevant_ids), 1)

            relevances = [1 if doc_id in relevant_set else 0 for doc_id in topk]
            ideal_rels = [1] * min(len(relevant_ids), k)
            ndcg_val = dcg(relevances) / (dcg(ideal_rels) or 1.0)
            totals[f"ndcg@{k}"] += ndcg_val

    if count == 0:
        return {key: 0.0 for key in totals}

    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval results.")
    parser.add_argument("--gold", required=True, help="Gold set JSONL")
    parser.add_argument("--results", required=True, help="Results JSONL")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()

    gold_rows = load_jsonl(args.gold)
    result_rows = load_jsonl(args.results)

    gold = {row["query_id"]: row.get("relevant_ids", []) for row in gold_rows}
    results = {
        row["query_id"]: row.get("retrieved_ids", row.get("retrieved", []))
        for row in result_rows
    }

    metrics = compute_metrics(gold, results, args.k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
