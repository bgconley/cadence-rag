# Evaluation Harness (Phase 0.5)

This is a lightweight, dependency-free evaluation harness for retrieval metrics.

## Files
- `gold_sample.jsonl`: sample gold set format (replace with your own)
- `run_eval.py`: computes recall@k, MRR, nDCG@k from gold + retrieval results

## Gold format (JSONL)
Each line:
```json
{"query_id":"q1","relevant_ids":["chunk:123","artifact_chunk:456"]}
```

## Results format (JSONL)
Each line:
```json
{"query_id":"q1","retrieved_ids":["chunk:123","chunk:999","artifact_chunk:456"]}
```

## Run
```bash
python eval/run_eval.py --gold eval/gold_sample.jsonl --results path/to/results.jsonl --k 5 10 20
```

## Regression gate
Fail the command when metrics fall below thresholds:
```bash
python eval/regression_gate.py \
  --gold eval/gold_sample.jsonl \
  --results path/to/results.jsonl \
  --k 5 10 20 \
  --min-recall-at 20 --min-recall 0.60 \
  --min-mrr 0.40 \
  --min-ndcg-at 10 --min-ndcg 0.50
```

## Real end-to-end gate (P620 DB schema sandbox)
Creates an isolated schema, runs migrations, ingests realistic fixtures, runs real `/retrieve` logic (`ids_only`), computes metrics, and drops the schema.

```bash
python eval/run_real_regression_gate.py \
  --database-url postgresql+psycopg://rag:rag@10.25.0.50:5432/rag \
  --min-recall-at 20 --min-recall 0.80 \
  --min-mrr 0.60 \
  --min-ndcg-at 10 --min-ndcg 0.70
```

Useful options:
- `--keep-schema`: keep the temporary schema for debugging.
- `--gold-out / --results-out`: change output JSONL paths.
