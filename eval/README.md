# Evaluation Harness (Phase 0.5)

This is a lightweight, dependency-free evaluation harness for retrieval metrics.

## Files
- `gold_sample.jsonl`: sample gold set format (replace with your own)
- `run_eval.py`: computes recall@k, MRR, nDCG@k from gold + retrieval results

## Gold format (JSONL)
Each line:
```json
{"query_id":"q1","relevant_ids":["chunk:123","artifact:456"]}
```

## Results format (JSONL)
Each line:
```json
{"query_id":"q1","retrieved_ids":["chunk:123","chunk:999","artifact:456"]}
```

## Run
```bash
python eval/run_eval.py --gold eval/gold_sample.jsonl --results path/to/results.jsonl --k 5 10 20
```
