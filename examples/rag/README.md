# Offline RAG integration fixtures

The baseline is a real TF-IDF/extractive export from RAG Knowledge Assistant's
public synthetic corpus. `questions.jsonl` keeps two existing questions and adds
a paraphrased evaluation question that TF-IDF misses at rank one. Source filename
IDs are used for metrics; chunk IDs remain document provenance. No private data
or live model calls are involved.

From the RAG repository (with its dependencies installed), regenerate:

```bash
python -m src.capture --questions ../ai-evaluation-harness/examples/rag/questions.jsonl --run-id rag-tfidf-baseline --top-k 1 --revision YOUR_RAG_COMMIT --output ../ai-evaluation-harness/examples/rag/baseline.json
```

Then from the harness repository:

```bash
python examples/rag/make_candidate.py
ai-eval run --capture examples/rag/baseline.json --config examples/rag/suite.toml --report scratch/baseline.json
ai-eval run --capture examples/rag/candidate.json --config examples/rag/suite.toml --report scratch/candidate.json
ai-eval compare --baseline scratch/baseline.json --candidate scratch/candidate.json --config examples/rag/regression-pass.toml --report scratch/comparison.json
ai-eval compare --baseline scratch/baseline.json --candidate scratch/candidate.json --config examples/rag/regression-fail.toml --report scratch/regression-failed.json
```

The last command deliberately returns 1. Baseline Recall@1 is 2/3. Candidate
retrieval improves on rag-1, regresses on rag-2, and stays unchanged on rag-3, so
aggregate recall stays 2/3. Citation F1 drops from 2/3 to 1/3: the improved retrieval
retains a stale citation, while the regressed case loses its citation. The tolerant
gate permits a 0.4 drop; zero tolerance fails. Both absolute Recall@1 gates pass.
Candidate edits are synthetic fault injection into captured outputs, not evidence
of a different RAG implementation or model. Stable generation requires the same
corpus/dependencies; floats can differ across dependency versions, so fixtures
are committed artifacts rather than cross-version retrieval benchmarks.
