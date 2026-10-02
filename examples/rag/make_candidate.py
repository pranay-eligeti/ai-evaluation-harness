"""Regenerate the controlled candidate fixture; never executes a RAG application."""

import copy
import json
from pathlib import Path
from typing import Any


def perturb(baseline: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(baseline)
    candidate["run_id"] = "rag-candidate-perturbed"
    candidate["revision"] = None
    candidate["metadata"]["fixture"] = (
        "Controlled captured-output perturbations; not a new retriever implementation"
    )
    cases = candidate["cases"]
    hit = next(
        c for c in cases if c["retrieved_documents"][0]["doc_id"] in c["expected_document_ids"]
    )
    miss = next(
        c for c in cases if c["retrieved_documents"][0]["doc_id"] not in c["expected_document_ids"]
    )
    hit["retrieved_documents"] = [
        {
            "doc_id": "synthetic-noise.md",
            "text": "Unrelated synthetic context.",
            "score": 0.0,
            "metadata": {"source_id": "synthetic-noise.md", "chunk_id": "noise-1"},
        }
    ]
    hit["actual_citations"] = []
    miss["retrieved_documents"] = [
        {
            "doc_id": miss["expected_document_ids"][0],
            "text": "Retrieval metrics check whether the correct context was found.",
            "score": 0.8,
            "metadata": {"source_id": miss["expected_document_ids"][0], "chunk_id": "controlled-1"},
        }
    ]
    # Keep the original answer/citations to exercise a stale citation despite better retrieval.
    return candidate


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    baseline = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
    (root / "candidate.json").write_text(
        json.dumps(perturb(baseline), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
