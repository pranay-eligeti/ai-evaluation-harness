# Metric definitions

IDs are exact and case-sensitive. Expected IDs are sets. Retrieval duplicates collapse at first occurrence **before** applying K: [A,A,B] at K=2 becomes [A,B]. This evaluates unique-document ranking; it intentionally does not penalize duplicate slots. K must be an integer >=1; booleans, floats, zero, and negatives are rejected even when annotations are empty. K may exceed available results.

Let R be the relevant ID set, T the first K unique retrieved IDs, and H=T intersect R.

- Recall@K = |H|/|R|. Empty R is undefined/skipped; empty retrieval with nonempty R scores zero.
- Precision@K with `retrieved` denominator = |H|/|T|. Empty T is undefined/skipped. Empty R with nonempty T scores zero.
- Precision@K with `k` denominator = |H|/K. Empty retrieval scores zero; short lists leave unfilled slots in the denominator.
- Reciprocal rank = 1/r for the first relevant unique-document rank r, or zero if none appears. Optional K limits the search. Empty R is undefined. MRR is the arithmetic mean of defined reciprocal ranks, not a per-case score.

Citation F1 compares expected E and actual C as sets. Precision = |E intersect C|/|C|, recall = |E intersect C|/|E|, F1 = 2PR/(P+R), zero when P+R is zero. An empty precision or recall denominator is defined as 1 by convention. Both sets empty therefore score F1=1; exactly one empty scores F1=0. Metadata includes matched, missing, and spurious IDs. Citation validity = |C intersect retrieved IDs|/|C|; no citations is undefined. Validity proves an ID resolves, not that the source supports a claim.

Answer normalization lowercases, replaces ASCII punctuation with spaces, removes English articles a/an/the, and collapses whitespace. Exact match compares normalized strings. Token F1 uses multiset token overlap: P=overlap/prediction count, R=overlap/reference count, then harmonic mean. Both normalized strings empty score 1; one empty scores zero. Unicode punctuation is not removed, and 10,000 differs from 10000. These English surface metrics cannot recognize semantic equivalents.

Lexical groundedness = distinct answer content tokens found in context / distinct answer content tokens. Explicit English stopwords are excluded. Missing context or zero content tokens is undefined. Negation can disappear as a stopword, so contradictory claims can score 1. This is only a vocabulary proxy, never semantic faithfulness.

At the evaluator layer, None annotations mean unannotated and skip. Explicit [] means annotated empty and follows each formula. Missing reference/generated answers skip; empty strings are recorded responses and follow answer matching conventions. Missing expected citations skip correctness; missing actual citations defaults to an empty list.

Aggregates are macro means over defined scores. Undefined means/pass rates remain null. Pass rate counts only thresholded passed/failed cases. Errors and skips contribute no fabricated zeros; their counts are available to gates. No combined overall score is calculated.
