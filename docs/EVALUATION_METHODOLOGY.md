# Evaluation methodology

Capture question, generated answer, ordered retrieved documents, source IDs, citations, and available reference annotations. Distinguish unannotated fields (null) from intentionally empty annotations. Validate the entire dataset before execution, and use stable case/document identifiers.

Evaluate retrieval, reference answer overlap, citation attribution, and faithfulness separately. Inspect case-level explanations, score coverage, and errors alongside averages. A high lexical score does not establish factual correctness. Human reference labels can also be incomplete or wrong.

The synthetic fixture includes correct retrieval/answer/citation, irrelevant retrieval with invented citation and unsupported answer, missing evidence, duplicate rankings, and an explicitly empty relevance set. It is an infrastructure regression fixture, not evidence of real-world RAG quality.

Pin dataset bytes and configuration when comparing runs. Reports record checksums and prompt versions. Fixed quality thresholds in CI detect changes crossing the accepted limits. Phase 1 does not implement a statistical baseline-diff engine or significance testing. For model judgments, calibrate against human reviews, measure repeatability, and compare identical model/prompt versions before attributing a change to the evaluated system.

For Phase 2 semantic runs, select a schema-capable account-accessible provider model explicitly and consent to external content transfer with --allow-live. Use separate relevance and faithfulness scores: a relevant answer can be false, and a faithful answer can fail to address the question. Missing context is a coverage gap; supplied but insufficient context is evidence against claim support. Model rubric anchors are qualitative judgments mapped to [0,1], not calibrated probabilities.

Prompt version 2 treats questions/answers/documents as untrusted JSON data. Evaluate injection attempts and contradiction/paraphrase cases against human annotations before using gates in production. Synthetic scripted judgments demonstrate this execution path, not semantic accuracy. Bounded retries can create additional charges and stochastic differences; preserve attempts and actual model/prompt provenance when comparing runs. Do not interpret a fixed threshold crossing as statistically significant without separate analysis.

## Paired regression methodology

Compare the same questions and annotations, not merely similar aggregate values.
Exact populations are required by default. Partial mode recomputes aggregates on
the matched intersection with equal scored-case coverage. Added/removed cases are
reported separately, never silently absorbed. Judge criterion/provider/model/prompt
provenance must agree. Synthetic examples demonstrate mechanics, not statistical
significance or general RAG quality. See [comparison methodology](REGRESSION_COMPARISON.md).
