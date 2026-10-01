# Evaluation methodology

Capture question, generated answer, ordered retrieved documents, source IDs, citations, and available reference annotations. Distinguish unannotated fields (null) from intentionally empty annotations. Validate the entire dataset before execution, and use stable case/document identifiers.

Evaluate retrieval, reference answer overlap, citation attribution, and faithfulness separately. Inspect case-level explanations, score coverage, and errors alongside averages. A high lexical score does not establish factual correctness. Human reference labels can also be incomplete or wrong.

The synthetic fixture includes correct retrieval/answer/citation, irrelevant retrieval with invented citation and unsupported answer, missing evidence, duplicate rankings, and an explicitly empty relevance set. It is an infrastructure regression fixture, not evidence of real-world RAG quality.

Pin dataset bytes and configuration when comparing runs. Reports record checksums and prompt versions. Fixed quality thresholds in CI detect changes crossing the accepted limits. Phase 1 does not implement a statistical baseline-diff engine or significance testing. For model judgments, calibrate against human reviews, measure repeatability, and compare identical model/prompt versions before attributing a change to the evaluated system.
