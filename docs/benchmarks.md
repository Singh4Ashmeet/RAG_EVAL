# Metrics and benchmarks

RAG_EVAL has two distinct kinds of measurement: evaluation of a RAG pipeline against labeled questions, and local inference timing for a model. Neither should be mistaken for the other.

## RAG evaluation metrics

The deterministic evaluation runner currently supports these measures:

| Metric | Meaning | Required ground truth |
| --- | --- | --- |
| Recall@k | Relevant chunk IDs retrieved in the first k ranks divided by all relevant chunk IDs. | `relevant_chunk_ids` |
| Precision@k | Unique relevant results in the first k ranks divided by k. Missing result slots count as non-relevant. | `relevant_chunk_ids` |
| MRR@k | Reciprocal rank of the first relevant result within k, or zero if none is found. | `relevant_chunk_ids` |
| AP@k / MAP@k | Precision at each relevant rank through k, normalized by `min(number of relevant chunks, k)`; MAP is the macro average across eligible queries. | `relevant_chunk_ids` |
| HitRate@k | 1 if at least one relevant chunk is found in the first k ranks, otherwise 0; aggregate is the macro average. | `relevant_chunk_ids` |
| Answer exact match | Case- and punctuation-insensitive token sequence equality after Unicode normalization; `[S#]` citation labels are ignored. | `reference_answer` and a generated answer |
| Answer token F1 | Multiset token-overlap F1 after the same normalization. | `reference_answer` and a generated answer |

Metrics are averaged over queries with the required labels. The JSON report's `metric_counts` gives the denominator for every aggregate. Token F1 and exact match are lexical comparisons; they do not establish semantic equivalence or answer faithfulness. No LLM judge or claim-level support metric is currently implemented.

The evaluation dataset digest is SHA-256 over exact JSONL bytes, including line endings. A changed ordering or newline convention therefore changes the digest. See the [dataset and evaluation runbook](runbook.md#evaluate-against-a-jsonl-dataset).

## Comparing pipeline variants

Experiments evaluate named runtime variants against the same dataset and cutoffs. Comparisons pair per-query values by query ID and report candidate-minus-baseline mean deltas with percentile bootstrap confidence intervals. The seed, number of resamples, and confidence level are configurable and stored with the experiment.

- Positive retrieval/answer deltas mean a higher metric. Positive latency deltas mean the candidate was slower.
- An interval is query-sample uncertainty; it does not capture run-to-run model or hardware variability.
- No threshold or statistical test declares a variant improved. Review effect size, interval, latency, and individual cases.
- The caller-supplied variant config is descriptive metadata and must match the runtime components.
- Per-query retrieval and generation duration are recorded. Memory/resource use is not part of experiment reports.

The golden dataset is empty pending reviewed project ground truth. Synthetic fixtures are implementation tests and must not be represented as RAG scores.

## Failure diagnoses

The diagnosis layer groups recorded retrieval errors, zero or partial recall, generation errors, unavailable answer metrics, low token overlap, and unresolved citation labels. Findings identify affected query IDs and recorded evidence. The default low-token-F1 threshold is `0.5`; it is configurable and used only to triage answers for review. Citation resolution checks whether a source label maps to a retrieved chunk, not whether the chunk supports the statement.

Recommendations are prompts for review or controlled experiments. They are not proof that a change will improve a system.

## Historical local model inference sample

The benchmark artifacts under [`docs/benchmarks/raw`](benchmarks/raw/) record local Ollama model inference, not retrieval or end-to-end RAG quality. The October 29, 2026 sample below used one warm generation run for `llama3.2:3b` and a 614-token prompt on Windows 11 with an AMD64 Family 25 Model 68 CPU, 8 physical/16 logical cores, 13.3 GB RAM, and Ollama 0.40.13.

| Measure | Recorded value |
| --- | ---: |
| Model file size | 2.4 GB |
| Cold time to first token | 18.45 s |
| Warm time to first token | 12.83 s |
| Warm prompt processing | 57.7 tokens/s |
| Warm generation | 11.0 tokens/s |
| Peak Ollama process memory | 2.6 GB |
| Minimum available system memory | 1.4 GB |

These values are a single machine-specific sample, not a comparative study or a current performance guarantee. The raw record is [llm_20261029_193510.json](benchmarks/raw/llm_20261029_193510.json). Re-run the benchmark script on the target machine before using timing or resource numbers for capacity decisions.
