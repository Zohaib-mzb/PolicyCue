# Evaluation implementation notes

## PolicyCue Eval v1

The default `dataset.json` is **PolicyCue Eval v1 — controlled policy QA benchmark**,
version **1.0.0**: 50 examples, 38 answerable and 12 unanswerable, across six fixed
authored synthetic policies. Commerce, privacy, service terms, community guidelines,
AI use, and cookies/tracking each produce eight real production chunks (48 total).
Nine questions need multiple chunks. All searches retain the production owner/document
filter, so competition is among eight chunks of the chosen document, not across policies.
See [benchmark design and label audit](BENCHMARK.md) for distribution and limitations.

The original four-example `dataset-starter.json` is retained for infrastructure tests;
it is not the default benchmark. Human-readable fixture files are the editable source
snapshots, and `dataset.json` embeds identical text for the existing runner schema.
Tests enforce agreement between these two representations. No live source fetches are
used as ground truth. Production behavior is unchanged.

## Production audit

- `ingestion/chunker.py`: `chunk_text(str) -> list[str]`, recursive splitter,
  1200 characters, 200 overlap. No existing versioned, labeled policy benchmark was
  found; policy text in existing tests is inline and used for mocks.
- `retrieval/embeddings.py`: `create_document_embeddings` uses passage mode;
  `create_query_embedding` uses query mode. Both use the configured Pinecone
  Inference model (`llama-text-embed-v2`, default dimension 768).
- `retrieval/vector_store.py`: vector ID is `document_id-chunk_index`, with zero-based
  indexes. Metadata contains document/owner IDs, text, source, filename, and optional
  URL/title/category/provenance fields. Search returns ordered metadata plus scores.
- `search_chunks` and `/ask` default to top-k 5. Production requests filter on both
  document and owner. Evaluation passes both for every search.
- `analysis/answer_generator.py`: `generate_answer(question, list[dict]) -> str`.
  `AnswerDecision.answerable` is internal; malformed/empty/unsupported decisions all
  become `I could not find that information in the provided document.`
  Evaluation uses that exact constant because no structured answerability boolean is
  exposed by the production function; `document_found` is not answerability.
- Production `answer_question` reports all retrieved chunks when it answers and
  clears sources on abstention. `source_attributions` groups genuine source metadata.
  Evaluation calls the same search, generator and attribution functions once, retaining
  pre-abstention retrieval results for diagnosis. A parity test checks both answer and
  abstention results against `answer_question`. No ranking or generation is recreated.
- `core/config.py` uses cached Pydantic settings and the root `.env`. Evaluation reads
  the same settings; reports whitelist model configuration and never serialize settings
  or raw SDK exceptions. `services/` is empty; models define ingestion source/input types.
- Existing tests use `unittest.mock.patch` at SDK and pipeline boundaries. New tests
  follow that pattern and block socket connections. No production files were changed.

## Dataset contract and expansion

`dataset.json` contains name/version/description, a `synthetic` flag, documents, and
examples. Documents embed fixed UTF-8 source text and an ordered SHA-256 manifest of
**actual production chunks**. Keep source snapshots and labels under version control.
Each example has a unique ID, document fixture ID, question, nullable expected answer,
relevant chunk IDs, strict `should_answer` boolean, and optional notes/category.
Answerable examples require a nonblank reference and at least one evidence label;
unanswerable examples require null and an empty evidence set. Unknown, duplicate,
out-of-range and cross-document labels are rejected before any API calls.

Labels use `fixture-id:zero-based-index`; physical vectors use the production
`runtime-document-id-index` format. The report maps fixed fixture IDs to unique runtime
IDs and preserves raw metadata. This separates reproducible labels from temporary IDs.
The chunk hashes reject source/splitter drift before upload.

To expand to 50–100 examples, add fixed policy text snapshots, run the production
`chunk_text` function on each, record `hashlib.sha256(chunk.encode()).hexdigest()` in
order, and **manually inspect** each question/reference and all its supporting chunks.
Use `--validate-only` to inspect the manifest's actual chunks. Increment the benchmark
version for any source/label change. V1 was labeled after running the production
chunker and verifying the complete section boundaries. Each example records its evidence
rationale or why a plausible negative has no answer. This author review is not a claim
of independent human annotation; independently review labels before public claims. URL/PDF text can be frozen after the
existing production extraction steps; this runner measures RAG over fixed text, not
fetch/extraction quality or changing live websites. Schema size does not limit example
count, but service upload limits still apply to very large documents.

## Exact semantic integration

Pinned **RAGAS 0.4.3**, verified against its downloaded wheel and current
[PyPI release](https://pypi.org/project/ragas/0.4.3/). Uses
`ragas.metrics.collections.{Faithfulness, AnswerRelevancy, ContextPrecision}` with
`await metric.ascore(...)` and `MetricResult.value`, not deprecated metric singletons
or legacy dataset/evaluate examples. API signatures are covered by offline tests.

The public `InstructorLLM` wrapper uses
`instructor.from_genai(google.genai.Client(...), use_async=True)`. The 0.4.3
`llm_factory` Google path creates a synchronous wrapper, while collections metrics
require `agenerate`; explicit async construction is necessary and avoids adding
LiteLLM. One event loop spans the run and async client cleanup. Instructor 1.14.5 with its `google-genai` extra and native JSON structured-output mode is pinned because RAGAS leaves it unbounded and resolution
can select an obsolete version. `langchain-community==0.4.1` preserves the internal
VertexAI import still loaded by RAGAS 0.4.3 (removed in community 0.4.2). Production pins
are constraints, not upgraded dependencies. Upstream RAGAS includes OpenAI packages as
mandatory transitive dependencies; no OpenAI service, key, tracing backend or additional
infrastructure is used. RAGAS anonymous telemetry is disabled before import.

- **Faithfulness:** question, real response, and actual ordered retrieval contexts;
  only actual non-abstaining answers with contexts.
- **Answer Relevancy:** question and real response; same eligibility. Its supported
  `BaseRagasEmbedding` adapter calls production `create_query_embedding` for both
  original and judge-generated questions (same model and query mode for symmetry).
- **ContextPrecision:** question, human reference answer, and ordered contexts;
  only labeled answerable examples with contexts. This measures rank-sensitive context
  usefulness, not deterministic Precision@k and not the old ContextRelevancy formula.

Scores are serial. Each judge subrequest has a 90-second timeout, with evaluation-only
pacing and bounded retries as described below. Deliberate waiting is not charged against
a whole-metric timeout. Errors/non-finite values are
reported as null with status/error type, excluded from the mean, and make the run
partial (nonzero CLI exit). Ineligible and explicitly disabled metrics are distinct.
Faithfulness/relevancy sample counts describe actual answers, not all questions;
abstention accuracy captures refusal errors. Do not compare runs using only the judge
means while ignoring sample counts. Judge model aliases and service updates may still
change results despite temperature zero and fixed inputs.

## Deterministic conventions

Recall/Precision/RR are calculated in top-k; RR is zero for a miss. MRR is a macro mean.
No-evidence negative examples contribute to abstention and citation diagnostics, not
retrieval averages. Precision divides by k even for short responses; duplicate chunk
IDs cannot increase hits. Citation correctness is a per-example fraction over distinct
reported source chunks, macro-averaged only when at least one citation exists.
Supporting-citation coverage includes **all labeled answerable cases**, so missing
citations/false abstentions count as false. Missing citation count is the number of
labeled evidence chunks absent from reported sources; incorrect count is reported
chunks outside that set. Negative answers with reported citations score zero correctness.

These citation labels measure evidence relevance, not merely provenance. Production
reports all retrieved chunks, not selected inline claim citations. Therefore this
metric evaluates the reported evidence set; it cannot establish claim-level entailment
or that the generator actually relied on a particular chunk. Incomplete human evidence
labels can penalize otherwise valid citations and must be reviewed.

## Isolation, cleanup and reports

Each run creates an unpredictable `policycue-eval-<uuid>` owner and fresh document IDs
under it, in the existing index. It never accepts existing owner/document IDs as CLI
inputs. Every write is registered before upsert, including failed partial writes.
Cleanup uses the existing owner-and-document filtered deletion, with a guard requiring
both the exact current run owner and a registered document ID. A prefix alone is
insufficient. Cleanup attempts every registered document even if a prior delete fails.
There is no namespace-wide or index-wide deletion.

A bounded fetch check verifies uploaded text and ownership before retrieval. Pinecone
is eventually consistent: fetch visibility cannot guarantee immediate search visibility,
and cleanup is reported as `delete_requested`, not falsely claimed as verified deletion.
Normal exceptions trigger cleanup in `finally`; a killed process/network outage can
leave temporary vectors. Failed cleanup is prominent in JSON and causes nonzero exit.
Use only the exact owner/document pairs in that report for manual recovery; never use a
broad prefix deletion. Reports contain private benchmark text and generated answers,
so generated files are gitignored. Reports include failed/not-run counts; an ingestion
failure is not a zero-scoring benchmark.

Offline tests cover schemas, manifests, formulas, abstention, citations, null counts,
JSON aggregation, production parity, judge arguments, and cleanup on failures. Existing
backend tests perform index host resolution during import and four public-DNS checks;
in restricted environments, the former needs a temporary mocked Index constructor and
the latter require DNS access. This is existing test behavior, not evaluation API usage.

## First v1 run status

One authorised live attempt was made on 21 September 2026 (Asia/Karachi), with all
production settings unchanged. It finished with a **partial** status because of
pipeline and judge errors; a complete 50-example baseline has **not** been established.
The raw report, environment snapshot, detailed case analysis and exact-ID cleanup
verification are preserved in the gitignored `results/` directory. No scores from this
incomplete attempt are published in the main README or presented here as a successful
baseline. Its raw report is `evaluation-20260920T230146.459772Z.json`; the companion
`evaluation-20260920T230146.459772Z-analysis.md` distinguishes execution errors, judge
judgments, reference granularity and actual retrieval behavior. No automatic rerun or
production optimisation followed the attempt.

## Evaluation reliability (second attempt)

The first attempt recorded only 20 pipeline results and 32 semantic-call errors. Its
exception classes were insufficient to distinguish quota, capacity and malformed
responses. Those historical files remain unmodified; their cause cannot be proved
retroactively from the available metadata.

The evaluation-only reliability layer now provides:

- Explicit query-embedding, Pinecone retrieval, generation, attribution and individual
  RAGAS metric stages. Ordered contexts, raw scores and mapped fixture IDs survive a
  generation error. Retrieval aggregation uses every successfully retrieved example,
  even when generation failed. Answerability/citation aggregation requires the relevant
  actual output; formulas and eligibility rules are unchanged.
- Safe diagnostics: stage, exception class/module/chain, HTTP status, allowlisted
  provider status, numeric Retry-After, quota scope when explicitly supplied, and a
  reconstructed short message. Raw exception text, Instructor messages/completions,
  arbitrary provider payloads, request headers, cookies and settings are never copied
  into diagnostics. Unknown details remain unknown rather than being guessed.
- A shared sequential Gemini request clock. `--request-delay` spaces generation
  requests; `--judge-delay` spaces every internal RAGAS LLM request, including each
  generated question and context judgment. Thus a metric cannot hide a burst of calls.
  Pinecone embeddings for answer relevancy are also sequential.
- `--max-retries` bounds extra attempts (default 2, maximum 5). Only confirmed transient
  HTTP/provider failures or timeouts are retried. Backoff defaults to 60 then 120 seconds.
  HTTP numeric/date Retry-After and Google's structured RetryInfo are respected. If the
  server requests more than 600 seconds, the call fails rather than retrying too early.
  Explicit daily-quota exhaustion is recorded without a futile immediate retry.
  Invalid requests, authentication failures, structured-output validation failures and
  unknown exceptions are not retried by the evaluator.
- Instructor's supported `max_retries=1` disables its implicit multiple attempts. The
  judge SDK also uses one attempt; the evaluation policy controls transient retries.
  Native Gemini JSON structured outputs and all RAGAS metric definitions are unchanged.
- Within this standalone sequential CLI process, scoped context managers temporarily
  replace the production functions' **retry boundary only**. Their original embedding,
  search and generation functions still execute with the same arguments, ranking,
  prompts, model and fallback. Every binding is restored even on exceptions. The
  application never imports this package; do not embed the runner in a serving process
  or run simultaneous evaluations in one Python process.
- Atomic per-example checkpoint snapshots and independent stage aggregation. Checkpoints
  are diagnostic artifacts, not an implemented resume feature; nothing automatically
  recomputes or resumes completed examples. The final report records the full reliability
  configuration and recovered as well as exhausted request failures.
- Cleanup requests still use the exact registered owner/document guard. Final verification
  fetches only those documents' exact vector IDs with bounded eventual-consistency polling,
  writing an independent `*-cleanup.json` receipt. No broad deletion is performed.

The separately authorised second attempt uses the frozen v1.0.0 dataset/hash, k=5,
Gemini `gemini-3.1-flash-lite` for both generation and judging, and
`llama-text-embed-v2` with 768 dimensions. Its exact invocation is:

```bash
PYTHONPATH=. .venv-evaluation/bin/python -m backend.evaluation.evaluator \
  --dataset backend/evaluation/dataset.json --k 5 \
  --request-delay 6 --judge-delay 6 --max-retries 2
```

Six-second shared spacing targets at most ten Gemini request starts per minute before
network latency and backoff; it is a conservative runner setting, **not a claim about
this account's actual quota**. Production settings and the main README's score-free
policy remain unchanged. The report's `metric_coverage` separates known eligibility,
successful measurements, errors, unavailable measurements and unknown eligibility.

The first segment of this second attempt stopped after 23/50 completed cases (19
answers and four correct abstentions). Its checkpoint is
`results/checkpoint-20260921T173645.622636Z.json`. No pipeline or semantic failures
were recorded in those cases. Safe provider metadata shows intermittent HTTP 503
capacity responses and two timeouts; it does not establish quota exhaustion. The
segment's six exact temporary documents were deleted, and an independent receipt
`results/checkpoint-20260921T173645.622636Z-cleanup.json` confirms all 48 expected
vector IDs were absent. The checkpoint is **not a complete baseline**.

An evaluation-only `--resume-checkpoint` option accepts this specific kind of cleaned
checkpoint. It requires matching frozen dataset hash, k, fixture mapping, ordered
successful prior rows, and a successful exact-ID cleanup receipt. It re-uploads the
same fixed fixtures under the checkpoint's guarded owner/document IDs, skips completed
examples, and records execution segments in the resulting report. It does not
recompute the first 23 cases. Resume measured the remaining 27 cases using the same
models and pacing. The final raw report is
`results/evaluation-20260921T231144.256581Z.json` and its exact-ID cleanup receipt is
`results/evaluation-20260921T231144.256581Z-cleanup.json`. Both segments are disclosed
in the raw report. All 50 rows have a pipeline result; all 38 eligible cases have all
three RAGAS results; there are no exhausted judge errors. Six document deletion
requests succeeded and all 48 exact vector IDs were verified absent.

### Complete segmented v1 baseline

| Metric | Value | Eligible / scored | Error or unavailable |
| --- | ---: | ---: | ---: |
| Recall@5 | 1.0000 | 38 / 38 | 0 |
| Precision@5 | 0.2526 | 38 / 38 | 0 |
| MRR | 0.9868 | 38 / 38 | 0 |
| RAGAS Faithfulness | 0.9934 | 38 / 38 | 0 |
| RAGAS Answer Relevancy | 0.7184 | 38 / 38 | 0 |
| RAGAS Context Precision | 0.8349 | 38 / 38 | 0 |
| Citation Correctness | 0.2526 | 38 / 38 | 0 |
| Supporting Citation Coverage | 1.0000 | 38 / 38 | 0 |
| Abstention Accuracy | 1.0000 | 50 / 50 | 0 |
| Answerable Accuracy | 1.0000 | 38 / 38 | 0 |
| Unanswerable Abstention Accuracy | 1.0000 | 12 / 12 | 0 |

The 38 answers have 48 labeled evidence chunks in total. With k=5, the fixed
denominator gives a benchmark-specific perfect-retrieval Precision@5 ceiling of
48/(38×5) = 0.2526; measured precision reaches that ceiling. Citation Correctness
has the same numeric value because production reports all five retrieved chunks as
sources. It measures precision against narrow labeled evidence, not whether citations
were fabricated. Supporting Citation Coverage confirms all 48 labeled evidence chunks
were included. There were 142 additional reported chunks outside those narrow labels,
and no missing labeled chunks.

The first relevant chunk ranked first in 37 answerable cases and second in one
(`ai-03`); none were missed. All 38 answerable cases had full labeled-evidence recall,
including all nine multi-evidence questions. All 12 negatives were exact abstentions:
`commerce-09`, `commerce-10`, `privacy-07`, `privacy-08`, `terms-07`, `terms-08`,
`community-07`, `community-08`, `ai-07`, `ai-08`, `cookies-07`, and `cookies-08`.
Confusion counts are 38 true answers, 12 true abstentions, zero false answers, and
zero false abstentions.

Manual review flags from the actual rows, without changing frozen labels:

- `commerce-03`, `commerce-08`, `privacy-03`, `privacy-05`, and `community-05`
  each received RAGAS Context Precision 0 despite full labeled evidence recall and
  visibly relevant retrieved chunks. Review judge/context disagreement before drawing
  conclusions about retrieval quality.
- `cookies-04` received Context Precision 0.5 despite both labeled chunks in the
  top five; inspect its per-context judgments.
- `ai-03` is the only answerable case where the first labeled evidence chunk ranked
  second. Its answer also scored 0.424 Answer Relevancy while directly addressing the
  question; this deserves manual inspection.
- `cookies-03` scored 0.385 Answer Relevancy despite an answer that appears to cover
  both the analytics-consent distinction and the media-player condition.
- `community-02` scored 0.75 Faithfulness; inspect the clause about strongly worded
  disagreement against the retrieved guideline text.

The second attempt recorded eleven recoverable HTTP 503 capacity failures and two
timeouts across its first segment: six context-precision judge calls, two generation
calls, two answer-relevancy calls, one faithfulness call, plus one timeout each in
answer relevancy and faithfulness. All recovered within the bounded policy. No HTTP
429 or confirmed quota exhaustion was observed. The first attempt's old
`ClientError`/`InstructorRetryException` causes remain unprovable from its historical
metadata; the second attempt's safe diagnostics identify its own failures only.
