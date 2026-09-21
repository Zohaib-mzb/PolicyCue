# PolicyCue Eval v1 — controlled policy QA benchmark

Version: **1.0.0**. Sources and labels were authored for this repository, without copying
external policies. Existing repository policy snippets were reviewed but were short mock
inputs, not suitable multi-chunk ground truth. The existing four-case smoke dataset is
preserved separately as `dataset-starter.json`.

## Composition

| Fixed source | Concepts | Production chunks | Questions | Answerable | Unanswerable |
| --- | --- | ---: | ---: | ---: | ---: |
| `fixtures/commerce.txt` | Returns, processing, exchanges, shipping, subscriptions, defects | 8 | 10 | 8 | 2 |
| `fixtures/privacy.txt` | Account data, diagnostic/support/billing retention, deletion, providers, exports | 8 | 8 | 6 | 2 |
| `fixtures/terms.txt` | Registration, trials, billing, ownership, suspension, closure | 8 | 8 | 6 | 2 |
| `fixtures/community.txt` | Harassment, privacy, promotion, graphic content, sanctions, appeals, bots | 8 | 8 | 6 | 2 |
| `fixtures/ai.txt` | Prompt handling, training, retention, sensitive inputs, review, high-impact use | 8 | 8 | 6 | 2 |
| `fixtures/cookies.txt` | Session/security/analytics cookies, aggregates, consent, media, browser storage | 8 | 8 | 6 | 2 |
| Total | | **48** | **50** | **38** | **12** |

Primary categories are exclusive labels, while reasoning demands overlap:

| Primary question type | Count |
| --- | ---: |
| Direct fact | 6 |
| Paraphrased | 6 |
| Qualifier-sensitive | 6 |
| Multiple sentences in one section | 6 |
| Multi-chunk | 6 |
| Semantically confusing periods/events | 3 |
| Negative/exception | 5 |
| Unanswerable | 12 |

Nine examples require multiple chunks: six designated multi-chunk and all three
semantic-confusion questions. Eight require two chunks; `commerce-07` requires three.
The 12 negatives are plausible absent facts or unsupported qualifiers, not questions
that should simply receive a factual "no". Answerable negative questions remain
`should_answer=true` when the policy explicitly supports the negative answer.

## Evidence review and reproducibility

1. The six complete source documents were authored first.
2. Production `chunk_text()` was run without changing its 1200-character size,
   200-character overlap or separator order. All resulting chunk IDs, lengths and
   boundaries were inspected. Each resulting chunk equals one complete authored
   paragraph/section; this equality was checked against the full output, not guessed
   from section numbering. SHA-256 hashes were calculated from that output in order.
3. Questions and references were then labeled against those actual chunks. Compound
   references include each required section. Notes distinguish similar facts, periods,
   entities and exceptions; negatives were reviewed against the whole document.
4. The strict loader validates document/example uniqueness, evidence references and
   reference/answerability consistency. It rejects any source or splitter drift that
   changes the ordered chunk manifest. Quality tests additionally require multiple
   documents, more than five chunks per document, question diversity, and multi-chunk
   evidence. Fixture text must match the text materialised in the JSON dataset.
5. Freeze sources, labels, version and dataset hash before running a baseline. Any
   subsequent source or ground-truth correction requires a version change and explicit
   documentation; do not silently rescore a baseline using revised labels. Report
   critique can identify a label defect without altering the original report.

To review every actual chunk offline:

```bash
PYTHONPATH=. .venv-evaluation/bin/python -m backend.evaluation.evaluator --validate-only
```

Questions, reference answers, fixed evidence IDs, categories and individual rationales
are all in `dataset.json`. This is an author-reviewed benchmark prepared with an AI
assistant, not an independently annotated human benchmark. Independent human review is
still appropriate before external or portfolio claims.

## Interpretation limits

- This is a controlled synthetic policy QA baseline, not a sample of deployed customer
  documents or a measure of all real-world policy performance. There are no live URL,
  OCR, PDF extraction, table-reading, multilingual or prompt-injection cases here.
- Real owner/document filtering is preserved. Each top-5 query ranks eight competing
  sections inside one policy; cross-document retrieval is intentionally not measured.
- The sections have clean boundaries. Although the real overlap configuration is
  preserved, these paragraphs do not require repeated overlap text. Boundary-spanning
  evidence and very long documents remain future benchmark coverage work.
- Chunk-level evidence labels indicate support for the specific reference, not every
  loosely related section. A competing chunk is not necessarily useless context merely
  because it is outside the labeled answer evidence. Review unexpected citations and
  alternate support before assigning fault to retrieval or generation.
- With k=5 and one relevant chunk, even perfect evidence retrieval yields Precision@5
  of 0.2. At perfect recall, the macro precision ceiling for these 38 answerable cases
  is **48 / (38 × 5) = 0.252632**. This is a property of the labels and denominator,
  **not a measured score**. Production reports all retrieved chunks as sources, so
  citation correctness has a similarly strict evidence-set interpretation. Low
  precision alone is not evidence of a production defect or a target of 1.0.
- Multi-chunk references are supported in full only when all required chunks are
  available. A fluent partial answer may still satisfy the binary answerability check;
  inspect answer completeness separately. RAGAS does not implement reference-answer
  correctness in this stack. Its three semantic metrics are useful signals, not a
  replacement for inspecting the actual answers and labels.
- Judge metrics are stochastic. No repeated live run should be performed simply to
  improve the displayed baseline. Nulls, errors, sample counts, exact models, dataset
  hash and cleanup outcomes belong alongside the scores.

Frozen v1 dataset SHA-256, using the evaluator's canonical `Dataset.model_dump_json()`:
`346c1176648c0fe49e322f481adc20e43bd67d3f5d78cc2f21b3f6193200a440`.
This includes sources, ordered manifests, references and label notes.
