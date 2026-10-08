# SFT2 response evaluation

`odyn-judge` grades saved base/adapter responses against the original claim and
executed Survey API evidence. Install the optional `judge` extra for the OpenAI
SDK. Generation and training remain separate and make no judge calls.

```bash
odyn-judge \
  --input data/qwen-sft2-local/validation_synthesis.jsonl \
  --base runs/qwen-sft2-inference-validation/base.jsonl \
  --sft runs/qwen-sft2-inference-validation/sft2.jsonl \
  --validation-report /path/to/synthesis_judge_validation.json \
  --out runs/qwen-sft2-inference-validation/judge --workers 4
```

Alternatively run `python -m odyn_sft.survey_evaluation.synthesis_cli`.
Set `OPENAI_API_KEY` in the environment, or supply `--credentials-file` pointing
to a private mode-0600 JSON file with that field. Keys are never copied into
the package, results or request ledgers. Add `--dry-run` to check context and
count requests without calling the API.

The harness uses `synthesis-judge-1.0.4`, `gpt-6-luna` and low reasoning effort.
Version 1.0.4 clarifies that scoped evidence support is not a truth verdict, and
that supplied respondent bases can be reported even when outcome values are
suppressed. A passing validation report for that judge
model/version is required before any call. The gate
requires at least 90% detection for critical error types, at most 5% false
item failures on reference answers and at least 95% self-agreement. This is
automated validation, not expert calibration on the current pilot. The current
version was checked on 23 controlled examples with 35 calls including repeats;
it does not inherit the historical version's 504-call validation as its own.

Expectations are derived deterministically from executed evidence: support,
contradiction, inconclusive findings, measurement gaps, unavailable comparisons, proxies, suppression,
causal limits and survey scope. The judge sees the claim, facts, expectations
and candidate text. Candidate model identity and reference prose are hidden.
Each answer is judged separately using Structured Outputs. Every expectation
must appear exactly once; required supporting quotes must occur in the answer
after normalising Unicode, spacing and Markdown emphasis/code markers.
Paraphrased or invented quotations remain invalid.
Prompt text and embedded labels are treated as data.

Separate deterministic checks verify evidence IDs, numerical grounding,
suppression and required citations. Their number matching is heuristic;
failures include offending text so they can be inspected.

The report separates three metrics:

| Metric | Definition |
|---|---|
| Critical semantic pass | Valid judge output; all critical expectations pass |
| Grounding + critical pass | Semantic pass plus all deterministic checks |
| Combined pass | Grounding + critical pass plus complete, nonempty generation |

Noncritical expectation failures are still reported. Repetitive or truncated
responses are judged as saved, without repair; truncation blocks combined pass.
Invalid judgments and provider errors stay unscored, with separate counts.
Neither a semantic score nor a case pass judges the claim itself true or false.

Each response has a cached API request and an auditable result with facts,
expectations, quotations, decisions and deterministic checks. The run writes
`manifest.json`, `validation_gate.json`, `cases/`, `ledgers/`, `results.jsonl`
and `summary.json`. Completed results can be reused only if input hashes,
judge implementation and settings match. A completed judgment that fails quote/ID validation gets at most one fresh
validation-repair call, with both attempts preserved. Failed or ambiguous API
requests are not silently replayed; reconcile their ledgers before retrying.
`--request-cache` can reuse an existing ledger directory; every fingerprint must
match before a cached response is accepted.

Validate a revised judge using a frozen control bank before scoring responses:

```bash
python -m odyn_sft.survey_evaluation.validate_synthesis_judge \
  --bank ../evidence/synthesis_judge_controls_v2.json \
  --out runs/judge-validation-1.0.4-complete
```

Then supply that run's `report.json` to `--validation-report`. The bank is
shipped alongside the delivery evidence, rather than generated during scoring.

The seven local validation cases support pilot diagnostics. They do not
establish the client's required >80% performance on 500–1,000 reviewed cases.

The provider uses the documented [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
contract through the Responses API.
