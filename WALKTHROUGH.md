# Code walkthrough

This document follows one run from start to finish and names the file that does each step.
It is written to be read alongside the code. Module paths are relative to
`src/odyn_sft/`; `configs/`, `scripts/`, `data/` and `runs/` are relative to this
delivery folder. Commands below assume the package is installed in the active
environment (with the `judge` extra for OpenAI evaluation).
See the [glossary](docs/glossary.md) for the research and dataset terminology.

## 1. What the system does

A client makes a claim about a survey ("Women in Florida know our brand better than men").
We answer in two steps, and each step is a separately fine-tuned LoRA adapter on the same base model:

| Step | Model input | Model output | How we grade it |
|---|---|---|---|
| **SFT1** | Claim + study catalogue + Survey API manual | A short Python program that calls the Survey API | Execution and measured gold-contract checks; a separate LLM review checks claim coverage and meaning. |
| **SFT2** | Claim + the executed program + its evidence | A written perspective: what supports, contradicts or leaves the claim open | Code checks for citations and numbers, plus an LLM judge on code-built expectations. |

SFT2 is instructed to describe what the evidence shows, without a true/false
verdict on the whole claim. The judge checks that requirement.

**End-to-end (E2E) evaluation** tests the complete two-model inference process on
held-out test claims, using frozen adapters and generation settings. For a claim
`c` and study `D`, the process is:

```text
p = SFT1(c, catalogue(D), API instructions)   # generated Python program
e = execute(p, D)                           # evidence from that program
a = SFT2(c, p, e)                           # generated research perspective
```

Reference programs and evidence are used for grading, never substituted for
`p` or `e` in this process. Define `G(c, e)` as successful execution and agreement
with the measured gold contract, and `J(c, e, a)` as a valid SFT2 judgment passing
all critical expectations, deterministic grounding checks and generation checks.
The standard E2E pass indicator is `G(c, e) AND J(c, e, a)`. Its pass rate is the
number of passing cases divided by **all selected test rows**, including upstream
failures and unknown judgments; report those counts separately.

The stricter **E2E with both judges** also requires the SFT1 meaning judge to
approve the same generated program's claim coverage, mappings, relationships
and limitations. Neither score judges whether the whole claim is true or false.
Standalone SFT2 evaluation instead supplies frozen reference evidence, so it
measures interpretation quality without testing the upstream SFT1 model.
Sections 6.4–6.5 describe the implementation, commands and saved results.

The package has five commands. Each one is a thin `argparse` front end:

| Command | Entry point | Purpose |
|---|---|---|
| `odyn-dataset` | `dataset_generation/cli.py` | Build and validate datasets; optional OpenAI target-writing and paraphrasing |
| `odyn-sft` | `run_management/cli.py` | Prepare tokens, probe memory, train, validate, test |
| `odyn-infer` | `inference/cli.py` | Generate answers from JSONL with the base model or an adapter |
| `odyn-judge` | `survey_evaluation/synthesis_cli.py` | Grade saved SFT2 answers with the validated judge |
| `odyn-e2e` | `inference/e2e.py` | SFT1 generation → execution → SFT2 generation → SFT2 judge on held-out test claims |

## 2. Package map

```
odyn_sft/
  survey_api/          The statistical engine programs call (the "Survey API")
  survey_metadata/     Catalogue encoding and response cleaning shared by prompts and execution
  prompts/             The instructions and inputs shown to the model
  dataset_generation/  Dataset builders and optional OpenAI input/target writing
  tasks/               SFT1 / SFT2 adapters: which files, how to score
  pytorch_sft/         Task-independent trainer: data, training, evaluation, monitoring
  run_management/          odyn-sft command, GPU supervisor, memory probe
  inference/           odyn-infer and two-adapter odyn-e2e
  survey_evaluation/   Execute and score outputs; SFT2 facts, checks and judge (odyn-judge)
  providers/           OpenAI client with request ledgers (judges and optional dataset writing)
  common/              JSON I/O, atomic writes, hashing
```

There is one dependency rule: the trainer (`pytorch_sft/`) never imports survey code directly.
It asks the task registry (`tasks/__init__.py`) for an adapter by name, and the adapter
brings the survey-specific parts. A test (`tests/test_runtime_boundary.py`) also
checks that scoring works with every `dataset_generation` import blocked.

## 3. The Survey API (`survey_api/`)

See the [Python Survey API guide](docs/survey-api.md) for runnable examples,
function signatures, metadata requirements and evidence/base rules.

Every SFT1 program is ordinary Python run against one study:

```python
g1 = group(var(1), {'North': ['North'], 'South': ['South']})
with subclaim('c1', 'North agrees more often than South'):
    e1 = proportion(var(2), 'Agree', by=g1)
    compare(e1, 'North', 'South', expect='>')
```

| File | Role |
|---|---|
| `service.py` | `SurveyAPI`: one analysis session over one study. It is composed from the four groups below |
| `vocabulary.py` | `var`, `values`, `where`, `eligible`, `group`, `subclaim`: resolve variables and respondents |
| `estimates.py` | `proportion`, `distribution`, `rank_options`, `top_box`, `mean_score` |
| `contrasts.py` | `compare`, `trend`, `association`, `adjusted_compare` |
| `evidence.py` | `not_measured`, `proxy`, `caveat`, `segment_sizes`, and `pack()`, which returns the evidence JSON |
| `statistics.py` | Wilson/Clopper-Pearson intervals, tests and Holm adjustment (scipy/statsmodels) |
| `filters.py` | Respondent filters that keep their observed base under `&`, `|`, `~` |
| `runtime.py` + `worker.py` | Run untrusted program text in a fresh, time-limited subprocess; return only the evidence |
| `constants.py`, `types.py`, `errors.py`, `version.py` | Shared vocabulary, typed JSON contracts, stable error codes, versions |

Every comparison records an id (e.g. `c1.r3`), its numbers, its interval and a label:
`consistent`, `inconsistent` or `no_clear_difference` relative to the declared `expect`.
`pack()` then applies Holm adjustment across all tests and records `label_holm`.
Answers in SFT2 cite these ids.

## 4. Building datasets (`dataset_generation/`, `odyn-dataset`)

The deterministic builders below need no LLM calls. The same CLI also exposes
`synthesis write-targets` and `variants validate-verifier/generate`, which use
OpenAI. `variants apply` assembles existing decisions without new calls.
Dataset manifests record file hashes and versions.

| Command | Code | What it produces |
|---|---|---|
| `synthetic curriculum/validate-curriculum/filter` | `synthetic/` | Seeded surveys, paired SFT1/SFT2 curricula and metadata-filtered subsets |
| `metadata build` | `metadata.py`, `ordering.py` | Study metadata JSON from the downloaded survey |
| `evaluation audit/generate` | `generation/` | Client-evaluation cases: claim, reference program, gold |
| `splits audit/generate/validate` | `dataset/`, `partition.py`, `selection.py` | Paired train/validation/test package |
| `synthesis export` | `synthesis_view.py` | SFT2-only package retaining existing deterministic response targets |
| `synthesis assemble/validate` | `synthesis_suite.py`, `tasks/sft2.py` | SFT2 suite from accepted written targets; verify hashes and split isolation |
| `use-cases build` | `use_cases/` | Fixture covering each Survey API capability |
| `rubrics build/validate` | `rubrics/` | Frozen judge rubrics for the pilot/methodology/use-case suites |

The client evaluation and paired split generators follow two rules:

1. **Families are assigned to splits before any outcome is computed.**
   `partition.py` puts each analysis family in one split using metadata only.
   No result can therefore influence which split a case lands in.
2. **Audit, review, then generate.** `audit` writes a capacity plan (`plan.json`) and no samples.
   `generate` needs `--acknowledge-plan`, recomputes the plan and refuses to run if the code or inputs have changed.

Inside `generation/`, the stages run in order:
`inventory.py` (which comparisons the study supports) → `scenarios.py` (what the client asserts beyond the data) →
`candidates.py` (combine the two into cases) → `selection.py` (coverage-balanced sampling) →
`cases.py` (render the claim, run the reference program, check the evidence independently with
`evidence.py`, write the reference perspective) → `pipeline.py` (the `audit` and `generate` commands).

Each suite gets a gold contract (`contracts.py`): what an acceptable program must measure.
The reference program is an example, not the only correct answer.

## 5. Training (`odyn-sft`)

### 5.1 Configuration (`pytorch_sft/config.py`)

One frozen JSON file per run. For example, `configs/mistral-bf16.json` contains the model and its
pinned revision, the data paths, LoRA settings, LR and epochs, `task` (`sft1` or `sft2`) and the system prompt.
`Config.__post_init__` rejects anything outside the tested profile:
BF16 compute, an NF4 or BF16 base, all-linear LoRA, AdamW, cosine schedule and at most 3 epochs.
`workspace` resolves relative to the config file; data/output paths then resolve
relative to that workspace. Absolute paths remain absolute.

### 5.2 Prepare: text to tokens (`pytorch_sft/data/`)

`odyn-sft prepare` → `data/preparation.py::prepare`

1. The task adapter validates the suite (`tasks/sft1_suite.py` or `tasks/sft2.py::validate_suite`):
   file hashes, split counts and split-isolation checks appropriate to that
   package. Synthetic studies are split-disjoint; a single-survey evaluation
   intentionally uses the same source survey across splits. See each validator
   for the exact identities checked.
2. `data/tokenization.py::tokenize_row` renders each row with the model's own chat template:
   optional system prompt, a user turn (instruction + input), then the assistant turn (the target).
3. **Only the assistant tokens, including EOS, carry loss.** Prompt tokens are masked.
   An example longer than `max_length` is rejected, never truncated.
4. Train and validation are written as flat token arrays (`*.tokens.bin`, `*.offsets.npy`, `*.prompts.npy`).
   The test split is checked but never tokenized.

### 5.3 Probe memory (`run_management/probe.py`)

`odyn-sft probe --batch N` loads the model and runs one optimizer update on the longest training
sequences. It then discards the weights and reports peak GPU memory. Use it to choose `microbatch_size`.

### 5.4 Train (`pytorch_sft/training/`)

`odyn-sft train` → `run_management/supervisor.py` starts the worker process (`pytorch_sft/cli.py train`),
records GPU telemetry and forwards a stop request. The worker runs `training/loop.py::train`:

| Step | File | What happens |
|---|---|---|
| Setup | `run_setup.py` | Re-audit the prepared data; compute the run fingerprint (config + data + code hashes); build the update windows |
| Model | `model_setup.py` | Load the pinned base (NF4 or BF16), freeze it, add LoRA, build the optimizer and scheduler |
| Manifest | `run_manifest.py` | Write the effective settings and all provenance |
| Resume | `checkpoints.py` | Restore the latest checkpoint if its fingerprint matches, including optimizer, scheduler, RNG and sampler position |
| Update | `updates.py` | One optimizer step over a window of examples |
| Loss | `objective.py` | Exact completion-token loss. The output head runs only on supervised positions, in checkpointed chunks |
| Validate | `selection.py` | At 25/50/75/100% of each epoch: full validation loss; keep `best_adapter` |
| Finish | `training_summary.py` | Save `adapter/`, write `training_summary.json` with acceptance checks |

Key behaviours to explain to a client:

- **Update windows.** `config.update_windows` groups examples into windows of
  `microbatch_size × gradient_accumulation` examples.
  A window is shortened so that it ends exactly on a quarter-epoch boundary.
  The loss is weighted by completion tokens across the whole window, so long and short answers count per token.
- **Validation and selection.** At each quarter boundary the full validation set is scored with teacher forcing.
  The lowest-loss adapter is saved as `best_adapter/` with `selection.json`.
  An adapter that has not moved from its initial state cannot be selected.
  Validation loss is a training signal. Generated-output quality is measured separately (section 6),
  and the best-loss checkpoint is not always the best generator.
- **Safe stop and resume.** SIGTERM or Ctrl-C sets a flag. The loop saves at the next window boundary and exits with code 75.
  A half-finished window is discarded and its RNG state restored, so the resumed run replays it exactly.
  Rerunning the training command with the same frozen config resumes. A fresh-run
  shell launcher may refuse an existing directory, so use `odyn-sft train` for
  resume. A changed config, dataset or code changes the fingerprint and blocks
  ordinary resume. Explicit `continuation_from` supports audited, limited
  changes, including a reduced epoch budget, in a new output directory.
- **Logging.** `monitoring/training_logs.py` writes TensorBoard (and W&B if configured):
  loss, learning rate, gradient norm, tokens per second and memory.
  It also writes `live_status.json` after every microbatch.

## 6. Evaluating generated outputs

### 6.1 Validation and test (`pytorch_sft/evaluation/`)

- `odyn-sft validate-base` / `validate` → `evaluation/validation.py`.
  These generate greedily on the validation split with the base model or the selected adapter.
- `odyn-sft test` → `evaluation/final.py`. It runs only after training completed, the selected adapter's hashes match,
  and the adapter is shown to have learned. It generates on full validation and
  test. Package integrity checks may read held-out files earlier; test targets
  are not used for fitting or checkpoint selection. This command makes no judge
  calls: SFT2's report is a generation diagnostic until separately judged.
- `evaluation/evaluator.py` is the shared engine: completion loss, greedy generation, per-case caching.
  It hands each completion to the task's scorer.

### 6.2 SFT1 scoring (`tasks/sft1.py`)

For each generated program:

1. Strip one surrounding Markdown code fence, if present (`code_text`).
2. Parse it. Failure is a `syntax_error`; empty/invalid output has its own category.
3. Execute it in the worker process. Failure is an `execution_error`.
4. Check its evidence against the gold contract (`survey_evaluation/execution.py`, `meaning.py`):
   right variables, populations, groups, comparison and direction. Failure is a `semantic_failure`.
5. Otherwise the outcome is `successful`; truncation and unavailable semantic
   checks have separate failure categories.

The report gives the success rate and a count for each failure type. Deterministic
success checks measured analyses; it does not establish that every gap, proxy or
material proposition was handled correctly.

`survey_evaluation/analysis_judge.py` adds a reference-conditioned LLM review of
material propositions, mappings/populations, comparison meaning and identification
limits. It grades saved held-out programs; the reference never enters inference.
Execution and numerical contracts remain separate requirements. Quotes must match
candidate code, and invalid judgments are reported separately from model failures.
This pilot rubric is not an expert-calibrated client acceptance test.

### 6.3 SFT2 scoring (`survey_evaluation/`, `odyn-judge`)

SFT2 answers are prose, so generation diagnostics (`tasks/sft2.py`) only check that an answer exists and was not cut off.
Quality is graded by `odyn-judge`:

| Stage | File | What it does |
|---|---|---|
| Facts | `synthesis_facts.py` | From the evidence JSON, code builds the facts: for each claim part, which results support, contradict or are inconclusive; gaps, proxies, suppressed groups |
| Expectations | `synthesis_facts.py` | Turn the facts into a yes/no checklist (E1–E11). Critical items include "cites the contradicting result" and "gives no verdict" |
| Code checks | `synthesis_checks.py` | D1 cited ids exist; D2 every number traces to the evidence; D3 no suppressed values; D4 every required id is cited |
| Judge | `synthesis_judge.py` | The LLM answers each expectation with a supporting quote. Quotes must appear in the answer, or the judgment is invalid |
| Run | `synthesis_runner.py` | Pairs base and adapter answers, hides which is which, caches requests in ledgers, writes `summary.json` |
| Gate | `validate_synthesis_judge.py` | Before use, the judge must catch planted errors (≥90% per type), pass correct answers (≤5% false failures) and agree with itself (≥95%) |

The current SFT2 judge uses `gpt-6-luna`. Facts and expectations are built from
the evidence in the candidate's own input, rather than from a reference prose
answer. Passing means a valid judgment with every **critical** expectation
passed, all D1–D4 checks passed, and a nonempty, untruncated response. Noncritical
criterion failures are reported separately. Ambiguous suppression attribution
is left to critical E7 review rather than accepted solely by the numeric check.

The gate is automated control validation, **not expert calibration**. The runner
checks the gate's model/version, any supplied prompt hash, and thresholds before
submitting candidates. Each answer normally needs one OpenAI call; an invalid
completed judgment may receive one format/quote repair. Remaining invalid or
provider-error judgments are unknown, not semantic failures or successes.

### 6.4 Standalone SFT2 versus actual end-to-end test

These answer different questions and must have separate reports:

| Evaluation | Evidence supplied to SFT2 | What a pass establishes |
|---|---|---|
| Standalone SFT2 | Frozen executed **reference** analysis | The response model interprets a correct supplied analysis |
| `odyn-e2e` | Executed **SFT1-generated** analysis | SFT1 meets the measured gold contract and SFT2 faithfully interprets its actual evidence |
| Pilot E2E with both judges | Same SFT1-generated evidence | The above, plus the SFT1 meaning judge passes coverage, mappings, relationships and limitations |

A faithful response to the wrong analysis can pass the SFT2 judge. That is why
E2E also requires SFT1's gold-contract success, and the stricter pilot adds the
SFT1 meaning review. Standalone stage scores cannot be multiplied to obtain the
E2E score; they use different evidence and their failures can be dependent.

For each selected **test** row, `inference/e2e.py` does this:

1. Load the claim/catalogue/API prompt from `test_analysis_inputs.jsonl` and
   align it with `test_lineage.jsonl`.
2. Activate the SFT1 adapter and generate code. The scorer executes it against
   the supplied study and checks the private measured-analysis contract.
3. If execution succeeds, build SFT2's input from that claim, generated code
   and its returned evidence. Reference evidence never replaces these results.
   SFT2 still runs when code executes but fails the semantic contract, so its
   interpretation can be diagnosed independently.
4. Activate the SFT2 adapter on the same loaded base and generate the perspective.
5. Build facts/expectations from the actual returned evidence and judge the
   perspective. Judge requests hide the adapter/model identity.
6. Join results by `record_id`. Execution failures have no SFT2 answer and stay
   in the E2E denominator; truncation, contract failures and unknown judgments
   also cannot pass.

The standard rate in `summary.json` is:

```text
count(SFT1 outcome == successful AND SFT2 combined_pass == true)
----------------------------------------------------------------
                 all selected test rows
```

The stricter pilot additionally requires `analysis_judge.semantic_pass == true`
for the **same pipeline program**, not a separately regenerated SFT1 answer.
It is recorded in `runs/pipeline-status.json:e2e_both_judges` and the archived
pilot report. `odyn-e2e` alone does **not** run that additional judge.

### 6.5 Commands and saved artifacts

Use frozen configs and adapters selected on validation. The shell paths below
are illustrative: replace them with the actual completed run paths. `PACKAGE`
must be a **paired analysis/synthesis test package**, including study data,
metadata, private gold contracts and lineage. A synthesis-only training subset
does not contain everything E2E needs.

```bash
SFT1_CONFIG=runs/sft1/effective-config.json
SFT2_CONFIG=runs/sft2/effective-config.json
SFT1_ADAPTER=runs/sft1/train/best_adapter
SFT2_ADAPTER=runs/sft2/train/best_adapter
PACKAGE=data/paired-suite
GATE=runs/judge-controls/report.json

# Standalone generated-test diagnostics; no OpenAI calls.
odyn-sft test --config "$SFT1_CONFIG" --disable-temperature-checks
odyn-sft test --config "$SFT2_CONFIG" --disable-temperature-checks

# Actual pipeline on all held-out test rows; includes paid SFT2 judging.
# OPENAI_API_KEY must already be set, or pass --credentials-file privately.
odyn-e2e \
  --sft1-config "$SFT1_CONFIG" --sft1-adapter "$SFT1_ADAPTER" \
  --sft2-config "$SFT2_CONFIG" --sft2-adapter "$SFT2_ADAPTER" \
  --package "$PACKAGE" --out runs/e2e-test \
  --validation-report "$GATE" --workers 8 --disable-temperature-checks
```

Use `--limit 10` only for a smoke test; omit it for the complete held-out result.
Use `--skip-judge` for generation/execution only; this produces no judged E2E
pass rate. Each step runs its adapter when one is given and the base model
otherwise. For the base-model comparison, omit both adapters and use a separate
directory (omit just one for a mixed chain, e.g. SFT1 adapter with base SFT2):

```bash
odyn-e2e --sft1-config "$SFT1_CONFIG" --sft2-config "$SFT2_CONFIG" \
  --package "$PACKAGE" --out runs/e2e-test-base \
  --validation-report "$GATE" --workers 8 --disable-temperature-checks
```

For full test runs use vLLM: start an OpenAI-compatible `vllm serve` with both adapters as
LoRA modules named `sft1`/`sft2`, then add `--engine vllm --vllm-url http://127.0.0.1:PORT/v1
--concurrency 64`. Prompts are tokenized in this package (same chat template as the HF
engine) and sent as token IDs; claims run concurrently so vLLM batches them and reuses the
shared prompt prefix. On a 1-claim Mistral check, SFT1 and SFT2 outputs were token-identical
to the HF engine. `scripts/h200/e2e_mistral_vllm.sh` starts the server, runs sft/sft and
base/base, and stops it. The engine is recorded in `e2e.manifest.json`.

Rerunning with the same `--out` resumes: finished claims are skipped, and
`e2e.manifest.json` makes it refuse if configs, adapters or package files changed.
A generation error (for example a prompt longer than `max_length`) is recorded as
`runtime_error` for that claim instead of stopping the run. Judge requests are cached by content
under `--judge-cache` (default `.cache/judge`, git-ignored) and shared across runs,
so rerunning a judged run costs nothing. Both configs must share base model, revision, quantization and regex
settings; system prompts and generation budgets remain task-specific. The E2E
runner currently loads one Transformers base with both adapters, rather than
using vLLM or batched inference.

For a **standalone SFT2 base-versus-adapter** comparison on reference evidence:

```bash
# ORACLE_INPUT contains instruction, input and metadata.record_id per test row.
# input already contains the frozen reference execution, never SFT1 predictions.
ORACLE_INPUT=data/test-synthesis-inputs-with-lineage.jsonl
odyn-infer --config "$SFT2_CONFIG" --input "$ORACLE_INPUT" \
  --out runs/sft2-test-base.jsonl --disable-temperature-checks
odyn-infer --config "$SFT2_CONFIG" --adapter "$SFT2_ADAPTER" \
  --input "$ORACLE_INPUT" --out runs/sft2-test-adapter.jsonl \
  --disable-temperature-checks
odyn-judge --input "$ORACLE_INPUT" --base runs/sft2-test-base.jsonl \
  --sft runs/sft2-test-adapter.jsonl --out runs/sft2-test-judge \
  --validation-report "$GATE" --workers 8
```

This path requires the same input file and inference settings for both models;
the judge verifies their adjacent `.jsonl.manifest.json` files. Standard
`test_synthesis_inputs.jsonl` can omit metadata: join its matching
`test_lineage.jsonl` before using it with `odyn-judge`. Keep the `output` target
out of the input export; only `instruction` and `input` become model messages.
`odyn-judge --dry-run` validates the paired inputs/gate and counts calls without
submission. It exits nonzero for remaining invalid/error judgments even though
per-case results and the summary are saved.

`GATE` is a passing report from
`python -m odyn_sft.survey_evaluation.validate_synthesis_judge --bank CONTROL_BANK.json --out runs/judge-controls`.
This command makes OpenAI calls and needs a frozen control bank, not test
candidate answers. An existing report can be reused only when it matches the
current judge checks. Automated controls do not certify client acceptance.

| Saved artifact | Contents |
|---|---|
| `runs/e2e-test/e2e.manifest.json` | Config, adapter and package hashes guarding resume |
| `runs/e2e-test/e2e.jsonl` | One row per case: claim/ID, SFT1 code/status, constructed SFT2 input and generated answer |
| `runs/e2e-test/judge/results.jsonl` | Facts, expectations, D1–D4 findings, judgment/quotes, generation status and combined pass per generated SFT2 answer |
| `runs/e2e-test/judge/ledgers/` | Cached provider requests, responses and any completed quote repair |
| `runs/e2e-test/judge/manifest.json` | Judge/model/prompt/code versions and source hashes |
| `runs/e2e-test/summary.json` | Standard E2E numerator/denominator, first-stage outcomes and template/paraphrase breakdown |

For a separate SFT1 meaning review, the public module command is
`python -m odyn_sft.survey_evaluation.analysis_judge --package "$PACKAGE" --cases CASE_DIRECTORY --out runs/sft1-judge`.
It expects full-test `case-*.json` records with `record_id`, `program`, `evidence`,
`error` and `outcome`. Completed SFT1 test evaluation supplies those under
`train/final_evaluation/test/selected-final/`. For **pipeline** code, the pilot
orchestrator extracts those records from `e2e.jsonl`, resolves numeric variable
aliases for reference matching, then invokes the same judge. That extraction
and both-judge aggregation are currently pilot scripts, not a generic E2E flag.

Report unknown-judgment counts alongside a conservative all-row rate. For
example, 80 passes, 15 valid failures and 5 unknowns among 100 rows means **80%**
conservative pass rate, not 80/95. Report repeated paraphrases separately and
treat the original claim as the unit when estimating uncertainty.

Current limitations: `odyn-sft test` enforces final training/selection checks;
`odyn-e2e` does not enforce those checks or validate the entire package itself.
Verify them before running E2E. Its generation resume skips saved record IDs
without a model/input/settings fingerprint, so use a new output directory after
any change. The judge phase does enforce its own provenance. Freeze prompts,
decoding and adapter selection before test; test-driven fixes require a new
held-out evaluation before claiming acceptance. Confirm the finished JSONL has
exactly the intended test record IDs before reporting a full-test rate; the
standard summary's denominator is the saved rows, so a partial run is not a
complete evaluation.

## 7. Inference (`inference/`)

`odyn-infer` reads JSONL rows (`instruction`, `input`), renders them with the same chat template as training,
and generates greedily one row at a time. It writes each result as soon as it is generated.
A run can be resumed; the resume checks that the inputs, model, adapter and settings are unchanged.
It never reads gold answers.

## 8. Provenance

Dataset, training, standalone inference and judge manifests record what produced
their artifacts (the E2E generation-resume exception is noted in section 6.5):
- **Code:** `*/provenance.py` hashes the source files involved.
- **Data:** manifests hash every file.
- **Run:** `run_manifest.json` records the model revision, library versions and the full resolved config.

This is why a code change blocks resuming an old run: the fingerprint no longer matches.
Use a fresh run, or an explicitly supported audited continuation. Never rewrite
the old fingerprint to force resume.

## 9. Isolated remote two-stage pilot

The completed Qwen pilot lives at `/dev/shm/odyn-e2e` on `odyn-h200`. It uses final
code and separate output directories, leaving production datasets unchanged.
Its 100/20/30 original train/validation/test claims have 81/18/28 accepted OpenAI
input paraphrases, giving 181/38/58 rows. Both stages share identical claim text
and record IDs, audited in `runs/shared-claims-audit.json`. Test therefore contains
30 underlying claims, not 58 independent cases. Training and validation SFT2
targets were written from executed reference evidence; test targets are not used
for fitting or adapter selection.

`scripts/pilot/complete_remote_e2e.py` waits for SFT1's completed training summary,
trains SFT2, runs the frozen final test commands, judges both stages separately,
then runs `odyn-e2e`. SFT2 receives the code and evidence actually generated by
SFT1 in that pipeline. The stricter combined report additionally requires the
SFT1 meaning judge to pass the actual pipeline code. A failed first stage remains
in the overall denominator. Every stage saves logs and per-case results.

These are orchestration commands for that **existing pilot layout**, not generic
evaluation commands. The second can train SFT2 and replay unfinished stages;
use section 6.5 when only evaluation is needed. On the remote box:

```bash
PYTHONPATH=/dev/shm/odyn-e2e/eval-sft/src /workspace/envs/hf/bin/python \
  /dev/shm/odyn-e2e/scripts/check_analysis_judge.py
PYTHONPATH=/dev/shm/odyn-e2e/eval-sft/src /workspace/envs/hf/bin/python \
  /dev/shm/odyn-e2e/scripts/complete_remote_e2e.py
```

Run long jobs through the remote supervisor service `odyn-qwen-e2e-pilot`.
Training uses its frozen `sft/` code snapshot; evaluation uses `eval-sft/`, so
adding a judge does not change an active training fingerprint. Outputs under
`/dev/shm` must be copied off the box before it is restarted or destroyed.

The completed pilot report is saved at `../evidence/qwen-e2e-small/report.md`,
alongside both selected adapters, datasets, code snapshots and per-case outputs.
Combined passes were 46/58 for SFT1, 45/58 for SFT2 on reference evidence, and
33/58 for the actual pipeline with both judges. Invalid judgments are retained as
unknown and counted conservatively as non-passes. `rescore_grounding.py` preserves
original judgments while auditing a deterministic suppression-attribution fix;
it makes no LLM calls. Audited results live in `sft2-judge-audited` and
`e2e/judge-audited` beside the raw scores. W&B training summaries include these
held-out metrics and invalid-judgment counts.
