# Mistral and Qwen: SFT1, SFT2 and end-to-end results

Updated 8 October 2026. Both original comparisons are complete. Qwen's separate step-132 SFT1 comparison started at approximately 14:44 UTC and is running.

## Evaluation setup and definitions

The same 1,000 original client test claims from the supplied 2022 survey are used throughout. These are template claims, not the separate paraphrased package.

**SFT1 — analysis generation:** claim + survey catalogue + API instructions → Python program → API execution. Success requires execution and satisfaction of the gold analysis contract.

**SFT2 — evidence interpretation:** claim + evidence actually produced by SFT1 → a perspective citing supporting and contradicting evidence and limitations. Success requires a valid judge pass and deterministic citation, numeric-grounding and suppression checks.

SFT2 results below come from the chained run, not a standalone test supplied with gold evidence. Faithful interpretation of an incorrect analysis can pass SFT2 but cannot pass E2E. Failed upstream programs also reduce the number of answers eligible for judging. Answer pass rates use all judged cases, including invalid/error judgments; invalid judgments are unscored, not semantic failures or successes.

**End-to-end success:** SFT1 succeeds **and** SFT2 succeeds for the same test claim. Its denominator is always all 1,000 claims.

Serving: remote H200, vLLM 0.31.0, BF16, greedy decoding, 32,768-token context, 1,536 new tokens maximum per stage, 64 in-flight claims. Qwen thinking is disabled. All comparisons retain the frozen source snapshot, API/contracts, judge gate and shared content-addressed judge cache. Temperature stops are disabled.

Judge: `gpt-6-luna`, rubric `synthesis-judge-1.0.4`, concurrency 8. Automated qualification passed; expert calibration on this evaluation remains pending. Results are provisional and validate this study and these claims only.

## Mistral-Small-24B-Instruct-2501

SFT1: `checkpoint-000112-cda6fdee928e`, stopped gracefully at epoch 1.75. SFT2: `checkpoint-000032-199db898c8b3`, completed one epoch on the research-1k run.

### SFT1: analysis-code generation

| Outcome, out of 1,000 | SFT1 adapter | Base model |
|---|---:|---:|
| **Correct executable analysis** | **688 (68.8%)** | **0 (0.0%)** |
| Runs but fails gold contract | 214 | 1 |
| Execution error | 81 | 16 |
| Syntax error | 17 | 970 |
| Truncated generation | 0 | 13 |

### SFT2: interpretation of generated evidence

| Metric | SFT2 adapter after SFT1 adapter | Base answer after base analysis |
|---|---:|---:|
| Answers submitted to judge | 902 | 1 |
| Valid judgments | 891 | 1 |
| Invalid/error judgments | 11 | 0 |
| Answer passes judge + grounding checks | 851 | 0 |
| Pass rate among submitted answers | 94.3% | 0.0% |

SFT2 failure diagnostics (counts overlap within a case):

| Check | Failed cases |
|---|---:|
| Unsupported numbers | 28 |
| Reveals suppressed values | 15 |
| Missing required citations | 3 |
| Invalid citations | 0 |
| Judge: unsupported causal interpretation | 0 |
| Judge: mishandles inconclusive evidence | 3 |
| Judge: incorrect scope | 0 |
| Judge: incorrect direction | 5 |

Only one base answer was judged; this cannot establish standalone base answer quality.

## Qwen3-4B — completed pilot

SFT1: `checkpoint-000072-a9b622cfd7ae`. SFT2: `checkpoint-000072-73f848fee9d5`. Both completed three epochs on 181 training / 38 validation rows. This comparison does not use the older SFT1 step-132 adapter.

### SFT1: analysis-code generation

| Outcome, out of 1,000 | SFT1 adapter | Base model |
|---|---:|---:|
| **Correct executable analysis** | **819 (81.9%)** | **0 (0.0%)** |
| Runs but fails gold contract | 113 | 0 |
| Execution error | 68 | 443 |
| Syntax error | 0 | 81 |
| Truncated generation | 0 | 476 |

### SFT2: interpretation of generated evidence

| Metric | SFT2 adapter after SFT1 adapter | Base answer after base analysis |
|---|---:|---:|
| Answers submitted to judge | 932 | 0 |
| Valid judgments | 905 | 0 |
| Invalid/error judgments | 27 | 0 |
| Answer passes judge + grounding checks | 673 | 0 |
| Pass rate among submitted answers | 72.2% | Not measurable: no judged answers |

SFT2 failure diagnostics (counts overlap within a case):

| Check | Failed cases |
|---|---:|
| Unsupported numbers | 99 |
| Reveals suppressed values | 3 |
| Missing required citations | 0 |
| Invalid citations | 0 |
| Judge: unsupported causal interpretation | 64 |
| Judge: mishandles inconclusive evidence | 49 |
| Judge: incorrect scope | 45 |
| Judge: incorrect direction | 37 |

No Qwen base answers reached judging, so its standalone answer quality is unmeasured.

## Qwen3-4B — main SFT1 run, evaluation in progress

### SFT1

Selected adapter: `checkpoint-000132-45289da7702b`, from the 5k training package. It processed 2,098 training examples before stopping; it is not a completed 5k fit. Test analysis results are pending.

### SFT2

Keep the same pilot SFT2 adapter at step 72. Its results on the new SFT1 evidence are pending. This holds the answer model fixed while comparing SFT1 adapters; base/base is not repeated.

Run root: `/dev/shm/odyn-qwen-5k-e2e`. Queue PID 128062. Test inputs, decoding and judging settings match the completed pilot run.

## Provenance and saved evidence

Checkpoint paths, hashes and training histories: [Mistral provenance](MISTRAL_RUNS_REPORT.md), [Qwen provenance](../QWEN_RUNS_REPORT.md), [Qwen inventory](../evidence/qwen_runs_inventory.json), and [step-132 run plan](../evidence/qwen_5k_e2e_run_plan.json).

Local summaries, per-claim outputs and judge results (artifact links refer to the workspace outside this repository):

- [Mistral SFT](../../runs/mistral-e2e-full/runs/sft-sft-vllm-limit1000/summary.json)
- [Mistral base](../../runs/mistral-e2e-full/runs/base-base-vllm-limit1000/summary.json)
- [Qwen pilot SFT](../../runs/qwen-e2e-full/runs/sft-sft-vllm-limit1000/summary.json)
- [Qwen base](../../runs/qwen-e2e-full/runs/base-base-vllm-limit1000/summary.json)

## Grouped results: complete end-to-end pipeline

| Model and chain | SFT1 correct analysis | SFT2 passing answers / submitted | **E2E pass / all test claims** |
|---|---:|---:|---:|
| Mistral, SFT1 + SFT2 | 688/1,000 (68.8%) | 851/902 (94.3%) | **651/1,000 (65.1%)** |
| Mistral, base + base | 0/1,000 (0.0%) | 0/1 (0.0%) | **0/1,000 (0.0%)** |
| Qwen pilot, SFT1 + SFT2 | 819/1,000 (81.9%) | 673/932 (72.2%) | **635/1,000 (63.5%)** |
| Qwen pilot, base + base | 0/1,000 (0.0%) | 0/0 (not measurable) | **0/1,000 (0.0%)** |
| Qwen main-run SFT1 step 132 + pilot SFT2 step 72 | Pending | Pending | **Pending** |

Neither completed SFT chain reaches the client's **>80% E2E target**. Qwen pilot leads on SFT1 by 13.1 percentage points; Mistral leads on the full pipeline by 1.6 points. Training datasets and histories differ, so these describe the selected adapters rather than a controlled comparison of architectures. No statistical superiority is established. Standalone SFT2 evaluation on shared gold evidence would be required to isolate answer-model quality.
