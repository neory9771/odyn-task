# Mistral SFT runs — findings

Started 2026-10-08. Source: read-only inspection of `odyn-h200` (`/dev/shm`) plus local
scored validation results under `odyn-task/runs/`. Times are UTC.

Base model for all runs: `mistralai/Mistral-Small-24B-Instruct-2501`
(revision `9527884be6e5616bdd54de542f9ae13384489724`).

## Configuration links (inside this repository)

The linked configs record the settings used for training and the full test evaluation. Paths inside them refer to the remote machine.

| Step | Configuration / command |
|---|---|
| Latest SFT1 training and teacher-forced validation, step 112 | [Recorded SFT1 training config](configs/recorded/mistral-sft1-step112-training.json) · [Reusable BF16 profile](configs/mistral-bf16.json) |
| Latest SFT2 training and teacher-forced validation, step 32 | [Recorded SFT2 training config](configs/recorded/mistral-sft2-step32-training.json) · [Research-1k profile](configs/mistral-sft2-bf16-research-1000-h200.json) · [Launcher](scripts/h200/sft2_research_1000.sh) |
| Full E2E test: SFT1 stage | [Recorded SFT1 inference config](configs/recorded/mistral-e2e-sft1.json) |
| Full E2E test: SFT2 stage | [Recorded SFT2 inference config](configs/recorded/mistral-e2e-sft2.json) |
| Full E2E test: BF16 vLLM server, adapter selection, concurrency and judging command | [vLLM launcher](scripts/h200/e2e_mistral_vllm.sh); full run used `LIMIT=1000 CONC=64` |
| Judge instructions and rubric | [Judge implementation](src/odyn_sft/survey_evaluation/synthesis_judge.py) · [Deterministic answer checks](src/odyn_sft/survey_evaluation/synthesis_checks.py) |

## Latest full client-test E2E — completed 2026-10-08

Both vLLM chains completed at **13:44:16 UTC** on the same 1,000 original
client test claims. Adapters: SFT1 step 112 and SFT2 step 32. This test does
not use the separate mixed paraphrase package. Results and the judge cache
were pulled locally to `runs/mistral-e2e-full/`.

| Chain | Cases | SFT1 contract passes | SFT2 grounded judge passes | E2E passes |
|---|---:|---:|---:|---:|
| SFT1/SFT2 | 1,000 | 688 | 851 / 902 judged | **651 / 1,000 (65.1%)** |
| Base/base | 1,000 | 0 | 0 / 1 judged | **0 / 1,000 (0%)** |

SFT1 failures in the tuned chain: 214 semantic failures, 81 execution errors
and 17 syntax errors. The base chain had 970 syntax errors, 16 execution
errors, 13 truncated outputs and one semantic failure. The tuned chain had
11 invalid/error judge responses, which cannot count as passes.

SFT2 can pass grounding against evidence from a semantically incorrect SFT1
analysis; those cases still fail E2E. The overall metric requires both correct
analysis and faithful interpretation. The automated judge gate passed, but
expert calibration/client acceptance remains pending. This result is below
the requested >80% target.

Machine-readable summaries:
`runs/mistral-e2e-full/runs/{sft-sft,base-base}-vllm-limit1000/summary.json`.
Training and validation results below are reported separately from the full test results.

## Summary

| Stage | Latest run (on box) | Status | Adapter to use | Key result |
|---|---|---|---|---|
| SFT1 (claim → program) | `/dev/shm/odyn-mistral-1000-bf16/run/train` | Stopped gracefully at step 112 / 192 (epoch 1.75 of 3) | `checkpoint-000112-*` | Generated validation: **89.4%** success (base: 1.2%) |
| SFT2 (evidence → perspective) | `/dev/shm/odyn-mistral-sft2-research-1000/sft-oneepoch/runs/mistral-sft2-research-1000/train` | Completed, 1 epoch, 32 steps | `best_adapter` (step 32) | Test answers: **851/902** pass judge and grounding checks |

---

## SFT2 — latest run

**Path:** `/dev/shm/odyn-mistral-sft2-research-1000/sft-oneepoch/runs/mistral-sft2-research-1000/`
**Supervisor job:** `odyn-mistral-sft2-research-1000` — EXITED 2026-10-08 11:26.

| Field | Value |
|---|---|
| Task | `sft2` |
| Precision | BF16 LoRA (`quantization: none`) |
| Epochs | 1 (completed), 32 optimizer steps |
| Training suite | `/dev/shm/odyn-mistral-sft2-research-1000/suite` |
| Acceptance checks | passed (learned delta verified, 4/4 validation points) |
| Best adapter | `train/best_adapter`, tag `epoch-1-examples-1000`, step 32 |
| Validation (898 ex.) | loss **0.000268**, perplexity 1.0003, teacher-forced token accuracy **99.997%** |
| Validation time | ~1304 s |
| Training validation | Teacher-forced loss; generated test answers were evaluated separately above. |

The low training-validation loss measures prediction of reference answers. Use the generated test results above to assess answer quality.

## SFT1 — latest run

**Path:** `/dev/shm/odyn-mistral-1000-bf16/run/train/`
**Supervisor job:** `odyn-mistral-1000-bf16` — STOPPED 2026-10-08 00:19
(`pipeline_status.json`: `interrupted`, `SystemExit 75`, which is a graceful stop that saved a checkpoint).

| Field | Value |
|---|---|
| Task | `sft1` |
| Precision | BF16 LoRA (`quantization: none`) |
| Planned | 3 epochs, 192 steps; batch 16, with 10-example windows at quarter boundaries |
| Reached | step **112**, epoch **1.75** (`training_summary.json` not written, because training did not complete) |
| Training suite | `/dev/shm/odyn-mistral-1000/suite-v2` |
| Checkpoints | `checkpoint-000112-cda6fdee928e`, `checkpoint-000112-cf1fc0a27435` |
| `best_adapter` | step 16 (`epoch-1-examples-0250`), selected by lowest validation loss |
| Train loss at step 112 | 4.7e-05 (grad norm 0.003) |
| Peak GPU memory | 103.8 GiB allocated / 132.4 GiB reserved |
| Training throughput | ~2000 input tokens/s, ~56 s per update |

### Validation loss (teacher-forced, 1000 examples)
| Tag | Loss | Token accuracy |
|---|---|---|
| epoch-1-examples-0250 (step 16) | **0.0841** (best) | 98.61% |
| epoch-1-examples-0500 | 0.1026 | 98.76% |
| epoch-1-examples-0750 | 0.0980 | 98.66% |
| epoch-1-examples-1000 | 0.1057 | 98.59% |
| epoch-2-examples-0250 | 0.1120 | 98.63% |
| epoch-2-examples-0500 | 0.0978 | 98.72% |
| epoch-2-examples-0750 | interrupted at 486/1000 | — |

Validation loss is flat to slightly rising after step 16, while train loss falls to ~0.
This is consistent with early overfitting on loss. Generated-code quality still improved, though (below).

### Generated validation (executed and scored, 1000 examples)
Local results: `odyn-task/runs/mistral-*-val1000*/scored.json`.

| Model | Success | Execution valid | Syntax valid | Failures |
|---|---|---|---|---|
| Mistral base (no adapter) | **1.2%** (12) | 7.2% | 14.9% | 851 syntax, 77 execution, 59 semantic, 1 truncated |
| SFT1 step 16 (`best_adapter`) | **80.4%** (804) | 95.5% | 99.9% | 151 semantic, 44 execution, 1 syntax |
| SFT1 step 112 (checkpoint) | **89.4%** (894) | 95.7% | 100% | 63 semantic, 43 execution |
| *Qwen3-4B SFT1, step 132 (reference)* | *92.7%* | *99.4%* | *100%* | *67 semantic, 6 execution* |

### Findings
- **Step 112 was selected for the test run.** Selecting on loss picked step 16, but step 112
  generates 9 points better (89.4% vs 80.4%), mainly through fewer semantic failures (63 vs 151).
- **Fine-tuning is essential:** the base model gets 1.2%, and 85% of its outputs are syntax errors.
- **Weak spot: `api_tutorial` records.** Mistral gets 13/100 right (44 semantic, 43 execution
  errors), while `research_claim` gets 881/900 (97.9%). The training subset contained
  no tutorial examples. A typical execution error is a wrong API signature,
  `association(..., by=...)`, which raises `TypeError`.
- Against Qwen3-4B, Mistral is better on `research_claim` (97.9% vs 93.8%) but much worse on
  `api_tutorial` (13% vs 91%).

## Remaining work

- Improve complete-pipeline accuracy from 65.1% to the client's >80% target.
- Address SFT1 API-tutorial failures and SFT2 numeric-grounding and suppression failures.
- Complete expert calibration of the judge and review of the evaluation set.
- Evaluate SFT2 on shared gold evidence to measure its quality independently of SFT1.
