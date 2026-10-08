# Local Qwen SFT2

SFT2 learns to write a research perspective from a claim, reference analysis code and executed evidence. It starts a fresh Qwen3-4B adapter; it does not load an SFT1 adapter. Inputs are oracle-conditioned, so this pilot does not measure the full code-generation → execution → response pipeline.

Run commands from `final/sft/`. Every stage is explicit. No script calls OpenAI, starts background training, or kills unrelated GPU processes.

## Current data and memory checks

The existing teacher-gated pilot supplies 39 train and 11 validation responses. The largest Qwen training conversation has 27,976 tokens. Batch-one probes at rank 16 and rank 8 both ran out of memory on the local 16 GiB RTX 3080 Ti. Trial weights were discarded. The selected short profile passed batch-one forward/backward/optimizer probing: 10.13 GiB peak allocated and 11.00 GiB peak reserved, with an 11,095-token example.

The local profile therefore selects **complete train/validation conversations of at most 12,000 Qwen tokens**, producing **24 train / 7 validation / 10 test** records. Maximum selected lengths are 11,095 train and 10,868 validation tokens. The model context limit remains **32,768**. No prompt or completion is truncated; the excluded record IDs, tokenizer/model revision and selection rule are in the suite manifest. The full 39/11 pilot remains in `data/qwen-sft2-pilot/`.

Train and validation references are existing teacher-gated answers. The 10 held-out client references are inherited template drafts, not teacher-gated responses. The split and study/family/atomic identities are checked. Test is never tokenized during preparation or used to select an adapter. This is a plumbing pilot with draft references and very few validation cases, not client acceptance evidence.

The legacy input/metadata prompt versions are preserved because the teacher gate graded those exact inputs. The current SFT2 system prompt is recorded in the config and prepared manifest.

## 1. Environment

```bash
scripts/local/setup_env.sh
scripts/local/sft2.sh doctor
scripts/local/sft2.sh config
```

`setup_env.sh` creates `.venv-local/` and installs the delivery's pinned dependencies. The runner prefers that environment, otherwise uses the existing repository `.venv-sft-gpu/`. Override with `ODYN_PYTHON=/absolute/path/to/python`. The existing environment used for the first preparation/probes has bitsandbytes 0.48.2; the clean delivery profile pins 0.49.2. Re-run the memory probe after changing environments.

Qwen weights/tokenizer are already cached locally. On another machine, download the pinned model once:

```bash
scripts/local/sft2.sh download
```

## 2. Assemble and prepare

These stages have already been completed for the local profile. Both reject an existing destination to keep artifacts immutable; do not rerun them on the same paths.

```bash
scripts/local/sft2.sh dataset
scripts/local/sft2.sh prepare
```

Use `odyn-dataset synthesis assemble` (also called by `scripts/local/sft2.sh dataset`); its implementation lives in `odyn_sft.dataset_generation.synthesis_suite`. See [dataset commands](dataset-generation.md). It verifies the accepted teacher gate against every exported answer, verifies held-out source hashes, freezes provenance and validates split isolation. It performs no paid calls. Model-token filtering uses the locally cached pinned tokenizer.

For another suite or experiment, copy the config, give `source_suite`, `prepared`, `output` and `thermal_log` new paths, and set `ODYN_SFT_CONFIG=/absolute/path/to/config.json`. The assembler also accepts `--teacher-dir`, `--source-suite`, `--out`, `--test-limit`, `--token-config` and `--max-example-tokens` overrides. Without token filtering, it exports all supplied accepted teacher targets.

## 3. Confirm memory fit

```bash
scripts/local/sft2.sh probe --batch 1 --out runs/qwen-sft2-local-short/probe-batch1.json
```

This runs one discarded optimizer update on the longest prepared training example with matching precision, LoRA and optimizer settings. It never creates an adapter for reuse. An OOM exits nonzero; do not proceed to training unless the probe passes. A passing trial is evidence for batch-one memory capacity, not proof that the whole run will finish.

## 4. Train and observe

```bash
scripts/local/sft2.sh train
```

In another terminal:

```bash
scripts/local/sft2.sh tensorboard
```

Open `http://127.0.0.1:6006`. Training logs completion loss, validation completion loss/perplexity, gradient norms, learning rate, epoch progress and GPU telemetry. Quarter-epoch validation performs loss only; response generation is a separate stage to avoid repeated long generation passes. The TensorBoard command includes update, microbatch and epoch event directories.

The profile uses NF4 + double quantization, BF16 computation, ordinary gradient checkpointing, batch 1, accumulation maximum 8, rank 16 / alpha 32 / dropout 0.05, all-linear targets, AdamW8bit, LR 2e-4, cosine scheduling, 3% warmup, clipping 1.0 and up to 3 epochs. Packing is off. With 24 training records and quarter boundaries, actual update windows contain 6 examples: 4 updates per epoch, 12 total. Manifests record actual windows; the nominal accumulation limit is not the achieved batch size.

Temperature monitoring stops cooperatively at 90 C and saves a safe checkpoint. Partial windows are discarded without committing an optimizer update. Rerunning `train` with unchanged config/code/data resumes the saved state. Ctrl+C on the supervisor requests a safe worker stop; a new experiment should use new output paths. W&B is off by default; set its project/entity/name in a new config if desired, using environment credentials.

## 5. Compare generated responses

After a learned adapter has been selected:

```bash
scripts/local/sft2.sh validate
scripts/local/sft2.sh validate-base
```

Both use the same validation inputs and deterministic decoding settings. Reports contain loss/perplexity, generated responses, nonempty/truncated/empty diagnostics, runtime provenance and base/adapter comparisons. SFT2 does not execute prose as Python. These mechanical metrics do **not** establish groundedness or semantic correctness; that needs the separate calibrated judge.

## 6. Held-out check after completed training

```bash
scripts/local/sft2.sh test
```

The final-training gate must pass first. The inherited draft test references and small subset limit interpretation of test loss; no client >80% claim follows from this run.

## Artifacts

All paths below are relative to `final/sft/`.

| Path | Contents |
|---|---|
| `configs/qwen-sft2-local.json` | Complete experiment settings |
| `data/qwen-sft2-local/` | 24/7/10 synthesis suite, lineage, gates/source provenance and token-selection metadata |
| `runs/qwen-sft2-local-short/prepared/` | Completion-masked train/val arrays, pinned tokenizer and exact token statistics |
| `runs/qwen-sft2-local-short/probe*.json` | Discarded memory trials |
| `runs/qwen-sft2-local-short/gpu_metrics.jsonl` | Supervisor telemetry |
| `runs/qwen-sft2-local-short/train/` | Checkpoints, selected/final adapters, manifests, loss caches, status and TensorBoard events |
| `runs/qwen-sft2-local-short/train/generated_validation_sft2_adapter/` | Adapter responses and report |
| `runs/qwen-sft2-local-short/train/generated_validation_sft2_base/` | Base responses and report |
| `runs/qwen-sft2-local-short/train/final_evaluation/` | Gated held-out results |

The local run was subsequently launched as `odyn-qwen-sft2-local.service`. Consult the probe report, service status and live artifacts for current progress. Dataset commands do not start or interrupt training.
