# odyn-survey-sft

This package trains and evaluates the two survey-research models:

- **SFT1:** claim → Survey API Python program. Graded by executing the program.
- **SFT2:** claim + executed evidence → research perspective. Graded by code checks plus an LLM judge.

Both use the same single-GPU LoRA trainer on a pinned base model, with a separate adapter per task.

**Start with [WALKTHROUGH.md](WALKTHROUGH.md).** It explains the code, behaviour and every step, file by file.
The Python analysis library is documented in the [Survey API guide](docs/survey-api.md).
Definitions of claims, contracts, evidence and evaluation terms are in the
[glossary](docs/glossary.md).
The [prompt map](src/odyn_sft/prompts/README.md) identifies the SFT1, SFT2, judge and dataset-generation prompts and explains how each model input is assembled.

The strategy is **prepare claims and reference contracts → train SFT1/SFT2 →
select on validation → evaluate the frozen pipeline on test**. `odyn-dataset`
handles preparation; `odyn-sft` fits adapters; `odyn-infer`, `odyn-judge` and
`odyn-e2e` generate and assess their outputs. Existing datasets can be validated
and reused without replaying generation. See [dataset strategy](docs/dataset-generation.md#where-the-dataset-cli-fits)
for the purpose of each preparation command.

## Install

Linux, Python 3.11/3.12, one NVIDIA GPU with BF16 support.

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install '.[dataset,tracking,judge,test]'
odyn-sft doctor            # nonzero exit if the GPU environment is incomplete
python -m pytest -q        # CPU tests; no GPU, downloads or API calls
```

## Run

Pick a config (see [docs/mistral.md](docs/mistral.md) for the profile matrix):

| Config | Task | Base |
|---|---|---|
| `configs/mistral-bf16.json` / `mistral-nf4.json` | SFT1 | Mistral-Small-24B, BF16 LoRA / NF4 QLoRA |
| `configs/mistral-sft2-bf16.json` / `mistral-sft2-nf4.json` | SFT2 | Mistral-Small-24B |
| `configs/mistral-sft2-bf16-h200.json` | SFT2 | Mistral-Small-24B, 1-epoch H200 profile |
| `configs/qwen-nf4.json`, `configs/qwen-sft2-local.json` | SFT1 / SFT2 | Qwen3-4B |

```bash
CONFIG=configs/mistral-bf16.json
odyn-sft download --config $CONFIG                  # pinned model snapshot into the HF cache
odyn-sft prepare  --config $CONFIG                  # audit the suite, tokenize train/validation
odyn-sft probe    --config $CONFIG --batch 1 --out runs/probe.json --disable-temperature-checks
odyn-sft train    --config $CONFIG --disable-temperature-checks
odyn-sft validate-base --config $CONFIG --disable-temperature-checks
odyn-sft validate      --config $CONFIG --disable-temperature-checks
odyn-sft test          --config $CONFIG --disable-temperature-checks   # only after training completes
```

- `train` resumes automatically from the last checkpoint if the config, data and code are unchanged.
- Ctrl-C or SIGTERM saves a checkpoint and exits with code 75.
- Without `--disable-temperature-checks`, training stops cooperatively at 90 °C.
- TensorBoard logs are written under the run's `output` directory. W&B is used when `wandb_project` is set.

Other commands:

| Task | Command | Guide |
|---|---|---|
| Build or validate datasets | `odyn-dataset ...` | [docs/dataset-generation.md](docs/dataset-generation.md) |
| Generate from JSONL (base or adapter) | `odyn-infer ...` | [docs/inference.md](docs/inference.md) |
| Grade SFT2 answers | `odyn-judge ...` | [docs/judging.md](docs/judging.md) |
| Local Qwen SFT2 pilot | `scripts/local/sft2.sh` | [docs/local-qwen-sft2.md](docs/local-qwen-sft2.md) |

## Data

Each config points `source_suite` at a dataset suite directory. A suite is a folder with:
- `manifest.json` (file hashes, split counts, release status);
- Alpaca-style records (`instruction`, `input`, `output`);
- a lineage file per split;
- study data and private gold.

The trainer accepts these formats:

| Task | Accepted suite formats |
|---|---|
| SFT1 | `synthetic-client-sft-1` (synthetic train/validation, client test), `synthetic-smallest100-sft-1`, frozen paired packages from `odyn-dataset splits` |
| SFT2 | `synthesis-sft-1` from `odyn-dataset synthesis assemble`, or a frozen paired package |

Preparation rejects:
- changed hashes;
- a record, family or study that appears in two splits;
- an example longer than `max_length` (never truncated);
- a draft suite, unless `allow_provisional` is set.

The test split is never used for fitting or checkpoint selection.

### Study bindings

Survey-specific group definitions live in JSON, not code. The default is
`dataset_generation/resources/survey_2022_bindings.json`. A generator config can set
`study_bindings` to a file path or an inline object:

```json
{"version": "survey-bindings-1", "study_id": "my-study",
 "groups": [{"variable": "segment", "groups": {"Group A": ["A"], "Group B": ["B"]}}]}
```

Groups must be disjoint. The refusal value `Prefer not to say` is rejected.

## Run artifacts

| Location (under `output`) | Contents |
|---|---|
| `resolved_config.json`, `run_manifest.json` | Effective settings, model revision, data and code hashes |
| `checkpoint-*` | Adapter, optimizer, scheduler, RNG and sampler position |
| `best_adapter/` + `selection.json` | Lowest validation-loss adapter that has learned |
| `adapter/`, `training_summary.json` | Final adapter and acceptance checks |
| `validation/<tag>/` | Per-quarter validation loss |
| `generated_validation_*`, `final_evaluation/` | Generated-output reports (validate / test) |
| `tensorboard*/`, `live_status.json` | Metrics and live progress |

## Scope and limits

- Single GPU only. No packing, no truncation, at most 3 epochs.
- Generated-output scores are diagnostics. Checkpoint selection uses validation loss only.
- SFT2 quality needs `odyn-judge`, and the judge must first pass its validation gate.
- Data and model weights are not bundled.
- A code change alters the run fingerprint, so start a fresh run rather than resuming an old one.
- Equivalence with the original research code is recorded in [VERIFICATION.md](VERIFICATION.md).
