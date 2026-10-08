# Mistral SFT2: existing research examples

Use SFT2 to learn evidence-conditioned research responses. SFT1 already has a
trained Mistral adapter; its BF16 run stopped at update 112/192, rather than
finishing the planned three epochs. This preparation does not resume SFT1 or
replay the pipeline. SFT2 is a separate adapter starting from the base model.

The existing 5k/1k package contains 500 train and 100 validation API tutorials.
Exclude records tagged `task_type=api_tutorial` from both fitting and validation.
Keep research claims even when their evidence was produced by API calls.

| Split | Original | Within 32k | Research only | Selected |
|---|---:|---:|---:|---:|
| Train | 5,000 | 4,880 | 4,380 | 1,000 |
| Validation | 1,000 | 998 | 898 | 898 |
| Test | 1,000 | Not filtered | Not filtered | Unchanged |

Train selection preserves proportions across the existing use-case and `P`
complexity strata, with seeded SHA-256 ranking within each stratum (seed 42).
It covers all 16 research use cases and 110 synthetic studies: 149 low, 676
medium and 175 high complexity records. No complexity is recomputed.

Inputs, reference executions and targets are reused unchanged. Targets remain
deterministic evidence summaries, not newly written perspectives. They are
provisional references pending expert review. No generation or OpenAI calls.
The inherited exact Mistral 32,768-token filter excludes whole examples; it
does not truncate them. Package validation verifies hashes and disjoint record,
family, study and atomic component identities across splits. Test artifacts are
copied byte-for-byte; they are neither selected on nor used for fitting. The
unchanged test still contains its original API tutorial records, so a future
SFT2 test report should distinguish research-only results from the full suite.

Local package: `final/sft/data/synthesis-research-1000-under32k/`.
Remote package: `/dev/shm/odyn-mistral-sft2-research-1000/suite/`.
The local copy protects these artifacts from remote shared-memory loss.

Reproduce the subset on the remote machine with the final code:

```bash
PYTHONPATH=/dev/shm/odyn-e2e/eval-sft/src \
/workspace/envs/hf/bin/python /dev/shm/odyn-e2e/subset_sft2_research.py \
  --source-suite /workspace/odyn-sft2-5000/sft/data/synthesis-suite-5000-under32k \
  --out /dev/shm/odyn-mistral-sft2-research-1000/suite \
  --train-count 1000 --seed 42
```

Use a fresh output directory when repeating. The reusable script is
`scripts/h200/subset_sft2_research.py`. The prepared training configuration is
`configs/mistral-sft2-bf16-research-1000-h200.json`: BF16 LoRA, rank 16,
alpha 32, dropout 0.05, all linear layers, LR 2e-4, cosine schedule with 3%
warmup, one epoch, microbatch two and accumulation 16. Probe actual memory
before launching, use `--disable-temperature-checks` on H200, and enable W&B
with the configured project. No training was launched during the dataset check.

## Remote launch (2026-10-08)

At the user's subsequent request, launched the fresh SFT2 workflow under
Supervisor service `odyn-mistral-sft2-research-1000`. The launcher prepares the
existing subset, probes microbatch two (falls back to one), and then trains.
Temperature checks are disabled for both the probe and training.

Frozen code/config snapshot:
`/dev/shm/odyn-mistral-sft2-research-1000/sft/`.
Log: `/dev/shm/odyn-mistral-sft2-research-1000/train.log`.
Outputs: `sft/runs/mistral-sft2-research-1000/train/` under that remote root.

```bash
ssh odyn-h200 'supervisorctl status odyn-mistral-sft2-research-1000'
ssh odyn-h200 'tail -f /dev/shm/odyn-mistral-sft2-research-1000/train.log'
```

The script and Supervisor configuration are saved locally in `scripts/h200/`.
The new run does not continue either the SFT1 adapter or previous SFT2 pilots.

### Corrected to one epoch

The user reduced the budget from three epochs to one. The initial run was
checkpoint-stopped at update 1 (32/1,000 examples), with all 280 LoRA B tensors
nonzero. Its immutable manifest and checkpoint remain intact. The continuation
uses `sft-oneepoch/one-epoch-config.json` and starts from that checkpoint: Adam
moments, trained weights, sampler position and RNG are restored. LambdaLR is
reconciled at the restored step against the new 32-update horizon (rather than
96); earlier updates are not replayed. Two continuation tests passed remotely,
and preflight verified the actual parent hashes and changed epoch setting.

Current code/output root: `/dev/shm/odyn-mistral-sft2-research-1000/sft-oneepoch/`.
Current log: `/dev/shm/odyn-mistral-sft2-research-1000/train-oneepoch.log`.
Supervisor service name is unchanged. The audited continuation configuration is
also saved locally in `final/evidence/mistral-sft2-research-1000/`.
