# Mistral SFT1 and SFT2

Both stages use the delivered shared PyTorch trainer and the pinned
`mistralai/Mistral-Small-24B-Instruct-2501` base model. They train separate LoRA
adapters from the base: SFT1 generates Survey API code; SFT2 interprets executed
evidence. SFT2 does not continue training the SFT1 adapter.

| Stage | BF16 LoRA | 4-bit QLoRA | Dataset suite |
|---|---|---|---|
| SFT1 | [mistral-bf16.json](../configs/mistral-bf16.json) | [mistral-nf4.json](../configs/mistral-nf4.json) | `data/suite` |
| SFT2 | [mistral-sft2-bf16.json](../configs/mistral-sft2-bf16.json) | [mistral-sft2-nf4.json](../configs/mistral-sft2-nf4.json) | `data/synthesis-suite` |

Each profile has an explicit task-specific system prompt and isolated prepared
data, adapters and metrics. BF16 uses AdamW; NF4 uses AdamW8bit. Defaults are
three epochs, LR 2e-4, rank 16, alpha 32, dropout 0.05, all-linear targets,
gradient checkpointing, batch one with accumulation 16, and a 32k context cap.
Validation loss is measured at 25%, 50%, 75% and 100% of each epoch.

From `final/sft`, after installing the package and supplying the complete suite:

```bash
# Select one profile; change this path to run the other stage or precision.
CONFIG=configs/mistral-sft2-bf16.json
odyn-sft config --config "$CONFIG"
odyn-sft download --config "$CONFIG"
odyn-sft prepare --config "$CONFIG"
odyn-sft probe --config "$CONFIG" --batch 2 --out runs/mistral-probe.json --disable-temperature-checks
# Set microbatch_size in the config to a size verified by the probe.
odyn-sft train --config "$CONFIG" --disable-temperature-checks
odyn-sft validate-base --config "$CONFIG" --disable-temperature-checks
odyn-sft validate --config "$CONFIG" --disable-temperature-checks
odyn-sft test --config "$CONFIG" --disable-temperature-checks
```

The explicit temperature flag is intended for the managed remote GPU. Omit it
to retain temperature stops. The probe runs a discarded training trial; it does
not change the configured batch size. Preparation rejects overlength examples
rather than truncating them. Final test requires completed training.

For standalone inference, use `odyn-infer --config "$CONFIG" --input INPUT.jsonl
--out RESULTS.jsonl`; add `--adapter PATH_TO_BEST_ADAPTER` for the trained model.
Keep the same profile and input when comparing base and adapter. See
[inference.md](inference.md) for saved-output details and
[judging.md](judging.md) for SFT2 semantic evaluation. SFT1 validation already
executes generated code and scores its evidence contract.

The profiles and commands are included; model weights and client datasets are
external assets. Adding these profiles does not start training or establish a
Mistral quality result.
