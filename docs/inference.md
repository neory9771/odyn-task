# JSONL inference

Two commands generate with the trained adapters:

| Command | Runs | Use it for |
|---|---|---|
| `odyn-infer` | One model (base, SFT1 **or** SFT2) on one JSONL file; no scoring | Single-stage base-vs-adapter comparisons |
| `odyn-e2e` | SFT1 → execute program → SFT2 → judge, on held-out claims; each step base or adapter; HF or vLLM engine | The real pipeline; see [WALKTHROUGH §6.5](../WALKTHROUGH.md) |

`odyn-infer` (or `python -m odyn_sft.inference`) accepts JSONL rows containing
`instruction` and `input`. Optional `metadata.record_id` is preserved.
An `output` field may be present for dataset compatibility, but is never sent
to the model. This command loads no evaluation gold or dataset generators.

Use the same model/chat config as training. Model weights and tokenizer must
already be cached at the configured pinned revision. Omit `--adapter` for the
base model; supply a saved adapter directory for fine-tuned inference.

```bash
odyn-infer --config configs/qwen-sft2-local.json \
  --input data/qwen-sft2-local/validation_synthesis.jsonl \
  --out runs/inference-base.jsonl

odyn-infer --config configs/qwen-sft2-local.json \
  --input data/qwen-sft2-local/validation_synthesis.jsonl \
  --adapter runs/qwen-sft2-local-short/train/best_adapter \
  --out runs/inference-sft2.jsonl
```

The same command runs SFT1 with an SFT1 config and adapter. Its output is the raw
program text: `odyn-infer` does not execute or score it (`odyn-sft test` and
`odyn-e2e` do).

```bash
odyn-infer --config configs/qwen-nf4.json \
  --input data/suite/validation_analysis.jsonl \
  --adapter runs/qwen-sft1/train/best_adapter \
  --out runs/inference-sft1.jsonl
```

`--limit N` caps the number of rows. `--max-new-tokens N` overrides the configured
generation budget. Greedy decoding and batch size one keep comparison settings
explicit. Total prompt plus generation length stays within `max_length`; prompts
are never truncated. NF4/BF16 base loading follows the config; saved LoRA weights
are cast to BF16 for inference, as in the existing deployment evaluator.

Each result has the completion, source index/record ID, model variant, token
counts, generation time/throughput, finish reason and status. `model_variant` is
`base`, `sft1` or `sft2` (the config's task). Status is
`successful`, `truncated`, `empty_output` or `runtime_error`. A successful
generation does not establish factual correctness or evidence grounding.
There are no external LLM calls or semantic judge calls.

Adjacent `.manifest.json` and `.summary.json` files record model/input/adapter
hashes, decoding settings and aggregate outcomes. Rows are flushed and synced
after each completion. Existing outputs require `--resume`; resume verifies
the original inputs, model, adapter, implementation and settings. A last row left
half-written by a crash is dropped and regenerated.

GPU temperature is checked during decoding every five seconds. The default
limit is 90 C; `--max-temperature` changes it, and
`--disable-temperature-checks` disables it explicitly. A thermal interruption
keeps completed rows and allows a later resume; the unfinished response is
regenerated. Temperature checks do not automatically restart inference.

From the delivery folder, run both variants sequentially on the same seven
local validation examples with the existing workspace environment:

```bash
scripts/local/infer_sft2_pair.sh
# Or supply a different input, new output directory and adapter:
scripts/local/infer_sft2_pair.sh INPUT.jsonl OUTPUT_DIR ADAPTER_DIR
```

This writes `base.jsonl`, `sft2.jsonl`, their manifests/summaries, and
`comparison.json` with paired responses (adapter side labelled by its task). The comparison verifies matching inputs
and decoding settings; it does not grade research quality. `ODYN_PYTHON` and
`ODYN_SFT_CONFIG` override the interpreter/config.

After generation, use the separate [judge command](judging.md) to evaluate
support, contradiction, uncertainty, gaps and evidence grounding.
