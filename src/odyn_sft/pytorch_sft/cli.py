"""Internal workers for supervised LoRA training and evaluation.

Commands:
  prepare        Audit the source suite and tokenize train/validation completions.
                 Save token arrays and the tokenizer; do not tokenize test data.
  train          Fit LoRA adapters, or resume a compatible checkpoint. Validate
                 completion loss at each quarter epoch and select best_adapter.
  validate       Generate task outputs with the selected adapter on validation;
                 score them using the configured task adapter.
  validate-base  Run the same validation generation/scoring with base weights,
                 providing the comparison without a trained adapter.

Task is selected by config: sft1 generates Survey API Python; sft2 generates
research perspectives. Both share tokenization, completion loss and fitting.
  test           Require completed training and a verified learned adapter, then
                 evaluate task outputs on validation and the held-out test split.

Use the public ``odyn-sft`` entry point for GPU commands: it starts the monitor
and this worker with ODYN_GPU_SUPERVISED=1. Direct GPU worker calls without that
supervision are rejected. ``prepare`` does not require GPU supervision.

Example public command:
  odyn-sft train --config configs/mistral-bf16.json --disable-temperature-checks

Internal preparation command (equivalent to public ``odyn-sft prepare``):
  python -m odyn_sft.pytorch_sft.cli prepare --config configs/mistral-bf16.json
"""

from __future__ import annotations

import argparse
import json

from .config import load_config


def main() -> int:
    """Dispatch one worker command and print its result as JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=("prepare", "train", "test", "validate", "validate-base"),
        help="Worker operation; see the command descriptions above",
    )
    parser.add_argument(
        "--config", required=True, help="JSON config; relative paths resolve against its workspace"
    )
    parser.add_argument(
        "--interrupt-after-updates",
        type=int,
        help="Training only: save and stop at a global optimizer update (exit 75)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.interrupt_after_updates is not None and args.interrupt_after_updates < 1:
        parser.error("Use a positive interruption boundary")
    if args.command == "prepare":
        # Materialize completion-masked train/validation data in a new directory.
        from .data import prepare

        result = prepare(config)
    elif args.command == "test":
        # Final scoring requires completed training and a verified learned adapter.
        if args.interrupt_after_updates is not None:
            parser.error("Update interruption is supported only for training")
        from .evaluation.final import evaluate_final

        result = evaluate_final(config)
    elif args.command == "validate":
        # Score the selected SFT adapter on validation; test is not used here.
        if args.interrupt_after_updates is not None:
            parser.error("Update interruption is supported only for training")
        from .evaluation.validation import evaluate_selected_validation

        result = evaluate_selected_validation(config)
    elif args.command == "validate-base":
        # Establish the base-model comparison with the same validation protocol.
        if args.interrupt_after_updates is not None:
            parser.error("Update interruption is supported only for training")
        from .evaluation.validation import evaluate_base_validation

        result = evaluate_base_validation(config)
    else:
        # Fit/resume adapters; the optional stop boundary supports resume checks.
        from .training.loop import train

        result = train(config, args.interrupt_after_updates)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
