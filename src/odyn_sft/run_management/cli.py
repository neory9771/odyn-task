"""Public commands for a portable, single-GPU evidence-analysis and response-synthesis SFT run."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from ..pytorch_sft.config import load_config

GPU_COMMANDS = ("train", "validate", "validate-base", "test", "probe")


def doctor() -> dict[str, Any]:
    """Report the installed environment without downloading or loading weights."""
    import torch

    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "tensorboard"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    count = torch.cuda.device_count()
    ready = count == 1 and torch.cuda.is_bf16_supported() and all(versions.values())
    return {"versions": versions, "cuda_version": torch.version.cuda,
            "visible_gpu_count": count, "single_gpu_training_ready": ready,
            "gpu": torch.cuda.get_device_name(0) if count else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "config", "download", "prepare", *GPU_COMMANDS))
    parser.add_argument("--config", help="JSON configuration; paths resolve relative to its workspace")
    parser.add_argument("--disable-temperature-checks", action="store_true",
                        help="Disable temperature/sensor stops while retaining GPU telemetry")
    parser.add_argument("--max-temperature", type=float, default=90.0)
    parser.add_argument("--interrupt-after-updates", type=int)
    parser.add_argument("--batch", type=int, help="Probe batch size, tested on the longest training examples")
    parser.add_argument("--out", help="Probe report JSON path")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        result = doctor()
        print(json.dumps(result, indent=2))
        return 0 if result["single_gpu_training_ready"] else 1
    if not args.config:
        parser.error("--config is required for this command")
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.interrupt_after_updates is not None:
        if args.command != "train" or args.interrupt_after_updates < 1:
            parser.error("--interrupt-after-updates requires train and a positive update count")
    if args.command == "probe" and (not args.batch or args.batch < 1 or not args.out):
        parser.error("probe requires --batch >= 1 and --out PATH")
    if args.command != "probe" and (args.batch is not None or args.out is not None):
        parser.error("--batch and --out are only used by probe")
    if args.max_temperature <= 0:
        parser.error("--max-temperature must be positive")
    if args.command == "config":
        result = config.as_dict()
    elif args.command == "download":
        from huggingface_hub import snapshot_download
        # The trainer loads cached, pinned model weights; authentication is supplied externally.
        result = {"snapshot": snapshot_download(repo_id=config.model, revision=config.revision)}
    elif args.command == "prepare":
        from ..pytorch_sft.data import prepare
        result = prepare(config)
    else:
        from .supervisor import run_supervised
        return run_supervised(args.command, config_path, config,
                              disable_temperature_checks=args.disable_temperature_checks,
                              max_temperature=args.max_temperature,
                              interrupt_after_updates=args.interrupt_after_updates,
                              batch=args.batch, out=args.out)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
