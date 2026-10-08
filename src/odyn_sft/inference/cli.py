"""Generate responses without loading gold targets, fitting or invoking an LLM API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="odyn-infer", description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Model/chat/precision SFT configuration",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL rows with instruction and input",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Generated JSONL; adjacent manifest and summary",
    )
    parser.add_argument(
        "--adapter", type=Path, help="Saved adapter directory; omit for the base model"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of input rows; default all"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, help="Override config generation budget"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue only if model/input/settings hashes match",
    )
    parser.add_argument(
        "--max-temperature", type=float, default=90.0, help="GPU stop temperature in C"
    )
    parser.add_argument(
        "--disable-temperature-checks",
        action="store_true",
        help="Disable GPU temperature stops",
    )
    args = parser.parse_args(argv)
    for name in ("limit", "max_new_tokens", "max_temperature"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    from .runner import run

    result = run(
        args.config,
        args.input,
        args.out,
        adapter=args.adapter,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
        max_temperature=args.max_temperature,
        disable_temperature_checks=args.disable_temperature_checks,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["state"] == "completed" else 1
