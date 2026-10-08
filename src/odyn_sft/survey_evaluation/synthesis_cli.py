"""Evaluate saved base/SFT2 responses with the previously validated OpenAI checklist judge."""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="odyn-judge", description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument(
        "--credentials-file",
        type=Path,
        help="Optional private JSON with OPENAI_API_KEY; otherwise environment",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--request-cache",
        type=Path,
        help="Optional existing completed request ledgers; fingerprints must match",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate context/gate and count calls without submitting",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    from .synthesis_runner import run

    result = run(
        args.input,
        args.base,
        args.sft,
        args.out,
        args.validation_report,
        credentials_file=args.credentials_file,
        workers=args.workers,
        dry_run=args.dry_run,
        request_cache=args.request_cache,
    )
    print(json.dumps(result, indent=2))
    return (
        1
        if any(
            v["invalid_or_error_judgments"] for v in result.get("variants", {}).values()
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
