"""Generate verified claim wording variants and assemble them. Implementation imports stay inside handlers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ._shared import positive


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Register arguments and the handler without loading the workflow."""
    parser.set_defaults(handler=dispatch)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("validate-verifier", "Gate: planted meaning changes must be rejected, untouched claims accepted"),
        ("generate", "Write and verify one paraphrase per claim"),
        ("apply", "Write a new package with template plus paraphrase rows"),
    ):
        command = commands.add_parser(action, help=help_text)
        command.add_argument("--package", type=Path, required=True, help="Source paired package")
        command.add_argument("--work", type=Path, required=True, help="Gate, variants and ledgers")
        if action == "apply":
            command.add_argument("--out", type=Path, required=True)
            continue
        command.add_argument("--workers", type=positive, default=8)
        command.add_argument("--credentials-file", type=Path, help="Private JSON with OPENAI_API_KEY")
        if action == "validate-verifier":
            command.add_argument("--originals", type=positive, default=40)
            command.add_argument("--per-type", type=positive, default=10)
            command.add_argument("--seed", type=int, default=0)
        else:
            command.add_argument("--max-attempts", type=positive, default=1)


def dispatch(args: argparse.Namespace) -> Any:
    """Execute this workflow after argument validation."""
    from ..claim_variants import pipeline
    if args.action == "apply":
        return pipeline.apply(args.package, args.work, args.out)
    records, skipped = pipeline.load_records(args.package)
    if args.action == "validate-verifier":
        from ...common.storage import read_json

        variables = read_json(args.package / "metadata.json")["variables"]
        result = pipeline.validate_verifier(
            records, variables, args.work, originals=args.originals, per_type=args.per_type,
            workers=args.workers, seed=args.seed, credentials=args.credentials_file)
    else:
        result = pipeline.generate(records, args.work, workers=args.workers,
                                   max_attempts=args.max_attempts, credentials=args.credentials_file)
    return {**result, "skipped_unparsed_programs": len(skipped)}
