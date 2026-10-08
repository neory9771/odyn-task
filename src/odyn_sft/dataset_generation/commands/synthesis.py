"""Export, write, assemble and validate SFT2 targets. Implementation imports stay inside handlers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ._shared import positive


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Register arguments and the handler without loading the workflow."""
    parser.set_defaults(handler=dispatch)
    parser.set_defaults(validate_arguments=validate_arguments)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("export", help="Export all unchanged deterministic targets from a paired suite")
    command.add_argument("--source-suite", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser(
        "write-targets", help="Write gated targets for a package's train/validation rows (OpenAI)"
    )
    command.add_argument("--package", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--workers", type=positive, default=8)
    command.add_argument("--credentials-file", type=Path, help="Private JSON with OPENAI_API_KEY")
    command = commands.add_parser(
        "assemble", help="Package accepted written targets; held-out references remain draft"
    )
    command.add_argument("--targets-dir", type=Path, required=True)
    command.add_argument("--source-suite", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--test-limit", type=positive, help="Default: every held-out record")
    command.add_argument(
        "--token-config",
        type=Path,
        help="SFT2 model/chat config for optional exact-token filtering",
    )
    command.add_argument(
        "--max-example-tokens",
        type=positive,
        help="Keep whole train/validation examples within this cap; never truncate",
    )
    command = commands.add_parser("validate", help="Validate an assembled SFT2 suite")
    command.add_argument("--suite", type=Path, required=True)


def dispatch(args: argparse.Namespace) -> Any:
    """Execute this workflow after argument validation."""
    if args.action == "export":
        from ..synthesis_view import export

        return export(args.source_suite, args.out)
    if args.action == "validate":
        from ...tasks import get_task

        get_task("sft2").validate_suite(args.suite)
        return {"integrity_valid": True}
    if args.action == "write-targets":
        from ..synthesis_targets.pipeline import write_targets

        return write_targets(args.package, args.out, workers=args.workers,
                             credentials=args.credentials_file)
    from ..synthesis_suite import assemble
    return assemble(
        args.targets_dir,
        args.source_suite,
        args.out,
        test_limit=args.test_limit,
        token_config=args.token_config,
        max_example_tokens=args.max_example_tokens,
    )


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject incomplete token-filter settings before starting the workflow."""
    if args.action == "assemble":
        if args.max_example_tokens is not None and args.token_config is None:
            parser.error("--max-example-tokens requires --token-config")
