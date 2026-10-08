"""Build and validate frozen judge rubrics. Implementation imports stay inside handlers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Register arguments and the handler without loading the workflow."""
    parser.set_defaults(handler=dispatch)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("build")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--pilot", type=Path, required=True)
    command.add_argument("--methodology", type=Path, required=True)
    command.add_argument("--use-cases", type=Path, required=True)
    command = commands.add_parser("validate")
    command.add_argument("--suite", type=Path, required=True)


def dispatch(args: argparse.Namespace) -> Any:
    """Execute this workflow after argument validation."""
    from .. import rubrics
    if args.action == "validate":
        return rubrics.validate_bundle(args.suite)
    return rubrics.prepare(
        args.out,
        pilot=args.pilot,
        methodology=args.methodology,
        use_cases=args.use_cases,
    )
