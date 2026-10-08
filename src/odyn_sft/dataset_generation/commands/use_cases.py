"""Build the controlled Survey API coverage fixture. Implementation imports stay inside handlers."""
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
    command.add_argument(
        "--source", type=Path, required=True, help="Existing pilot suite"
    )
    command.add_argument("--methodology", type=Path, required=True)


def dispatch(args: argparse.Namespace) -> Any:
    """Execute this workflow after argument validation."""
    from ..use_cases import prepare
    return prepare(args.out, source=args.source, methodology=args.methodology)
