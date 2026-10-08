"""Build metadata from supplied study files. Implementation imports stay inside handlers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Register arguments and the handler without loading the workflow."""
    parser.set_defaults(handler=dispatch)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser("build", help="Write metadata JSON to a new file")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)


def dispatch(args: argparse.Namespace) -> Any:
    """Execute this workflow after argument validation."""
    from ...common.storage import write_json
    from ..metadata import build_metadata
    if args.out.exists():
        raise ValueError("Use a new metadata output file")
    result = build_metadata(args.source)
    write_json(args.out, result)
    return {"metadata_written": True}
