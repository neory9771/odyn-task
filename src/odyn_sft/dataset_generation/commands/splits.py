"""Plan, generate and validate paired dataset splits. Implementation imports stay inside handlers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ._shared import progress


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Register arguments and the handler without loading the workflow."""
    parser.set_defaults(handler=dispatch)
    commands = parser.add_subparsers(dest="action", required=True)
    command = commands.add_parser(
        "audit", help="Compute capacity and write a plan; does not generate samples"
    )
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser(
        "generate", help="Reproduce a reviewed, feasible audit plan"
    )
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument(
        "--acknowledge-plan",
        action="store_true",
        required=True,
        help="Confirm that the saved audit plan has been reviewed",
    )
    command = commands.add_parser(
        "validate", help="Check frozen package integrity and split isolation"
    )
    command.add_argument("--suite", type=Path, required=True)
    command = commands.add_parser(
        "validate-plan", help="Recompute and verify the frozen split plan"
    )
    command.add_argument("--plan", type=Path, required=True)


def dispatch(args: argparse.Namespace) -> Any:
    """Execute this workflow after argument validation."""
    if args.action == "validate":
        from ...survey_evaluation.package_integrity import validate_splits_package

        return validate_splits_package(args.suite)
    from .. import dataset
    if args.action == "audit":
        return dataset.audit_splits(args.config, args.out, progress)
    if args.action == "generate":
        return dataset.generate_splits(
            args.plan, args.out, acknowledge_plan=args.acknowledge_plan
        )
    return dataset.validate_split_plan(args.plan)
