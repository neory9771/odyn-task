"""Audit and render client evaluation candidates. Implementation imports stay inside handlers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


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


def dispatch(args: argparse.Namespace) -> Any:
    """Execute this workflow after argument validation."""
    from .. import generation
    if args.action == "generate":
        return generation.generate(
            args.plan, args.out, acknowledge_plan=args.acknowledge_plan
        )
    from ..config import load_config
    config = load_config(args.config)
    return generation.audit(
        source=config["source"],
        metadata_path=config["metadata"],
        output=args.out,
        size=config["size"],
        calibration_size=config["calibration_size"],
        seed=config["seed"],
        overlaps=config["overlap_corpus"],
        coverage_minima=config["coverage_minima"],
        interaction_minima=config["interaction_minima"],
        heldout_blocks=config["heldout_outcome_blocks"],
        scenario_rules=config["scenario_rules"],
        study_bindings=config["study_bindings"],
    )
