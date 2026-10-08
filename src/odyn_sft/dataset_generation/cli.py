"""Unified entry point for dataset preparation.

Each workflow owns its arguments and handler in commands/ (or synthetic/cli.py).
Help imports only lightweight CLI modules. Implementations load on dispatch.
Training and generated-answer evaluation use separate CLIs.
See docs/dataset-generation.md for workflow roles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .commands import evaluation, metadata, rubrics, splits, synthesis, use_cases, variants
from .synthetic import cli as synthetic


def build_parser() -> argparse.ArgumentParser:
    """Register the workflow CLIs without importing their implementations."""
    parser = argparse.ArgumentParser(
        prog="odyn-dataset",
        description="Prepare claims, reference contracts and datasets for SFT1 analysis and SFT2 responses. Training and held-out judging use separate CLIs.",
    )
    groups = parser.add_subparsers(dest="domain", required=True)
    for name, module, description in (
        ("synthetic", synthetic, "Generate seeded surveys and paired SFT curricula; no OpenAI calls"),
        ("metadata", metadata, "Build study metadata from downloaded survey files"),
        ("evaluation", evaluation, "Audit and generate client evaluation candidates"),
        ("splits", splits, "Audit, generate and validate paired train/validation/test packages"),
        ("synthesis", synthesis, "Write, assemble and validate SFT2 targets"),
        ("variants", variants, "Paraphrase claims with fixed meaning and add them to a package (OpenAI)"),
        ("use-cases", use_cases, "Build the controlled Survey API coverage fixture"),
        ("rubrics", rubrics, "Build and validate frozen judge contracts"),
    ):
        workflow = groups.add_parser(name, help=description)
        module.configure_parser(workflow)
    return parser


def _dispatch(args: argparse.Namespace) -> Any:
    """Invoke the handler registered by the selected workflow."""
    return args.handler(args)


def main(argv: list[str] | None = None) -> int:
    """Run one stage and print a compact report; full details remain in artifacts."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if validator := getattr(args, "validate_arguments", None):
        validator(parser, args)
    try:
        result = _dispatch(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"odyn-dataset: {exc}\n")
    report: dict[str, Any] = {"command": f"{args.domain} {args.action}"}
    path = (
        getattr(args, "out", None)
        or getattr(args, "suite", None)
        or getattr(args, "plan", None)
        or getattr(args, "output", None)
        or getattr(args, "run", None)
    )
    if path is not None:
        report["path"] = str(Path(path).resolve())
    if isinstance(result, dict):
        # Plans contain large pools and configurations: leave those in saved JSON.
        for key in (
            "version",
            "feasible",
            "counts",
            "target_policy",
            "integrity_valid",
            "case_count",
            "suite_counts",
            "metadata_written",
            "step1_status",
            "gate_passed",
            "records",
            "accepted",
            "by_split",
            "paraphrases",
            "skipped_unparsed_programs",
            "validation",
            "cases",
            "llm_calls",
            "passed",
            "paired_records",
            "studies",
            "study_id",
            "respondents",
            "variables",
        ):
            if key in result:
                report[key] = result[key]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
