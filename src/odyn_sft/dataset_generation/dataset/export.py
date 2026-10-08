"""Generate the package: reference programs, gold, lineage, Alpaca views, diversity and manifest."""

from __future__ import annotations

import ast
import copy
import json
import shutil
from collections import Counter
from pathlib import Path

from ...common.storage import (
    atomic_text,
    digest,
    file_hash,
    read_json,
    write_json,
    write_jsonl,
)
from ...prompts.survey import (
    ANALYSIS_INSTRUCTION,
    PROMPT_VERSION,
    SYNTHESIS_INSTRUCTION,
    analysis_input,
    synthesis_input,
)
from ...survey_api import run_reference
from ...survey_evaluation.package_integrity import SPLITS, VERSION
from ...survey_evaluation.study_data import load_inputs
from ...survey_metadata.catalogue import numeric_program, numeric_references
from ..contracts import analysis_contract, synthesis_contract
from ..generation import (
    lexical_audit,
    materialize,
    reference_perspective,
    validate_reference,
)
from ..provenance import implementation_hashes
from ..types import DataObject, PathLike
from .families import training_claim
from .planning import build_selection


def generate_splits(
    plan_path: PathLike, output: PathLike, acknowledge_plan: bool = False, progress=None
) -> DataObject:
    if not acknowledge_plan:
        raise ValueError("Review the split plan and pass --acknowledge-plan")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    expected = read_json(plan_path)
    if (
        expected["version"] != VERSION
        or expected["implementation_hashes"] != implementation_hashes()
    ):
        raise ValueError("Implementation changed; repeat split audit")
    if not expected["feasible"]:
        raise ValueError("Infeasible split plan")
    if (
        file_hash(expected["config"]["source"]) != expected["source_sha256"]
        or file_hash(expected["config"]["metadata"]) != expected["metadata_sha256"]
    ):
        raise ValueError("Source/metadata changed; repeat split audit")
    actual, metadata, selections = build_selection(expected["config"], progress)
    if digest(actual) != digest(expected):
        raise ValueError("Frozen split selection/inputs changed")
    data, _ = load_inputs(actual["config"]["source"], actual["config"]["metadata"])
    gold = []
    for split in SPLITS:
        for candidate in selections[split]:
            case = materialize(candidate, split, actual["config"]["seed"], metadata)
            case["evidence"] = run_reference(case["program"], data, metadata)
            validate_reference(case, case["evidence"])
            case["analysis_contract"] = analysis_contract(case)
            case["synthesis_contract"] = synthesis_contract(case)
            case["reference_perspective"] = reference_perspective(case).replace(
                "observed unweighted 2022 analysis bases",
                "observed analysis bases of the supplied study",
            )
            case["reference_language_status"] = "template_pending_expert_review"
            variants = actual["config"]["train_variants"] if split == "train" else 1
            for variant in range(variants):
                record = copy.deepcopy(case)
                record["claim"] = training_claim(record, variant)
                record["language_style"] = (
                    "primary"
                    if variant == 0
                    else [
                        "",
                        "business",
                        "informal",
                        "formal",
                        "presentation",
                        "working_interpretation",
                    ][variant]
                )
                record["record_id"] = record["case_id"] + f".v{variant}"
                record["variant"] = variant
                gold.append(record)
    # No files published until all selected reference programs have passed.
    output.mkdir(parents=True)
    write_json(output / "plan.json", actual)
    # Preserve frozen byte hashes, not just JSON semantic equivalence.
    shutil.copy2(actual["config"]["metadata"], output / "metadata.json")
    shutil.copy2(actual["config"]["source"], output / "respondents.parquet")
    write_jsonl(output / "private_gold.jsonl", gold)
    export_alpaca(output, gold, metadata)
    diversity = {
        split: diversity_report([c for c in gold if c["split"] == split]) for split in SPLITS
    }
    write_json(output / "diversity.json", diversity)
    atomic_text(
        output / "README.md",
        "# Two-level draft datasets\n\nAnalysis and synthesis are paired Alpaca views of one frozen family partition.\nTest outputs and private_gold.jsonl are private scoring material.\nSynthesis inputs contain reference execution and measure oracle-conditioned ability; use the runtime for candidate-conditioned end-to-end evaluation.\nTemplate reference prose requires expert/language review. No model evaluation, training or paid calls performed by preparation.\n",
    )
    write_manifest(output)
    return {
        "families": {s: len(selections[s]) for s in SPLITS},
        "records": {s: sum(c["split"] == s for c in gold) for s in SPLITS},
        "llm_calls": 0,
        "status": "draft_pending_language_and_expert_review",
    }


def export_alpaca(output: Path, gold: list[DataObject], metadata: DataObject) -> None:
    """Strict three-field Alpaca records; identities/lineage live in sidecars."""
    references = numeric_references(metadata["variables"])
    write_json(output / "numeric_references.json", references)
    for split in SPLITS:
        records = [c for c in gold if c["split"] == split]
        analysis = []
        synthesis = []
        lineage = []
        for c in records:
            analysis.append(
                {
                    "instruction": ANALYSIS_INSTRUCTION,
                    "input": analysis_input(c["claim"], metadata),
                    "output": numeric_program(c["program"], references),
                }
            )
            synthesis.append(
                {
                    "instruction": SYNTHESIS_INSTRUCTION,
                    "input": synthesis_input(c["claim"], c["program"], c["evidence"], metadata),
                    "output": c["reference_perspective"],
                }
            )
            lineage.append(
                {
                    "record_id": c["record_id"],
                    "family_id": c["family_id"],
                    "component_keys": c["component_keys"],
                    "variant": c["variant"],
                }
            )
        suffix = "_gold" if split == "test" else ""
        write_jsonl(output / (split + "_analysis" + suffix + ".jsonl"), analysis)
        write_jsonl(output / (split + "_synthesis" + suffix + ".jsonl"), synthesis)
        write_jsonl(output / (split + "_lineage.jsonl"), lineage)
        if split == "test":
            write_jsonl(
                output / "test_analysis_inputs.jsonl",
                [{k: v for k, v in r.items() if k != "output"} for r in analysis],
            )
            write_jsonl(
                output / "test_synthesis_oracle_inputs.jsonl",
                [{k: v for k, v in r.items() if k != "output"} for r in synthesis],
            )


def diversity_report(cases: list[DataObject]) -> DataObject:
    # Score/audit one primary variant per family; training rows are not new cases.
    primary = [c for c in cases if c["variant"] == 0]
    return {
        "families": len(primary),
        "rows": len(cases),
        "dimensions": {
            d: dict(
                Counter(
                    v for c in primary for v in (set(c[d]) if isinstance(c[d], list) else [c[d]])
                )
            )
            for d in ("C", "D", "A", "E", "R", "S")
        },
        "outcome_blocks": dict(Counter(b for c in primary for b in c["outcome_blocks"])),
        "grouping_variables": dict(
            Counter(c.get("group_variable", "context_only") for c in primary)
        ),
        "population_contrasts": dict(
            Counter(
                json.dumps(
                    c.get("groups", c.get("context_measure", {}).get("groups", {})), sort_keys=True
                )
                for c in primary
            )
        ),
        "estimands": dict(
            Counter(x["estimand"] for c in primary for x in c["components"] if "estimand" in x)
        ),
        "api_operations": dict(
            Counter(
                n.func.id
                for c in primary
                for n in ast.walk(ast.parse(c["program"]))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            )
        ),
        "template_families": dict(Counter(str(c["template_family"]) for c in cases)),
        "language_styles": {
            "counts": dict(Counter(c["language_style"] for c in cases)),
            "method": "E-independent deterministic client register variants with canonical propositions",
            "generative_paraphrasing_complete": False,
        },
        "lexical_audit": lexical_audit(primary),
        "language_review_complete": False,
    }


def write_manifest(output: Path) -> None:
    write_json(
        output / "manifest.json",
        {
            "version": VERSION,
            "prompt_version": PROMPT_VERSION,
            "catalogue_format": "variable-blocks-0.2.0",
            "files": {
                p.name: file_hash(p)
                for p in output.iterdir()
                if p.is_file() and p.name != "manifest.json"
            },
            "release_status": "draft_pending_expert_review",
            "expert_review_complete": False,
            "independent_judge_validation": "deferred",
        },
    )
