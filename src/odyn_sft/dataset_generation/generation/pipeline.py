"""The two commands: audit (capacity plan, no samples) and generate (reproduce a reviewed plan)."""

from __future__ import annotations

import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ...common.storage import atomic_text, file_hash, read_json, write_json, write_jsonl
from ...survey_api import run_reference
from ...survey_evaluation.study_data import load_inputs
from ..comparison_groups import load_comparison_groups
from ..provenance import implementation_hashes
from ..selection import (
    DEFAULT_INTERACTIONS,
    DEFAULT_MINIMA,
    coverage,
    scaled_minima,
    select,
)
from ..study import rubric
from ..types import DataObject, PathLike
from .candidates import candidates
from .cases import materialize, reference_perspective, validate_reference
from .inventory import blocked_cores, enumerate_analyses
from .scenarios import validate_scenario_bindings
from .version import VERSION


def audit(
    source: PathLike,
    metadata_path: PathLike,
    output: PathLike,
    size: int = 1000,
    calibration_size: int = 121,
    seed: int = 20261005,
    overlaps: list[PathLike] | None = None,
    coverage_minima: dict[str, dict[str, int]] | None = None,
    interaction_minima: dict[str, int] | None = None,
    heldout_blocks: list[str] | None = None,
    scenario_rules: DataObject | None = None,
    study_bindings: DataObject | None = None,
) -> DataObject:
    """Save a dimension-constrained capacity plan without writing sample files."""
    if size < 1 or calibration_size < 0:
        raise ValueError("Eval size must be positive and calibration size nonnegative")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new audit directory")
    minima = (
        coverage_minima
        if coverage_minima is not None
        else scaled_minima(DEFAULT_MINIMA, size / 1000)
        if size >= 100
        else {}
    )
    interactions = (
        interaction_minima
        if interaction_minima is not None
        else {r: math.ceil(n * size / 1000) for r, n in DEFAULT_INTERACTIONS.items()}
        if size >= 100
        else {}
    )
    heldout_blocks = list(heldout_blocks or [])
    if not heldout_blocks and minima.get("holdout"):
        raise ValueError("Held-out coverage required but no held-out blocks declared")
    data, metadata = load_inputs(source, metadata_path)
    validate_scenario_bindings(scenario_rules or {}, metadata)
    blocked, corpus = blocked_cores(overlaps or [], metadata)
    study_bindings = load_comparison_groups(study_bindings)
    analyses = enumerate_analyses(data, metadata, seed, study_bindings)
    pool = candidates(analyses, seed, blocked, scenario_rules)
    cal_minima = scaled_minima(minima, calibration_size / size)
    cal_interactions = {r: math.ceil(n * calibration_size / size) for r, n in interactions.items()}
    cal, reserved, cal_shortages = select(
        pool, calibration_size, seed + 1, cal_minima, cal_interactions, heldout_blocks
    )
    evaluation, _, shortages = select(
        pool, size, seed + 2, minima, interactions, heldout_blocks, reserved
    )
    cal_coverage = coverage(cal, calibration_size, cal_minima, cal_interactions, heldout_blocks)
    eval_coverage = coverage(evaluation, size, minima, interactions, heldout_blocks)
    all_shortages = [{"split": "calibration", **v} for v in cal_shortages] + [
        {"split": "eval", **v} for v in shortages
    ]
    plan = {
        "version": VERSION,
        "seed": seed,
        "size": size,
        "calibration_size": calibration_size,
        "source": str(Path(source).resolve()),
        "source_sha256": file_hash(source),
        "metadata": str(Path(metadata_path).resolve()),
        "metadata_sha256": file_hash(metadata_path),
        "generator_sha256": file_hash(__file__),
        "implementation_hashes": implementation_hashes(),
        "overlap_corpus": corpus,
        "scope": "Only supplied 2022 survey; no candidate model outputs used.",
        "coverage_minima": {"eval": minima, "calibration": cal_minima},
        "interaction_minima": {"eval": interactions, "calibration": cal_interactions},
        "scenario_rules": scenario_rules or {},
        "heldout_outcome_blocks": heldout_blocks,
        "study_bindings": study_bindings,
        "selection_rule": "seeded_coverage_constraints_then_feasible_joint_stratum_balancing_no_component_reuse",
        "boundary_policy": "exclude_unstable_analyses_from_primary_and_calibration",
        "coverage": {"eval": eval_coverage, "calibration": cal_coverage},
        "coverage_checks": eval_coverage["checks"],
        "shortages": all_shortages,
        "feasible": not all_shortages
        and all(eval_coverage["checks"].values())
        and all(cal_coverage["checks"].values()),
        "selected_family_ids": {
            "calibration": [c["family"] for c in cal],
            "eval": [c["family"] for c in evaluation],
        },
        "scenario_counts_diagnostic_only": {
            "calibration": dict(Counter(c["scenario"] for c in cal)),
            "eval": dict(Counter(c["scenario"] for c in evaluation)),
        },
        "expert_review_complete": False,
        "lexical_audit_complete": False,
        "status": "capacity_plan_pending_review_not_a_released_benchmark",
    }
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    write_json(
        output / "capacity.json",
        {
            "analyses": len(analyses),
            "blocked_cores": len(blocked),
            "excluded_boundary_analyses": sum(not a["stability"]["stable"] for a in analyses),
            "candidate_scenarios_diagnostic_only": dict(Counter(c["scenario"] for c in pool)),
            "candidate_dimensions": {
                d: dict(Counter(v for c in pool for v in c["dimensions"][d]))
                for d in ("C", "D", "A", "E", "R", "S")
            },
            "scope_of_overlap_audit": corpus,
            "limitations": [
                "Candidate counts share resources and are not jointly selectable capacity.",
                "Greedy shortage is not proof of global infeasibility.",
                "No evaluation samples written by audit.",
            ],
        },
    )
    return plan


def lexical_audit(cases: list[DataObject]) -> DataObject:
    """Diagnostic multinomial NB, held-out template families, no LLM/sklearn."""
    rows = []
    for c in cases:
        measured = [x["E"] for x in c["components"] if x["D"] == ["statistical_inference"]]
        if (
            c["C"] == "single"
            and len(measured) == 1
            and measured[0] in {"support", "contradict", "inconclusive"}
        ):
            rows.append(
                (re.findall(r"\b\w+\b", c["claim"].lower()), measured[0], c["template_family"])
            )
    train = [r for r in rows if r[2] < 4]
    test = [r for r in rows if r[2] >= 4]
    labels = sorted({r[1] for r in train})
    if len(labels) != 3 or len(test) < 20:
        return {
            "status": "insufficient_coverage",
            "train": len(train),
            "test": len(test),
            "release_ready": False,
        }
    counts, docs = defaultdict(Counter), Counter()
    for words, label, _ in train:
        counts[label].update(words)
        docs[label] += 1
    vocabulary = set(w for count in counts.values() for w in count)
    totals = {label: sum(counts[label].values()) for label in labels}

    def predict(words):
        return max(
            labels,
            key=lambda label: math.log(docs[label] / len(train))
            + sum(
                math.log((counts[label][w] + 1) / (totals[label] + len(vocabulary)))
                for w in words
                if w in vocabulary
            ),
        )

    accuracy = sum(predict(words) == label for words, label, _ in test) / len(test)
    majority = docs.most_common(1)[0][0]
    baseline = sum(label == majority for _, label, _ in test) / len(test)
    return {
        "status": "diagnostic_completed_pending_language_review",
        "train": len(train),
        "test": len(test),
        "split": "template families 0-3 train; 4-5 test; no analytical component reuse",
        "accuracy": accuracy,
        "train_majority_baseline_test_accuracy": baseline,
        "flag_for_inspection": accuracy > baseline + 0.1,
        "threshold": "Heuristic 10 percentage point excess; not a formal significance test.",
        "limitations": "Measured single propositions only. Result balancing and item priors can affect text predictability. No expert language review performed.",
    }


def generate(plan_path: PathLike, output: PathLike, acknowledge_plan: bool = False) -> DataObject:
    """Reproduce a reviewed plan, validate references, then write draft artifacts."""
    if not acknowledge_plan:
        raise ValueError("Review plan.json, then pass --acknowledge-plan; no implicit generation.")
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    plan = read_json(plan_path)
    if (
        plan.get("version") != VERSION
        or plan.get("implementation_hashes") != implementation_hashes()
    ):
        raise ValueError("Unsupported or changed implementation; repeat audit")
    if not plan["feasible"]:
        raise ValueError("Plan is infeasible; inspect shortages and coverage_checks")
    for path, expected in [
        (plan["source"], plan["source_sha256"]),
        (plan["metadata"], plan["metadata_sha256"]),
        (__file__, plan["generator_sha256"]),
    ]:
        if file_hash(path) != expected:
            raise ValueError("Input/generator changed; repeat the audit: " + str(path))
    data, metadata = load_inputs(plan["source"], plan["metadata"])
    paths = [r["path"] for r in plan["overlap_corpus"]]
    blocked, corpus = blocked_cores(paths, metadata)
    if corpus != plan["overlap_corpus"]:
        raise ValueError("Overlap corpus changed; repeat audit")
    pool = candidates(
        enumerate_analyses(data, metadata, plan["seed"], plan.get("study_bindings")),
        plan["seed"],
        blocked,
        plan["scenario_rules"],
    )
    cal, reserved, shortages = select(
        pool,
        plan["calibration_size"],
        plan["seed"] + 1,
        plan["coverage_minima"]["calibration"],
        plan["interaction_minima"]["calibration"],
        plan["heldout_outcome_blocks"],
    )
    evaluation, _, eval_shortages = select(
        pool,
        plan["size"],
        plan["seed"] + 2,
        plan["coverage_minima"]["eval"],
        plan["interaction_minima"]["eval"],
        plan["heldout_outcome_blocks"],
        reserved,
    )
    if (
        shortages
        or eval_shortages
        or not all(
            coverage(
                evaluation,
                plan["size"],
                plan["coverage_minima"]["eval"],
                plan["interaction_minima"]["eval"],
                plan["heldout_outcome_blocks"],
            )["checks"].values()
        )
    ):
        raise ValueError("Plan no longer passes feasibility checks")
    for split, selected in [("calibration", cal), ("eval", evaluation)]:
        if [c["family"] for c in selected] != plan["selected_family_ids"][split]:
            raise ValueError("Selection changed; repeat audit")
    gold, oracle_checks = [], 0
    for split, selected in [("calibration", cal), ("eval", evaluation)]:
        for candidate in selected:
            case = materialize(candidate, split, plan["seed"], metadata)
            case["evidence"] = run_reference(case["program"], data, metadata)
            oracle_checks += validate_reference(case, case["evidence"])
            case["rubric"] = rubric(case, case["evidence"])
            case["rubric"]["required_components"] = case["components"]
            case["rubric"]["primary_pass_rule"] = (
                "All required material propositions and relations correctly handled."
            )
            case["reference_perspective"] = reference_perspective(case)
            gold.append(case)
    all_keys = [k for c in gold for k in c["component_keys"]]
    if len(all_keys) != len(set(all_keys)) or set(all_keys) & blocked:
        raise ValueError("Overlap detected")
    # Publish only after reference validation succeeds; existing outputs never overwritten.
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    write_json(output / "metadata.json", metadata)
    shutil.copy2(plan["source"], output / "respondents.parquet")
    for c in gold:
        c["block_holdout_status"] = (
            "not_applicable"
            if not c["outcome_blocks"]
            else "prior_training_outcome_block_held_out"
            if all(b in plan["heldout_outcome_blocks"] for b in c["outcome_blocks"])
            else "new_analysis_on_familiar_block"
        )
    for split in ("calibration", "eval"):
        cases = [c for c in gold if c["split"] == split]
        write_jsonl(output / (split + "_gold.jsonl"), cases)
        write_jsonl(
            output / (split + "_requests.jsonl"),
            [
                {
                    "case_id": c["case_id"],
                    "input": {
                        "claim": c["claim"],
                        "study": metadata["study"],
                        "catalogue_ref": "metadata.json",
                        "data_ref": "respondents.parquet",
                    },
                }
                for c in cases
            ],
        )
    evaluation_gold = [c for c in gold if c["split"] == "eval"]
    diagnostics = {
        "cases": len(evaluation_gold),
        "independent_numeric_checks": oracle_checks,
        "C": dict(Counter(c["C"] for c in evaluation_gold)),
        **{
            d: dict(
                Counter(
                    x
                    for c in evaluation_gold
                    for x in set(c[d] if isinstance(c[d], list) else [c[d]])
                )
            )
            for d in ("D", "A", "E", "R", "S", "diagnostic_flags")
        },
        "dimension_counting_unit": "cases containing the tag; multi-label totals may exceed cases",
        "coverage_constraints": plan["coverage"]["eval"],
        "block_holdout": dict(Counter(c["block_holdout_status"] for c in evaluation_gold)),
        "component_reuse": 1,
        "blocked_component_overlap": 0,
        "calibration_eval_overlap": 0,
        "lexical_audit": lexical_audit(evaluation_gold),
        "model_evaluated": False,
        "expert_review_complete": False,
        "independent_judge_validation": "deferred",
    }
    write_json(output / "coverage.json", diagnostics)
    atomic_text(
        output / "README.md",
        "# Revised local claim-evidence drafts\n\nGenerated with no LLM calls. Inputs and gold are separate.\n\nStatus: pending expert, language and judge review; not a released benchmark. No model evaluated.\n\nSee plan.json for capacity-derived allocations and overlap audit scope; coverage.json for dimensions and lexical diagnostic.\n\nAll cases use the supplied 2022 survey. No claim of production representativeness, independent studies or >80% performance.\n",
    )
    write_json(
        output / "manifest.json",
        {
            "version": VERSION,
            "plan_sha256": file_hash(plan_path),
            "files": {p.name: file_hash(p) for p in output.iterdir() if p.is_file()},
            "expert_review_complete": False,
            "release_status": "draft_pending_review",
            "llm_calls": 0,
        },
    )
    return diagnostics
