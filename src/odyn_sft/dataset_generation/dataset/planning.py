"""Plan the splits: partition families, select coverage-balanced cases, check isolation."""

from __future__ import annotations

import multiprocessing
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from ...common.storage import digest, file_hash, read_json, write_json
from ...survey_evaluation.package_integrity import SPLITS, VERSION
from ...survey_evaluation.study_data import load_inputs
from ..comparison_groups import load_comparison_groups
from ..generation import blocked_cores, candidates, validate_scenario_bindings
from ..partition import assign_owners, enumerate_family_specs
from ..provenance import implementation_hashes
from ..selection import coverage, select
from ..types import Candidate, DataObject, PathLike
from .families import execute_family
from .settings import load_config

_INVENTORY: tuple[Any, DataObject, int] | None = None


def _execute_one(spec: DataObject) -> DataObject:
    data, metadata, seed = _INVENTORY
    return execute_family(spec, data, metadata, seed)


def execute_inventory(specs: list[DataObject], data: Any, metadata: DataObject, seed: int) -> Iterator[DataObject]:
    """Execute every family's reference analysis, in input order, across all CPU cores.

    Each family is independent and deterministic, so the parallel result equals the
    sequential one. Forked workers inherit the study data instead of pickling it per task.
    """
    global _INVENTORY
    _INVENTORY = (data, metadata, seed)
    workers = min(len(specs), os.cpu_count() or 1)
    if workers <= 1:
        yield from map(_execute_one, specs)
        return
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
        yield from pool.map(_execute_one, specs, chunksize=16)


def build_selection(
    config: DataObject, progress=None
) -> tuple[DataObject, DataObject, dict[str, list[Candidate]]]:
    """Partition metadata-only families before any reference execution or cases."""
    config = {**config, "study_bindings": load_comparison_groups(config.get("study_bindings"))}
    data, metadata = load_inputs(config["source"], config["metadata"])
    validate_scenario_bindings(config["scenario_rules"], metadata)
    blocked, corpus = blocked_cores(config["overlap_corpus"], metadata)
    specs = enumerate_family_specs(metadata, config["seed"], config.get("study_bindings"))
    owners = assign_owners(
        specs, config["partition_ratios"], config["seed"], config["heldout_outcome_blocks"], blocked
    )
    # Ownership is now fixed. Only after this point are data/code/results touched.
    if progress:
        progress({"stage": "partitioned", "families": dict(Counter(owners.values()))})
    analyses = {split: [] for split in SPLITS}
    owned = [spec for spec in specs if spec["core"] in owners]
    for i, record in enumerate(execute_inventory(owned, data, metadata, config["seed"])):
        analyses[owners[owned[i]["core"]]].append(record)
        if progress and i % 250 == 0:
            progress({"stage": "reference_inventory", "completed": i, "inventory": len(owned)})
    selections = {}
    reports = {}
    all_shortages = []
    for i, split in enumerate(SPLITS):
        pool = candidates(analyses[split], config["seed"] + i, blocked, config["scenario_rules"])
        policy = config["splits"][split]
        chosen, _, shortages = select(
            pool,
            policy["size"],
            config["seed"] + 100 + i,
            policy["coverage_minima"],
            policy["interaction_minima"],
            config["heldout_outcome_blocks"],
        )
        selections[split] = chosen
        report = coverage(
            chosen,
            policy["size"],
            policy["coverage_minima"],
            policy["interaction_minima"],
            config["heldout_outcome_blocks"],
        )
        report.update(
            candidate_cases=len(pool),
            owned_families=len(analyses[split]),
            excluded_boundary=sum(not a["stability"]["stable"] for a in analyses[split]),
            scenario_counts_diagnostic_only=dict(Counter(c["scenario"] for c in chosen)),
        )
        reports[split] = report
        all_shortages.extend({"split": split, **s} for s in shortages)
    assert_isolation(selections, owners)
    plan = {
        "version": VERSION,
        "config": config,
        "implementation_hashes": implementation_hashes(),
        "source_sha256": file_hash(config["source"]),
        "metadata_sha256": file_hash(config["metadata"]),
        "overlap_corpus": corpus,
        "family_owners": owners,
        "family_partition_hash": digest(owners),
        "partition_before_reference_execution": True,
        "coverage": reports,
        "shortages": all_shortages,
        "selected_family_ids": {s: [c["family"] for c in selections[s]] for s in SPLITS},
        "selected_atomic_keys": {
            s: [
                k
                for c in selections[s]
                for k in ([a["core"] for a in c["analyses"]] + c.get("context_keys", []))
            ]
            for s in SPLITS
        },
        "feasible": not all_shortages and all(all(r["checks"].values()) for r in reports.values()),
        "llm_calls": 0,
        "status": "draft_plan_pending_review",
    }
    return plan, metadata, selections


def assert_isolation(selections: dict[str, list[Candidate]], owners: dict[str, str]) -> None:
    """Check compound families, constituent atoms and unavailable context cores."""
    seen, seen_families = {}, {}
    for split, cases in selections.items():
        for case in cases:
            if case["family"] in seen_families:
                raise ValueError("Repeated case family across/within splits")
            seen_families[case["family"]] = split
            atoms = [a["core"] for a in case["analyses"]] + case.get("context_keys", [])
            for core in atoms:
                if owners.get(core) != split or core in seen:
                    raise ValueError("Atomic family isolation failure")
                seen[core] = split


def audit_splits(config_path: PathLike, output: PathLike, progress=None) -> DataObject:
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new audit directory")
    plan, _, _ = build_selection(load_config(config_path), progress)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    return plan


def validate_split_plan(path: PathLike) -> DataObject:
    plan = read_json(path)
    if plan.get("version") != VERSION or plan["implementation_hashes"] != implementation_hashes():
        raise ValueError("Unsupported or changed split implementation")
    config = plan["config"]
    if (
        file_hash(config["source"]) != plan["source_sha256"]
        or file_hash(config["metadata"]) != plan["metadata_sha256"]
    ):
        raise ValueError("Frozen split source changed")
    metadata = read_json(config["metadata"])
    blocked, corpus = blocked_cores(config["overlap_corpus"], metadata)
    if corpus != plan["overlap_corpus"]:
        raise ValueError("Frozen overlap corpus changed")
    owners = assign_owners(
        enumerate_family_specs(metadata, config["seed"], config.get("study_bindings")),
        config["partition_ratios"],
        config["seed"],
        config["heldout_outcome_blocks"],
        blocked,
    )
    if owners != plan["family_owners"] or digest(owners) != plan["family_partition_hash"]:
        raise ValueError("Family partition changed")
    seen = set()
    families = set()
    for split in SPLITS:
        ids = plan["selected_family_ids"][split]
        atoms = plan["selected_atomic_keys"][split]
        if (
            len(ids) != config["splits"][split]["size"]
            or len(ids) != len(set(ids))
            or set(ids) & families
        ):
            raise ValueError("Split family counts/isolation failure")
        if (
            len(atoms) != len(set(atoms))
            or set(atoms) & seen
            or any(owners.get(k) != split for k in atoms)
        ):
            raise ValueError("Split atomic isolation failure")
        seen.update(atoms)
        families.update(ids)
    return {
        "kind": "two_level_split_plan",
        "integrity_valid": True,
        "feasible": plan["feasible"],
        "llm_calls": 0,
    }
