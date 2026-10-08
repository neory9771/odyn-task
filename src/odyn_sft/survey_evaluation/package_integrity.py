"""Validate frozen dataset artifacts without generating or selecting examples."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..common.storage import file_hash, read_json, read_jsonl
from ..prompts.survey import PROMPT_VERSION, analysis_input, synthesis_input
from ..survey_metadata.catalogue import numeric_program, numeric_references
from .types import DataObject, PathLike

VERSION = "two-level-dataset-1.1.0"
SPLITS = ("train", "validation", "test")


def validate_splits_package(directory: PathLike) -> DataObject:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("version") != VERSION:
        raise ValueError("Unsupported two-level package")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Frozen package prompt version differs from the current exporter")
    for name, h in manifest["files"].items():
        if Path(name).name != name or file_hash(directory / name) != h:
            raise ValueError("Artifact hash mismatch: " + name)
    plan = read_json(directory / "plan.json")
    metadata = read_json(directory / "metadata.json")
    references = numeric_references(metadata["variables"])
    if read_json(directory / "numeric_references.json") != references:
        raise ValueError("Numeric references differ from the frozen metadata order")
    if (
        file_hash(directory / "respondents.parquet") != plan["source_sha256"]
        or file_hash(directory / "metadata.json") != plan["metadata_sha256"]
    ):
        raise ValueError("Packaged source/metadata mismatch")
    gold = read_jsonl(directory / "private_gold.jsonl")
    owner = plan["family_owners"]
    seen_atoms = {}
    seen_families = {}
    for c in gold:
        split = c["split"]
        if c["family_id"] in seen_families and seen_families[c["family_id"]] != split:
            raise ValueError("Case family crosses splits")
        seen_families[c["family_id"]] = split
        for k in c["component_keys"]:
            if k in seen_atoms and seen_atoms[k] != split:
                raise ValueError("Atomic/context family crosses splits")
            if k in owner and owner[k] != split:
                raise ValueError("Family owner mismatch")
            seen_atoms[k] = split
    for split in SPLITS:
        records = [c for c in gold if c["split"] == split]
        # Paraphrase rows (claim variants) restate a template row's claim; everything else is shared.
        templates = [c for c in records if c.get("claim_variant", "template") == "template"]
        families = {c["family_id"] for c in templates}
        if families != set(plan["selected_family_ids"][split]):
            raise ValueError("Package differs from selected families")
        variants = plan["config"]["train_variants"] if split == "train" else 1
        if len(templates) != len(families) * variants or len(
            {c["record_id"] for c in records}
        ) != len(records):
            raise ValueError("Invalid variant counts")
        if any(
            {c["variant"] for c in templates if c["family_id"] == f} != set(range(variants))
            for f in families
        ):
            raise ValueError("Invalid per-family variant coverage")
        by_id = {c["record_id"]: c for c in templates}
        for c in records:
            if c.get("claim_variant", "template") == "template":
                continue
            source = by_id.get(c.get("variant_of"))
            if c["claim_variant"] != "paraphrase" or source is None or any(
                c[k] != source[k] for k in ("family_id", "program", "evidence", "reference_perspective")
            ):
                raise ValueError("Paraphrase row must restate a template row in its split")
        lineage = read_jsonl(directory / (split + "_lineage.jsonl"))
        if [r["record_id"] for r in lineage] != [c["record_id"] for c in records]:
            raise ValueError("Lineage alignment mismatch")
        suffix = "_gold" if split == "test" else ""
        for task in ("analysis", "synthesis"):
            alpaca = read_jsonl(directory / (split + "_" + task + suffix + ".jsonl"))
            if len(alpaca) != len(records):
                raise ValueError("Paired Alpaca count mismatch")
            for row, c in zip(alpaca, records):
                if set(row) != {"instruction", "input", "output"} or not all(
                    isinstance(v, str) for v in row.values()
                ):
                    raise ValueError("Alpaca records require exactly three string fields")
                target = (
                    numeric_program(c["program"], references)
                    if task == "analysis"
                    else c["reference_perspective"]
                )
                payload = (
                    analysis_input(c["claim"], metadata)
                    if task == "analysis"
                    else synthesis_input(c["claim"], c["program"], c["evidence"], metadata)
                )
                if row["output"] != target or row["input"] != payload:
                    raise ValueError("Alpaca gold/input alignment failure")
        if split == "test":
            for filename, target in [
                ("test_analysis_inputs.jsonl", "test_analysis_gold.jsonl"),
                ("test_synthesis_oracle_inputs.jsonl", "test_synthesis_gold.jsonl"),
            ]:
                public = read_jsonl(directory / filename)
                private = read_jsonl(directory / target)
                if public != [{k: v for k, v in r.items() if k != "output"} for r in private]:
                    raise ValueError("Test inputs expose/mismatch private outputs")
    return {
        "kind": "two_level_draft_package",
        "integrity_valid": True,
        "families": dict(Counter(seen_families.values())),
        "release_status": manifest["release_status"],
        "expert_review_complete": False,
    }
