"""Export unchanged deterministic perspective targets from a frozen paired suite."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..common.storage import file_hash, read_json, read_jsonl, write_json, write_jsonl
from ..tasks import get_task
from ..tasks.sft1_suite import validate_training_suite


def export(source: Path, output: Path) -> dict[str, Any]:
    """Preserve all splits and targets; audit before atomically publishing a new view."""
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Use a new synthesis output directory")
    validate_training_suite(source)
    original = read_json(source / "manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
    try:
        names = ["train_synthesis.jsonl", "validation_synthesis.jsonl",
                 "test_synthesis_gold.jsonl", "test_synthesis_inputs.jsonl",
                 "train_lineage.jsonl", "validation_lineage.jsonl", "test_lineage.jsonl"]
        for name in names:
            if original["files"].get(name) != file_hash(source / name):
                raise ValueError("Source synthesis artifact missing or changed: " + name)
            shutil.copyfile(source / name, stage / name)
        # Public prompts have no gold or metadata; lineage is kept in its sidecar.
        test_inputs = read_jsonl(source / "test_synthesis_inputs.jsonl")
        test_gold = read_jsonl(source / "test_synthesis_gold.jsonl")
        for prompt, gold in zip(test_inputs, test_gold, strict=True):
            if "output" in prompt or any(prompt[key] != gold[key] for key in ("instruction", "input")):
                raise ValueError("Source held-out prompt/reference mismatch")
        write_jsonl(stage / "test_synthesis_inputs.jsonl",
                    [{key: row[key] for key in ("instruction", "input")} for row in test_inputs])
        result = {
            "version": "synthesis-sft-1", "task": "sft2",
            "counts": original["counts"],
            "release_status": original["release_status"],
            "conditioning": "oracle_reference_execution",
            "target_policy": {split: "unchanged_deterministic_evidence_summary"
                              for split in ("train", "validation", "test")},
            "selection": "all source records; original splits and metadata unchanged",
            "llm_calls": 0, "test_used_for_fitting": False,
            "client_acceptance_certified": False,
            "source_files": {str(source / "manifest.json"): file_hash(source / "manifest.json")},
            "files": {name: file_hash(stage / name) for name in names},
        }
        write_json(stage / "manifest.json", result)
        get_task("sft2").validate_suite(stage)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result
