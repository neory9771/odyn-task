"""Assemble accepted written targets into an auditable SFT2 suite.

No model/provider calls. Preserve source inputs, responses, split IDs and metadata.
Held-out draft references remain explicitly separate from gated written targets.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from ..common.storage import file_hash, read_json, read_jsonl, write_json, write_jsonl
from ..tasks import get_task


def assemble(
    targets: Path,
    source: Path,
    output: Path,
    *,
    test_limit: int | None = None,
    token_config: Path | None = None,
    max_example_tokens: int | None = None,
) -> dict[str, Any]:
    """Validate gates, package all accepted train/val targets and a held-out subset."""
    targets, source, output = targets.resolve(), source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory; existing suites are immutable")
    if test_limit is not None and test_limit < 1:
        raise ValueError("test_limit must be positive")
    gate_path = targets / "targets.jsonl"
    gates: dict[str, dict[str, Any]] = {}
    for gate in read_jsonl(gate_path):
        key = gate["record_id"]
        if key in gates:
            raise ValueError("Duplicate target decision: " + key)
        gates[key] = gate
    files = [gate_path, targets / "summary.json", source / "manifest.json"]
    records: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation"):
        path = targets / f"{split}_synthesis.jsonl"
        files.append(path)
        records[split] = read_jsonl(path)
        for row in records[split]:
            item = row["metadata"]
            gate = gates.get(item["record_id"], {})
            if (
                gate.get("accepted") is not True
                or gate.get("split") != split
                or gate.get("answer") != row["output"]
                or item.get("split") != split
                or item.get("target_source") != "writer_gated"
            ):
                raise ValueError("Missing/mismatched accepted target: " + item["record_id"])
    token_filter = None
    if max_example_tokens is not None:
        if max_example_tokens < 1 or token_config is None:
            raise ValueError("Token filtering requires a positive cap and token_config")
        from transformers import AutoTokenizer

        from ..pytorch_sft.config import load_config
        from ..pytorch_sft.data import encode_row

        config = load_config(token_config)
        if config.task != "sft2":
            raise ValueError("Token filtering requires an SFT2 config")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            revision=config.revision,
            local_files_only=True,
            fix_mistral_regex=config.fix_mistral_regex,
        )
        token_filter = {
            "model": config.model,
            "revision": config.revision,
            "max_train_validation_example_tokens": max_example_tokens,
            "method": "exact completed-conversation tokens including EOS; no truncation",
            "test_filtered": False,
            "splits": {},
        }
        files.append(token_config.resolve())
        for split in ("train", "validation"):
            original = records[split]
            measured = [
                (row, len(encode_row(row, tokenizer, config)["input_ids"])) for row in original
            ]
            retained = [(row, size) for row, size in measured if size <= max_example_tokens]
            if not retained:
                raise ValueError("No complete examples fit token cap for " + split)
            records[split] = [row for row, size in retained]
            token_filter["splits"][split] = {
                "original": len(original),
                "retained": len(retained),
                "max_tokens": max(size for row, size in retained),
                "excluded_record_ids": [
                    row["metadata"]["record_id"]
                    for row, size in measured
                    if size > max_example_tokens
                ],
            }
    manifest = read_json(source / "manifest.json")
    test_path = source / "test_synthesis_gold.jsonl"
    lineage_path = source / "test_lineage.jsonl"
    for path in (test_path, lineage_path):
        if manifest["files"].get(path.name) != file_hash(path):
            raise ValueError(
                "Held-out artifact missing from source hashes or changed: " + path.name
            )
    files += [test_path, lineage_path]
    test, lineage = read_jsonl(test_path), read_jsonl(lineage_path)
    test_limit = len(test) if test_limit is None else test_limit
    if len(test) != len(lineage) or test_limit > len(test):
        raise ValueError("Invalid held-out alignment or requested test size")
    records["test"] = []
    for row, item in zip(test[:test_limit], lineage[:test_limit], strict=True):
        if "metadata" in row and row["metadata"] != item:
            raise ValueError("Held-out record/lineage mismatch")
        records["test"].append({**row, "metadata": item})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
    try:
        for split, rows in records.items():
            filename = f"{split}_synthesis" + ("_gold" if split == "test" else "") + ".jsonl"
            write_jsonl(stage / filename, rows)
            write_jsonl(stage / f"{split}_lineage.jsonl", [r["metadata"] for r in rows])
        write_jsonl(
            stage / "test_synthesis_inputs.jsonl",
            [{k: r[k] for k in ("instruction", "input")} for r in records["test"]],
        )
        result = {
            "version": "synthesis-sft-1",
            "counts": {k: len(v) for k, v in records.items()},
            "release_status": "draft_pending_expert_review",
            "task": "sft2",
            "conditioning": "oracle_reference_execution",
            "token_filter": token_filter,
            "llm_calls": 0,
            "target_policy": {
                "train": "writer_gated",
                "validation": "writer_gated",
                "test": "source_draft_reference_not_writer_gated",
            },
            "selection": "all accepted written targets; first held-out records in frozen order",
            "test_used_for_fitting": False,
            "client_acceptance_certified": False,
            "source_files": {str(p): file_hash(p) for p in files},
            "target_writer_summary": read_json(targets / "summary.json"),
            "files": {p.name: file_hash(p) for p in sorted(stage.iterdir()) if p.is_file()},
        }
        write_json(stage / "manifest.json", result)
        get_task("sft2").validate_suite(stage)
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def main() -> int:
    """Compatibility entry point; parsing is owned by the unified dataset CLI."""
    import sys

    from .cli import main as dataset_main

    return dataset_main(["synthesis", "assemble", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
