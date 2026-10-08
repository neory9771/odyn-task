"""Audit prepared data and freeze the run identity before allocating model memory."""

from __future__ import annotations

import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...common.storage import digest, read_json
from ...tasks import get_task
from ..config import Config, update_windows
from ..data import TokenDataset, validate_prepared
from ..provenance import implementation_hashes
from .continuation import continuation_checkpoint


@dataclass(frozen=True)
class RunPlan:
    config: Config
    output: Path
    tokenizer: Any
    train_data: TokenDataset
    val_data: TokenDataset
    windows: list[tuple[int, int]]
    planned_steps: int
    warmup_steps: int
    data_manifest: dict[str, Any]
    versions: dict[str, str]
    implementation: dict[str, str]
    scoring_implementation: dict[str, str]
    fingerprint: str
    continuation: tuple[Path, dict[str, Any]] | None
    completed_summary: dict[str, Any] | None


def prepare_run(config: Config) -> RunPlan:
    from transformers import AutoTokenizer

    output, prepared = Path(config.output), Path(config.prepared)
    data_manifest = validate_prepared(config)
    tokenizer = AutoTokenizer.from_pretrained(
        prepared / "tokenizer", local_files_only=True, fix_mistral_regex=config.fix_mistral_regex
    )
    if digest(tokenizer.chat_template) != data_manifest["chat_template_hash"]:
        raise ValueError("Chat template changed")
    train_data = TokenDataset(prepared, "train", config.train_example_limit)
    val_data = TokenDataset(prepared, "validation", config.validation_example_limit)
    windows = update_windows(len(train_data), config.gradient_accumulation * config.microbatch_size)
    planned_steps = len(windows) * config.epochs
    if config.max_steps is not None:
        planned_steps = min(planned_steps, config.max_steps)
    warmup_steps = math.ceil(planned_steps * config.warmup_ratio)
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "peft",
            "bitsandbytes",
            "tensorboard",
            "numpy",
            "accelerate",
        )
    }
    implementation = implementation_hashes()
    scoring = get_task(config.task).implementation_hashes()
    fingerprint = digest(
        {
            "config": config.as_dict(),
            "data": data_manifest,
            "versions": versions,
            "implementation": implementation,
            "scoring_implementation": scoring,
        }
    )
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
        raise ValueError("Existing output settings/data/software changed; use a new run")
    summary_path = output / "training_summary.json"
    completed = read_json(summary_path) if summary_path.exists() else None
    continuation = (
        None
        if completed is not None
        else continuation_checkpoint(config, data_manifest, versions, implementation, scoring)
    )
    return RunPlan(
        config=config,
        output=output,
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        windows=windows,
        planned_steps=planned_steps,
        warmup_steps=warmup_steps,
        data_manifest=data_manifest,
        versions=versions,
        implementation=implementation,
        scoring_implementation=scoring,
        fingerprint=fingerprint,
        continuation=continuation,
        completed_summary=completed,
    )
