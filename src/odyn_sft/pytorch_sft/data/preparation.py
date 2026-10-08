"""Prepare and audit completion-masked train/validation token files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ...common.storage import digest, file_hash, read_json, write_json
from ..config import Config


def encode_row(row: dict[str, Any], tokenizer: Any, config: Config) -> dict[str, Any]:
    """Supervise the completion through its EOS, excluding template whitespace."""
    from .tokenization import tokenize_row

    encoded = tokenize_row(
        row,
        tokenizer,
        config.max_length,
        enable_thinking=config.enable_thinking,
        system_prompt=config.system_prompt,
    )
    prompt = encoded["completion_mask"].count(0)
    eos = tokenizer.eos_token_id
    tail = encoded["input_ids"][prompt:]
    if eos not in tail:
        raise ValueError("The assistant completion must end with an EOS token")
    end = prompt + tail.index(eos) + 1
    if tokenizer.decode(encoded["input_ids"][end:], skip_special_tokens=False).strip():
        raise ValueError("Unexpected non-whitespace content after assistant EOS")
    return {key: value[:end] for key, value in encoded.items()}


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare(config: Config) -> dict[str, Any]:
    from transformers import AutoTokenizer

    root = Path(config.prepared)
    if root.exists():
        raise ValueError("Use a new prepared directory")
    suite = Path(config.source_suite)
    from ...tasks import get_task

    task = get_task(config.task)
    validation = task.validate_suite(suite)
    provisional = read_json(suite / "manifest.json").get("release_status") != "human_approved"
    if provisional and not config.allow_provisional:
        raise ValueError("Draft data requires allow_provisional for this development experiment")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model,
        revision=config.revision,
        trust_remote_code=False,
        fix_mistral_regex=config.fix_mistral_regex,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    root.mkdir(parents=True)
    tokenizer.save_pretrained(root / "tokenizer")
    split_info = {}
    # Test records remain in the source suite; no test tokenization or fitting.
    for split in ("train", "validation"):
        offsets, prompts, lengths, completion_lengths = [0], [], [], []
        filename = task.record_path(suite, split)
        binary = root / f"{split}.tokens.bin"
        with binary.open("wb") as handle:
            for row in iter_jsonl(filename):
                encoded = encode_row(row, tokenizer, config)
                ids = encoded["input_ids"]
                prompt_length = encoded["completion_mask"].count(0)
                np.asarray(ids, dtype=np.int32).tofile(handle)
                offsets.append(offsets[-1] + len(ids))
                prompts.append(prompt_length)
                lengths.append(len(ids))
                completion_lengths.append(len(ids) - prompt_length)
        if not lengths:
            raise ValueError(f"Empty {split} split")
        np.save(root / f"{split}.offsets.npy", np.asarray(offsets, dtype=np.int64))
        np.save(root / f"{split}.prompts.npy", np.asarray(prompts, dtype=np.int32))
        split_info[split] = {
            "rows": len(lengths),
            "min_tokens": min(lengths),
            "max_tokens": max(lengths),
            "mean_tokens": float(np.mean(lengths)),
            "max_completion_tokens": max(completion_lengths),
            "source_sha256": file_hash(filename),
            "lineage_sha256": file_hash(suite / f"{split}_lineage.jsonl"),
        }
    manifest = {
        "version": "pytorch-qlora-data-2",
        "task": config.task,
        "model": config.model,
        "revision": config.revision,
        "max_length": config.max_length,
        "enable_thinking": config.enable_thinking,
        "fix_mistral_regex": config.fix_mistral_regex,
        "system_prompt": config.system_prompt,
        "packing": config.packing,
        "provisional": provisional,
        "splits": split_info,
        "suite_validation": validation,
        "suite_manifest_sha256": file_hash(suite / "manifest.json"),
        "chat_template_hash": digest(tokenizer.chat_template),
        "truncation": False,
        "loss_scope": "assistant completion tokens only, including assistant EOS",
        "post_eos_template_tokens": "excluded",
        "test_used_for_fitting": False,
        "files": {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()},
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def validate_prepared(config: Config) -> dict[str, Any]:
    root = Path(config.prepared)
    manifest = read_json(root / "manifest.json")
    if manifest.get("task", "sft1") != config.task:
        raise ValueError("Prepared task changed; prepare a separate task directory")
    if manifest.get("post_eos_template_tokens") != "excluded":
        raise ValueError("Prepared data uses the legacy EOS mask; prepare a new directory")
    if any(manifest[k] != getattr(config, k) for k in ("model", "revision", "max_length")):
        raise ValueError("Prepared model, revision or context budget changed")
    if any(manifest.get(k, False) != getattr(config, k) for k in ("enable_thinking", "packing")):
        raise ValueError("Prepared thinking or packing settings changed")
    if (
        manifest.get("fix_mistral_regex", False) != config.fix_mistral_regex
        or manifest.get("system_prompt") != config.system_prompt
    ):
        raise ValueError("Prepared tokenizer regex or system prompt changed")
    if manifest["provisional"] and not config.allow_provisional:
        raise ValueError("Draft data requires explicit development settings")
    for name, expected in manifest["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    suite = Path(config.source_suite)
    if file_hash(suite / "manifest.json") != manifest["suite_manifest_sha256"]:
        raise ValueError("Source suite changed")
    from ...tasks import get_task

    task = get_task(config.task)
    for split, info in manifest["splits"].items():
        if file_hash(task.record_path(suite, split)) != info["source_sha256"]:
            raise ValueError(f"Source {split} records changed")
        if file_hash(suite / f"{split}_lineage.jsonl") != info["lineage_sha256"]:
            raise ValueError(f"Source {split} lineage changed")
    return manifest


def validate_source_suite(suite: Path, task_name: str = "sft1") -> dict[str, Any]:
    """Delegate suite integrity to the selected task adapter."""
    from ...tasks import get_task

    return get_task(task_name).validate_suite(suite)
