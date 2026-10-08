"""Run task-specific output validation on base or adapter using validation data only."""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...common.storage import digest, file_hash, read_json, write_json
from ..config import Config
from ..data import TokenDataset, validate_prepared
from ..provenance import implementation_hashes
from .final import selected_adapter


def evaluate_selected_validation(config: Config) -> dict[str, Any]:
    """Generate and score task outputs on validation; never select with test data."""
    return _evaluate_validation(config, use_adapter=True)


def evaluate_base_validation(config: Config) -> dict[str, Any]:
    """Run the same task validation with the unmodified base model."""
    return _evaluate_validation(config, use_adapter=False)


def _evaluate_validation(config: Config, *, use_adapter: bool) -> dict[str, Any]:
    if os.environ.get("ODYN_GPU_SUPERVISED") != "1":
        raise ValueError("Launch validation through odyn-sft validate or validate-base")
    if use_adapter:
        adapter, selection = selected_adapter(config)
    else:
        adapter, selection = None, None
    validate_prepared(config)
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Expose one BF16-capable CUDA GPU")

    from ...tasks import get_task

    task = get_task(config.task)
    if use_adapter:
        from peft import PeftModel
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .evaluator import evaluate

    suite = Path(config.source_suite)
    provenance = {
        "model_kind": "selected_sft_adapter" if use_adapter else "base_model",
        "adapter": str(adapter.resolve()) if adapter else None,
        "selection": selection,
        "prepared_manifest_sha256": file_hash(Path(config.prepared) / "manifest.json"),
        "validation_source_files": task.source_files(suite, "validation"),
        "task": config.task,
        "implementation": implementation_hashes(),
        "scoring_implementation": task.implementation_hashes(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes")
        },
        "generation_max_new_tokens": config.generation_max_new_tokens,
        "generation_do_sample": False,
        "split": "validation",
        "test_data_accessed": False,
    }
    fingerprint = digest(provenance)
    layout = task.report_layout(use_adapter)
    output = Path(config.output) / layout.directory
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {"fingerprint": fingerprint, **provenance}
    if manifest_path.exists() and read_json(manifest_path).get("fingerprint") != fingerprint:
        raise ValueError("Generated-validation cache model/data/software mismatch")
    write_json(manifest_path, manifest)

    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    free, _ = torch.cuda.mem_get_info(device)
    budget = min(
        gpu.total_memory - config.vram_headroom_gib * 1024**3,
        free - config.available_memory_headroom_gib * 1024**3,
    )
    if budget <= 0:
        raise ValueError("Insufficient GPU headroom")
    torch.cuda.set_per_process_memory_fraction(budget / gpu.total_memory, device)
    tokenizer = AutoTokenizer.from_pretrained(
        Path(config.prepared) / "tokenizer",
        local_files_only=True,
        fix_mistral_regex=config.fix_mistral_regex,
    )
    quantization = config.bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.revision,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    if use_adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    for module in model.modules():
        if module.__class__.__name__ in {"Qwen3RMSNorm", "Qwen2RMSNorm", "MistralRMSNorm"}:
            module.to(dtype=torch.bfloat16)
    for name, parameter in model.named_parameters():
        if use_adapter and "lora_" in name:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    model.config.use_cache = True

    settings = replace(config, validation_generation=True)
    dataset = TokenDataset(Path(config.prepared), "validation")
    writer = SummaryWriter(str(Path(config.output) / "tensorboard"))
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            metrics = evaluate(
                model,
                tokenizer,
                dataset,
                settings,
                device,
                output,
                layout.tag,
                lambda: False,
                fingerprint,
                split="validation",
            )
        step = selection["global_step"] if selection else 0
        for name, value in metrics.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(
                    f"generated_{'sft' if use_adapter else 'base'}_validation/{name}", value, step
                )
        outcome_metrics = metrics.get("generated_outcomes", {})
        if outcome_metrics:
            prefix = f"generated_{'sft' if use_adapter else 'base'}_validation"
            for count_name in ("successful", "generated"):
                if count_name in outcome_metrics:
                    writer.add_scalar(prefix + "/" + count_name, outcome_metrics[count_name], step)
            writer.add_scalar(prefix + "/failed", outcome_metrics["failed"], step)
            for failure_type, count in outcome_metrics["failure_types"].items():
                writer.add_scalar(f"{prefix}/failure/{failure_type}", count, step)
            for failure_type, count in outcome_metrics.get("failure_subtypes", {}).items():
                writer.add_scalar(f"{prefix}/failure_detail/{failure_type}", count, step)
        writer.flush()
        report = {
            "status": "complete",
            "split": "validation",
            "model_kind": "selected_sft_adapter" if use_adapter else "base_model",
            "selected_global_step": selection["global_step"] if selection else None,
            "validation": metrics,
            "test_data_accessed": False,
            "scope": task.diagnostic_scope,
            "task": config.task,
        }
        report_path = output / layout.filename
        write_json(report_path, report)
        if not use_adapter:
            report["comparison"] = compare_with_selected_adapter(config, metrics)
            write_json(report_path, report)
        return report
    finally:
        writer.close()


def compare_with_selected_adapter(config: Config, base_metrics: dict[str, Any]) -> dict[str, Any]:
    """Compare same validation record IDs and summarize metric differences."""
    root = Path(config.output)
    from ...tasks import get_task

    task = get_task(config.task)
    selected, base = task.report_layout(True), task.report_layout(False)
    selected_dir = root / selected.directory / "validation" / selected.tag
    base_dir = root / base.directory / "validation" / base.tag
    if not (selected_dir / "metrics.json").exists():
        return {"available": False, "reason": "Selected-adapter validation has not completed"}
    selected_metrics = read_json(selected_dir / "metrics.json")
    selected_records = sorted(selected_dir.glob("case-*.json"))
    base_records = sorted(base_dir.glob("case-*.json"))
    selected_ids = [read_json(path)["record_id"] for path in selected_records]
    base_ids = [read_json(path)["record_id"] for path in base_records]
    if not selected_ids or selected_ids != base_ids:
        raise ValueError("Base and SFT outputs do not cover the same validation records")
    return {
        "same_validation_record_ids": True,
        "examples": len(base_ids),
        "base_minus_sft": {
            name: base_metrics[name] - selected_metrics[name]
            for name in task.comparison_metrics
            if isinstance(base_metrics.get(name), (float, int))
            and isinstance(selected_metrics.get(name), (float, int))
        },
        "base": base_metrics.get("generated_outcomes"),
        "sft": selected_metrics.get("generated_outcomes"),
        "sft_report": str(root / selected.directory / selected.filename),
    }
