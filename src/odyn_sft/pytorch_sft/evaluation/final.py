"""Evaluate one frozen, selected adapter after all requested epochs finish.

The test split cannot influence fitting, checkpoint selection, or decoding
settings. Runs are resumable at case/token boundaries under the same thermal
supervisor used by training. No external LLM or judge calls are made.
"""

from __future__ import annotations

import fcntl
import importlib.metadata
import os
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ...common.storage import digest, file_hash, read_json, write_json
from ..config import Config, quarter_boundaries, training_acceptance, update_windows
from ..data import HeldOutDataset, TokenDataset, validate_prepared
from ..provenance import implementation_hashes
from ..training.adapters import assess_adapter

_stop = False


def request_stop(*_: Any) -> None:
    global _stop
    _stop = True


def selected_adapter(config: Config) -> tuple[Path, dict[str, Any]]:
    """Validate finality before loading any test targets or allocating CUDA."""
    output = Path(config.output)
    summary_path = output / "training_summary.json"
    if not summary_path.exists():
        raise ValueError("Training is incomplete; no final training_summary.json")
    summary, manifest = read_json(summary_path), read_json(output / "run_manifest.json")
    train_count = manifest["effective"]["train_examples"]
    required = (
        len(update_windows(train_count, config.gradient_accumulation * config.microbatch_size))
        * config.epochs
    )
    expected_tags = {
        f"epoch-{epoch + 1}-examples-{boundary:04d}"
        for epoch in range(config.epochs)
        for boundary in quarter_boundaries(train_count)
    }
    if (
        not summary.get("completed_max_epochs")
        or summary.get("epoch") != config.epochs
        or summary.get("global_step") != required
        or summary.get("planned_steps") != required
        or not expected_tags <= set(summary.get("validated", []))
    ):
        raise ValueError(
            "Final test requires every requested epoch, update and validation boundary"
        )
    if not training_acceptance(summary, train_count, config.epochs)["passed"]:
        raise ValueError(
            "Final test requires verified first-update learning and all validation points"
        )
    if digest(manifest["config"]) != digest(config.as_dict()):
        raise ValueError("Evaluation config differs from the frozen training config")
    if summary.get("run_fingerprint") != manifest["fingerprint"]:
        raise ValueError("Training summary fingerprint mismatch")
    adapter = output / "best_adapter"
    if not (adapter / "selection.json").exists():
        raise ValueError("No eligible learned best adapter has been selected")
    selection = read_json(adapter / "selection.json")
    if (
        not selection.get("training_complete")
        or selection.get("run_fingerprint") != manifest["fingerprint"]
        or selection.get("global_step", 0) < 1
    ):
        raise ValueError("Selected adapter lacks final, matching selection provenance")
    files = selection.get("files", {})
    if "adapter_model.safetensors" not in files or "adapter_config.json" not in files:
        raise ValueError("Selected adapter lacks complete adapter file hashes")
    for name, expected in files.items():
        if file_hash(adapter / name) != expected:
            raise ValueError("Selected adapter file hash mismatch: " + name)
    assessment = assess_adapter(adapter)
    if not assessment["has_learned_delta"]:
        raise ValueError("Selected adapter is base-only, not a learned adapter")
    return adapter, {**selection, "verified_adapter_assessment": assessment}


def evaluate_final(config: Config) -> dict[str, Any]:
    """Run final validation generation, then the untouched complete test split."""
    global _stop
    _stop = False
    adapter, selection = selected_adapter(config)
    data_manifest = validate_prepared(config)
    if os.environ.get("ODYN_GPU_SUPERVISED") != "1":
        raise ValueError("Run final evaluation through odyn-sft test")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Expose one BF16-capable CUDA GPU")
    from ...tasks import get_task

    task = get_task(config.task)
    from peft import PeftModel
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .evaluator import evaluate, loss_precision_comparison

    suite = Path(config.source_suite)
    split_integrity = task.validate_suite(suite)
    directory = Path(config.output) / "final_evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {
        "adapter": str(adapter.resolve()),
        "selection": selection,
        "training_summary_sha256": file_hash(Path(config.output) / "training_summary.json"),
        "prepared_manifest_sha256": file_hash(Path(config.prepared) / "manifest.json"),
        "task": config.task,
        "source_files": {
            **task.source_files(suite, "validation"),
            **task.source_files(suite, "test"),
        },
        "implementation": implementation_hashes(),
        "scoring_implementation": task.implementation_hashes(),
        "versions": {
            n: importlib.metadata.version(n)
            for n in ("torch", "transformers", "peft", "bitsandbytes")
        },
        "generation_max_new_tokens": config.generation_max_new_tokens,
        "generation_do_sample": False,
        "inference_precision": f"BF16 norms and frozen adapters; {config.base_weight_precision()} base",
        "selection_metric": "validation completion loss among learned adapters",
        "test_used_for_selection": False,
        "llm_api_calls": 0,
    }
    fingerprint = digest(provenance)
    with (directory / ".evaluation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
            raise ValueError("Final evaluation model/data/software cache mismatch")
        write_json(manifest_path, {"fingerprint": fingerprint, **provenance})
        if (directory / "report.json").exists():
            return read_json(directory / "report.json")
        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
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
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
        for module in model.modules():
            if module.__class__.__name__ in {"Qwen3RMSNorm", "Qwen2RMSNorm", "MistralRMSNorm"}:
                module.to(dtype=torch.bfloat16)
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.data = parameter.data.to(dtype=torch.bfloat16)
        model.config.use_cache = True
        settings = replace(config, validation_generation=True)
        writer = SummaryWriter(str(Path(config.output) / "tensorboard"))
        try:
            results = {}
            final_validation_indices = []
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                # Validation task metrics diagnose the selected adapter; they
                # never change selection or generation settings before test.
                for split in ("validation", "test"):
                    dataset = (
                        TokenDataset(Path(config.prepared), "validation")
                        if split == "validation"
                        else HeldOutDataset(config, tokenizer)
                    )
                    if split == "validation":
                        final_validation_indices = dataset.indices
                    metrics = evaluate(
                        model,
                        tokenizer,
                        dataset,
                        settings,
                        device,
                        directory,
                        "selected-final",
                        lambda: _stop,
                        fingerprint,
                        split=split,
                    )
                    if metrics.get("interrupted"):
                        raise SystemExit(75)
                    results[split] = metrics
                    for name, value in metrics.items():
                        if isinstance(value, (float, int)):
                            writer.add_scalar(
                                f"final_{split}/{name}", value, selection["global_step"]
                            )
                    writer.flush()
            comparison = loss_precision_comparison(
                selection, results["validation"], final_validation_indices
            )
            writer.add_scalar(
                "final_validation/selection_loss",
                comparison["selection_loss"],
                selection["global_step"],
            )
            writer.add_scalar(
                "final_validation/loss_difference",
                comparison["final_minus_selection_loss"],
                selection["global_step"],
            )
            writer.add_text(
                "final_validation/precision_comparison",
                comparison["note"],
                selection["global_step"],
            )
            writer.flush()
            report = {
                "status": "complete",
                "selected_global_step": selection["global_step"],
                "validation_loss_comparison": comparison,
                "validation": results["validation"],
                "test": results["test"],
                "split_integrity": split_integrity,
                "provisional": data_manifest["provisional"],
                "results_scope": "smoke/plumbing check"
                if data_manifest["provisional"] or results["test"]["examples"] < 100
                else "held-out evaluation",
                "selection_uses_test": False,
                "llm_api_calls": 0,
                "task": config.task,
                "scope": task.scope,
            }
            write_json(directory / "report.json", report)
            return report
        finally:
            writer.close()
