"""Record effective settings and model/data/software provenance for a run."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from ...common.storage import read_json, write_json
from ..config import lr_multiplier, quarter_boundaries
from ..evaluation.evaluator import selection_loss_precision
from .model_setup import TrainingComponents
from .run_setup import RunPlan


def write_run_manifest(plan: RunPlan, components: TrainingComponents) -> dict[str, Any]:
    config = plan.config
    gpu_snapshot = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    thermal_settings = Path(config.thermal_log).parent / "thermal_settings.json"
    manifest = {
        "fingerprint": plan.fingerprint,
        "config": config.as_dict(),
        "dataset": plan.data_manifest,
        "versions": plan.versions,
        "implementation": plan.implementation,
        "scoring_implementation": plan.scoring_implementation,
        "effective": {
            "train_examples": len(plan.train_data),
            "validation_examples": len(plan.val_data),
            "configured_batch_size_upper_bound": config.microbatch_size
            * config.gradient_accumulation,
            "effective_batch_sizes": [end - start for start, end in plan.windows],
            "actual_accumulation_windows": (
                [math.ceil((end - start) / config.microbatch_size) for start, end in plan.windows]
                if config.microbatch_token_budget is None
                else None
            ),
            "microbatch_token_budget": config.microbatch_token_budget,
            "batching_policy": "length-aware within each update; actual counts logged"
            if config.microbatch_token_budget
            else "fixed microbatch size",
            "updates_per_epoch": len(plan.windows),
            "total_optimizer_updates": plan.planned_steps,
            "warmup_steps": plan.warmup_steps,
            "warmup_ramp_active": plan.warmup_steps > 1,
            "warmup_note": "One-step warmup has no sub-base-LR ramp"
            if plan.warmup_steps == 1
            else "Sub-base-LR ramp"
            if plan.warmup_steps > 1
            else "Warmup disabled",
            "first_update_learning_rates": [
                config.learning_rate
                * lr_multiplier(s, plan.planned_steps, plan.warmup_steps, config.minimum_lr_ratio)
                for s in range(min(3, plan.planned_steps))
            ],
            "schedule_version": config.schedule_version,
            "first_update_learning_rate": config.learning_rate
            * lr_multiplier(0, plan.planned_steps, plan.warmup_steps),
            "validation_after_examples": quarter_boundaries(len(plan.train_data)),
            "gradient_norm": "L2 norm before clipping",
            "loss_weighting": "completion-token weighted",
            "optimizer_options": components.optimizer_options,
            "mixed_precision_scaler": None,
            "selection_metric": "minimum full validation completion loss",
            "selection_eligibility": "nonzero learned LoRA delta; base-only validations kept separately",
            "generation_validation_enabled": config.validation_generation,
            "selection_loss_precision": selection_loss_precision(config),
            "results_scope": "provisional smoke/plumbing check"
            if plan.data_manifest["provisional"] or len(plan.val_data) < 100
            else "development evaluation",
        },
        "lora": components.lora.to_dict(),
        "quantization": components.quantization.to_dict()
        if components.quantization is not None
        else {"quant_method": "none", "base_dtype": "bfloat16"},
        "trainable_parameters": [
            {
                "name": name,
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "numel": p.numel(),
            }
            for name, p in components.named_trainable
        ],
        "trainable_parameter_count": sum(p.numel() for p in components.parameters),
        "gpu": {
            "snapshot": gpu_snapshot,
            "total_bytes": components.memory.total_bytes,
            "allocator_budget_bytes": components.memory.allocator_budget_bytes,
            "available_bytes_before_loading": components.memory.available_bytes,
            "reported_total_bytes": components.memory.reported_total_bytes,
            "torch_cuda_version": torch.version.cuda,
        },
        "environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "CUDA_MPS_PIPE_DIRECTORY",
                "OMP_NUM_THREADS",
                "TOKENIZERS_PARALLELISM",
            )
        },
        "thermal": read_json(thermal_settings) if thermal_settings.exists() else {},
        "test_used_for_fitting": False,
        "continuation": plan.continuation[1] if plan.continuation else None,
        "trainer": "plain PyTorch; PEFT adapter/model loading only",
    }
    # LoraConfig.to_dict() contains sets; normalize to JSON without losing settings.
    manifest = json.loads(
        json.dumps(
            manifest,
            default=lambda value: (sorted(value) if isinstance(value, set) else str(value)),
        )
    )
    write_json(plan.output / "run_manifest.json", manifest)
    write_json(plan.output / "resolved_config.json", config.as_dict())
    return manifest
