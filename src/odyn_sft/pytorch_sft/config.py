"""Frozen experiment settings shared by preparation and training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# "nf4": QLoRA on a 4-bit base. "none": LoRA on the unquantized BF16 base.
QUANTIZATIONS = {"nf4", "none"}


@dataclass(frozen=True)
class Config:
    model: str
    revision: str
    source_suite: str
    prepared: str
    output: str
    thermal_log: str
    task: str = "sft1"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_name: str | None = None
    fix_mistral_regex: bool = False
    system_prompt: str | None = None
    allow_provisional: bool = False
    max_length: int = 32768
    epochs: int = 3
    learning_rate: float = 2e-4
    microbatch_size: int = 1
    microbatch_token_budget: int | None = None
    gradient_accumulation: int = 1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    quantization: str = "nf4"
    double_quantization: bool = True
    precision: str = "bf16"
    frozen_embedding_precision: str = "bf16"
    frozen_head_precision: str = "bf16"
    attention: str = "sdpa"
    sdpa_backend: str = "flash"
    gradient_checkpointing: bool = True
    checkpoint_use_reentrant: bool = False
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    schedule_version: str = "nonzero-first-v2"
    minimum_lr_ratio: float = 0.0
    validation_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    validation_generation: bool = True
    continuation_from: str | None = None
    generation_max_new_tokens: int = 1536
    generation_do_sample: bool = False
    enable_thinking: bool = False
    packing: bool = False
    cooling_mode: str = "restart"
    loss_chunk_tokens: int = 128
    save_every_updates: int = 25
    seed: int = 42
    dataloader_num_workers: int = 0
    vram_headroom_gib: float = 3.0
    available_memory_headroom_gib: float = 1.0
    # Explicit development bounds. None means the entire declared split/run.
    train_example_limit: int | None = None
    validation_example_limit: int | None = None
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.task not in {"sft1", "sft2"}:
            raise ValueError("task must be sft1 or sft2")
        for name in (
            "max_length",
            "epochs",
            "microbatch_size",
            "gradient_accumulation",
            "lora_rank",
            "lora_alpha",
            "generation_max_new_tokens",
            "loss_chunk_tokens",
            "save_every_updates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.fix_mistral_regex) is not bool:
            raise ValueError("fix_mistral_regex must be boolean")
        if self.system_prompt is not None and (
            not isinstance(self.system_prompt, str) or not self.system_prompt.strip()
        ):
            raise ValueError("system_prompt must be nonempty text or None")
        if self.epochs > 3:
            raise ValueError("This implementation supports at most 3 epochs")
        if self.dataloader_num_workers != 0:
            raise ValueError("Use zero loader workers with the local thermal supervisor")
        if (
            self.quantization not in QUANTIZATIONS
            or self.precision != "bf16"
            or self.attention != "sdpa"
            or self.sdpa_backend != "flash"
            or self.frozen_embedding_precision != "bf16"
            or self.frozen_head_precision != "bf16"
        ):
            raise ValueError(
                "This tested CUDA profile uses an NF4 or BF16 base, BF16 compute and SDPA"
            )
        if (
            self.target_modules != "all-linear"
            or self.optimizer not in {"adamw", "adamw_8bit"}
            or self.scheduler != "cosine"
        ):
            raise ValueError("Use all-linear LoRA, AdamW/AdamW8bit and cosine scheduling")
        if self.packing:
            raise ValueError("Long records are trained individually without packing")
        if self.cooling_mode not in {"stop", "restart"}:
            raise ValueError("Cooling mode must be stop or restart")
        if self.generation_do_sample:
            raise ValueError("Validation uses deterministic greedy generation")
        if type(self.validation_generation) is not bool:
            raise ValueError("validation_generation must be a boolean")
        if self.continuation_from is not None and not isinstance(self.continuation_from, str):
            raise ValueError("continuation_from must name a parent training directory")
        if tuple(self.validation_fractions) != (0.25, 0.5, 0.75, 1.0):
            raise ValueError("Evaluate at all four quarter-epoch boundaries")
        if not 0 <= self.lora_dropout < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Invalid dropout/warmup ratio")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("Invalid minimum learning-rate ratio")
        if self.schedule_version != "nonzero-first-v2":
            raise ValueError("Use the nonzero-first-v2 learning-rate schedule")
        for name in (
            "learning_rate",
            "adam_epsilon",
            "max_grad_norm",
            "vram_headroom_gib",
            "available_memory_headroom_gib",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {name}")
        if (
            not 0 <= self.weight_decay
            or not 0 <= self.adam_beta1 < 1
            or not 0 <= self.adam_beta2 < 1
        ):
            raise ValueError("Invalid AdamW settings")
        for name in (
            "train_example_limit",
            "validation_example_limit",
            "max_steps",
            "microbatch_token_budget",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"Invalid {name}")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("Pin the model to a full commit SHA")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def bnb_config(self) -> Any:
        """4-bit loading config for QLoRA, or None to load the base in BF16 (plain LoRA)."""
        if self.quantization == "none":
            return None
        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.quantization,
            bnb_4bit_use_double_quant=self.double_quantization,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    def base_weight_precision(self) -> str:
        return "NF4" if self.quantization == "nf4" else "bf16"


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    workspace = (path.parent / values.pop("workspace", "..")).resolve()
    for key in ("source_suite", "prepared", "output", "thermal_log"):
        values[key] = str((workspace / values[key]).resolve())
    if values.get("continuation_from") is not None:
        values["continuation_from"] = str((workspace / values["continuation_from"]).resolve())
    if "validation_fractions" in values:
        values["validation_fractions"] = tuple(values["validation_fractions"])
    return Config(**values)


def quarter_boundaries(examples: int) -> list[int]:
    """Finish accumulation before each quarter; never evaluate stale gradients."""
    return sorted({math.ceil(examples * f) for f in (0.25, 0.5, 0.75, 1.0)})


def update_windows(examples: int, accumulation: int) -> list[tuple[int, int]]:
    """Shorten a window if needed to hit a quarter boundary exactly."""
    windows = []
    start = 0
    for boundary in quarter_boundaries(examples):
        while start < boundary:
            end = min(start + accumulation, boundary)
            windows.append((start, end))
            start = end
    return windows


def training_acceptance(state: dict[str, Any], examples: int, epochs: int) -> dict[str, Any]:
    """Report whether actual training met the first-update and validation checks."""
    expected = {
        f"epoch-{epoch + 1}-examples-{boundary:04d}"
        for epoch in range(epochs)
        for boundary in quarter_boundaries(examples)
    }
    completed = set(state["validated"]) & expected
    first_passed = state.get("first_update_has_learned_delta") is True
    return {
        "first_update_learned_delta_verified": first_passed,
        "expected_validation_points": len(expected),
        "completed_validation_points": len(completed),
        "missing_validation_tags": sorted(expected - completed),
        "passed": first_passed and completed == expected,
    }


def lr_multiplier(step: int, total: int, warmup: int, minimum: float = 0.0) -> float:
    """LR for the next update, where step is the count of completed updates.

    LambdaLR invokes this at construction with step=0. Warmup must therefore
    start at 1/warmup, rather than silently wasting the first gradient window.
    """
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))
