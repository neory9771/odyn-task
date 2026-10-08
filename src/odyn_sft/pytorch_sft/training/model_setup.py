"""Base precision, LoRA preparation and optimizer construction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..config import Config, lr_multiplier
from .run_setup import RunPlan


def prepare_base_for_lora(model: Any, config: Config, quantized: bool) -> Any:
    """Freeze the loaded base and set up gradient checkpointing before adding LoRA."""
    import torch
    from peft import prepare_model_for_kbit_training

    if quantized:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": config.checkpoint_use_reentrant},
        )
        # PEFT upcasts all non-quantized weights; frozen embedding outputs would
        # then make residual/checkpoint states FP32 along the entire long prompt.
        # Keep norms and trainable adapter weights FP32, frozen embeddings/head BF16.
        model.get_input_embeddings().to(dtype=torch.bfloat16)
        model.get_output_embeddings().to(dtype=torch.bfloat16)
        return model
    # No k-bit preparation: it would upcast every frozen BF16 weight to FP32.
    # The base stays BF16; PEFT keeps the trainable adapters in FP32.
    model.requires_grad_(False)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": config.checkpoint_use_reentrant}
        )
        model.enable_input_require_grads()
    return model


@dataclass(frozen=True)
class MemoryBudget:
    device: torch.device
    total_bytes: int
    allocator_budget_bytes: int
    available_bytes: int
    reported_total_bytes: int


@dataclass(frozen=True)
class TrainingComponents:
    model: Any
    parameters: list[torch.nn.Parameter]
    named_trainable: list[tuple[str, torch.nn.Parameter]]
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    optimizer_options: dict[str, Any]
    lora: Any
    quantization: Any
    memory: MemoryBudget


def configure_memory(config: Config) -> MemoryBudget:
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_properties(device)
    available, reported_total = torch.cuda.mem_get_info(device)
    budget = min(
        gpu.total_memory - config.vram_headroom_gib * 1024**3,
        available - config.available_memory_headroom_gib * 1024**3,
    )
    if budget <= 0:
        raise ValueError("GPU memory headroom exceeds capacity")
    torch.cuda.set_per_process_memory_fraction(budget / gpu.total_memory, device)
    torch.cuda.reset_peak_memory_stats()
    return MemoryBudget(device, gpu.total_memory, int(budget), available, reported_total)


def create_optimizer(
    config: Config, parameters: list[torch.nn.Parameter]
) -> tuple[Any, dict[str, Any]]:
    if config.optimizer == "adamw_8bit":
        from bitsandbytes.optim import AdamW8bit

        optimizer_class, options = AdamW8bit, {"min_8bit_size": 4096}
    else:
        optimizer_class, options = torch.optim.AdamW, {"foreach": False}
    return optimizer_class(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        **options,
    ), options


def load_training_components(plan: RunPlan) -> TrainingComponents:
    """Load exactly the configured base precision and train only LoRA parameters."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    config = plan.config
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    memory = configure_memory(config)
    quantization = config.bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        revision=config.revision,
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=config.attention,
    )
    if model.config.model_type not in ("qwen2", "qwen3", "mistral"):
        raise ValueError("Require a Qwen or Mistral model")
    if bool(getattr(model, "is_loaded_in_4bit", False)) != (quantization is not None):
        raise ValueError("Loaded model quantization does not match the configured profile")
    if quantization is None and model.get_input_embeddings().weight.dtype != torch.bfloat16:
        raise ValueError("BF16-base LoRA requires BF16 base weights")
    if config.max_length > model.config.max_position_embeddings:
        raise ValueError("Requested context exceeds the model's supported positions")
    for dataset in (plan.train_data, plan.val_data) if config.validation_generation else ():
        if int(max(dataset.prompts)) + config.generation_max_new_tokens > config.max_length:
            raise ValueError("Prompt plus validation generation budget exceeds context")
    model.config.use_cache = False
    model = prepare_base_for_lora(model, config, quantized=quantization is not None)
    torch.cuda.empty_cache()
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named or any("lora_" not in name for name, _ in named):
        raise ValueError("Only LoRA adapter weights may be trainable")
    parameters = [p for _, p in named]
    optimizer, options = create_optimizer(config, parameters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step, plan.planned_steps, plan.warmup_steps, config.minimum_lr_ratio
        ),
    )
    return TrainingComponents(
        model=model,
        parameters=parameters,
        named_trainable=named,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_options=options,
        lora=lora,
        quantization=quantization,
        memory=memory,
    )
