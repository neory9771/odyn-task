"""Distinguish an initialized LoRA adapter from a learned inference delta."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch


def assess_tensors(tensors: Iterable[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    """Default PEFT LoRA initializes B to zero, so all-zero B is base-only.

    A nonzero B proves this is no longer the initialized adapter. It does not
    prove useful learning or improvement over the base model.
    """
    count = nonzero = 0
    for name, value in tensors:
        if "lora_B" not in name:
            continue
        count += 1
        if not torch.isfinite(value).all().item():
            raise FloatingPointError("Non-finite LoRA B weights")
        nonzero += int(torch.count_nonzero(value).item() > 0)
    if not count:
        raise ValueError("Expected standard LoRA B tensors")
    return {
        "lora_B_tensors": count,
        "nonzero_lora_B_tensors": nonzero,
        "has_learned_delta": nonzero > 0,
        "interpretation": "changed adapter" if nonzero else "base-only initialized adapter",
    }


def assess_model(model: Any) -> dict[str, Any]:
    return assess_tensors(model.named_parameters())


def verify_first_update(model: Any, learning_rate: float) -> dict[str, Any]:
    """Check the actual optimizer result, including on the supervised GPU run."""
    assessment = assess_model(model)
    if learning_rate <= 0 or not assessment["has_learned_delta"]:
        raise RuntimeError(
            "First optimizer update did not produce a learned LoRA delta at nonzero LR"
        )
    return {
        "global_step": 1,
        "learning_rate_used": learning_rate,
        "passed": True,
        "adapter_assessment": assessment,
    }


def assess_adapter(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(path / "adapter_model.safetensors", framework="pt", device="cpu") as handle:
        return assess_tensors(
            (name, handle.get_tensor(name)) for name in handle.keys() if "lora_B" in name
        )
