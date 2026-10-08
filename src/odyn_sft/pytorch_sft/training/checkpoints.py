"""Atomic complete optimizer checkpoints, including RNG and sampler position."""

from __future__ import annotations

import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, NotRequired, TypedDict

import numpy as np
import torch

from ...common.storage import file_hash, read_json, write_json


class TrainingState(TypedDict):
    global_step: int
    epoch: int
    position: int
    validated: list[str]
    best_loss: float | None
    nonzero_lr_updates: NotRequired[int]
    first_update_has_learned_delta: NotRequired[bool]


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    output: Path,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    state: TrainingState,
    fingerprint: str,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    token = uuid.uuid4().hex[:12]
    staging = output / f".checkpoint-{token}"
    destination = output / f"checkpoint-{state['global_step']:06d}-{token}"
    staging.mkdir()
    try:
        model.save_pretrained(staging, safe_serialization=True)
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "state": state,
                "rng": rng_state(),
            },
            staging / "training_state.pt",
        )
        files = {p.name: file_hash(p) for p in staging.iterdir() if p.is_file()}
        write_json(
            staging / "complete.json",
            {
                "fingerprint": fingerprint,
                "global_step": state["global_step"],
                "committed_at_ns": time.time_ns(),
                "files": files,
            },
        )
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    # Keep the latest two complete checkpoints. Best adapter is exported separately.
    committed = sorted(
        (read_json(p / "complete.json")["committed_at_ns"], p)
        for p in output.glob("checkpoint-*")
        if (p / "complete.json").exists()
    )
    for _, old in committed[:-2]:
        shutil.rmtree(old)
    return destination


def latest_checkpoint(output: Path, fingerprint: str) -> Path | None:
    complete = []
    for path in output.glob("checkpoint-*"):
        marker = path / "complete.json"
        if not marker.exists():
            continue
        record = read_json(marker)
        if record["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint model/data/settings/software mismatch")
        for name, expected in record["files"].items():
            if file_hash(path / name) != expected:
                raise ValueError(f"Corrupt checkpoint: {path / name}")
        complete.append((record["global_step"], record["committed_at_ns"], path))
    return max(complete)[2] if complete else None


def restore_checkpoint(path: Path, model: Any, optimizer: Any, scheduler: Any) -> TrainingState:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(path / "adapter_model.safetensors"))
    # This is a locally created, hash-verified full optimizer checkpoint, not an external pickle.
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return payload["state"]
