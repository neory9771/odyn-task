"""Explicit mutable training state and resources; checkpointing has one owner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch

from .checkpoints import TrainingState, save_checkpoint
from .model_setup import TrainingComponents
from .run_setup import RunPlan

if TYPE_CHECKING:
    from ..monitoring.training_logs import TrainingLogs


@dataclass
class TrainingSession:
    plan: RunPlan
    components: TrainingComponents
    state: TrainingState
    manifest: dict[str, Any]
    logs: TrainingLogs
    stop_requested: Callable[[], bool]

    def save(self) -> Path:
        return save_checkpoint(
            self.plan.output,
            self.components.model,
            self.components.optimizer,
            self.components.scheduler,
            self.state.copy(),
            self.plan.fingerprint,
        )

    def stop_at_boundary(self) -> None:
        """Save committed updates; callers restore RNG before abandoning gradients."""
        self.save()
        self.logs.flush()
        raise SystemExit(75)

    @property
    def device(self) -> torch.device:
        return self.components.memory.device
