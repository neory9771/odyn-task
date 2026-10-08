"""TensorBoard/W&B axes and durable live-status records for fitting."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from ...common.storage import write_json
from ..training.checkpoints import TrainingState
from ..training.run_setup import RunPlan
from .telemetry import WandbMetrics


def peak_memory(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0  # Internal CPU correctness fixtures never query GPU state.
    return torch.cuda.max_memory_allocated() / 1024**3, torch.cuda.max_memory_reserved() / 1024**3


def log_thermal(writer: Any, path: Path, step: int) -> None:
    if not path.exists():
        return
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 4096))
        lines = handle.read().splitlines()
    if not lines:
        return
    try:
        sample = json.loads(lines[-1])
        for key in ("temperature_c", "used_mib", "utilization_pct", "power_w"):
            value = float(sample[key])
            if math.isfinite(value):
                writer.add_scalar("gpu/" + key, value, step)
    except (ValueError, KeyError):
        return  # Stop enforcement belongs to the external supervisor.


class TrainingLogs:
    def __init__(
        self, plan: RunPlan, manifest: dict[str, Any], state: TrainingState, resumed: bool
    ):
        from torch.utils.tensorboard import SummaryWriter

        self.plan = plan
        self.writer = SummaryWriter(
            str(plan.output / "tensorboard"),
            purge_step=state["global_step"] + 1 if resumed else None,
        )
        # Microbatches and epochs have their own axes and resume purge markers.
        self.microbatch_writer = SummaryWriter(
            str(plan.output / "tensorboard_microbatches"),
            purge_step=state["epoch"] * len(plan.train_data) + state["position"] + 1
            if resumed
            else None,
        )
        self.epoch_writer = SummaryWriter(
            str(plan.output / "tensorboard_epochs"),
            purge_step=round((state["epoch"] + state["position"] / len(plan.train_data)) * 1000) + 1
            if resumed
            else None,
        )
        self.writer.add_text("experiment/resolved_parameters", json.dumps(manifest, indent=2), 0)
        self.writer.flush()
        self.tracking = WandbMetrics(plan.config, manifest, plan.output)

    def flush(self) -> None:
        for writer in (self.writer, self.microbatch_writer, self.epoch_writer):
            writer.flush()

    def close(self) -> None:
        for writer in (self.epoch_writer, self.microbatch_writer, self.writer):
            writer.close()
        self.tracking.finish()

    def microbatch(
        self,
        state: TrainingState,
        start: int,
        end: int,
        processed: int,
        index: int,
        batch: dict[str, Any],
        loss: float,
        seconds: float,
    ) -> None:
        step = state["epoch"] * len(self.plan.train_data) + start + processed
        self.microbatch_writer.add_scalar("train/microbatch_loss", loss, step)
        self.microbatch_writer.flush()
        write_json(
            self.plan.output / "live_microbatch.json",
            {
                "global_step": state["global_step"],
                "microbatch": index + 1,
                "window_size": end - start,
                "loss": loss,
                "microbatch_step": step,
                "batch_size": int(batch["input_ids"].shape[0]),
                "padded_tokens": int(batch["input_ids"].numel()),
                "upcoming_optimizer_step": state["global_step"] + 1,
                "seconds": seconds,
            },
        )

    def update(self, state: TrainingState, metrics: dict[str, float | int]) -> None:
        step, epoch = state["global_step"], metrics["epoch"]
        self.tracking.log("train", metrics, step, epoch)
        for name, value in metrics.items():
            self.writer.add_scalar("train/" + name, value, step)
        self.epoch_writer.add_scalar("epoch/train_loss", metrics["loss"], round(epoch * 1000))
        log_thermal(self.writer, Path(self.plan.config.thermal_log), step)
        self.writer.flush()
        self.epoch_writer.flush()
        write_json(
            self.plan.output / "live_status.json",
            {**state, **metrics, "planned_steps": self.plan.planned_steps},
        )
        write_json(
            self.plan.output / "latest_update.json",
            {"step": step, "seconds": metrics["update_seconds"]},
        )
        print(json.dumps({"stage": "train", "global_step": step, **metrics}), flush=True)

    def validation(self, metrics: dict[str, Any], state: TrainingState, epoch: float) -> None:
        step = state["global_step"]
        for name, value in metrics.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar("val/" + name, value, step)
                self.epoch_writer.add_scalar("epoch/val_" + name, value, round(epoch * 1000))
        self.tracking.log("val", metrics, step, epoch)
        generated = metrics.get("generated_outcomes", {})
        scalars = {
            f"generated_{name}": count
            for name, count in generated.items()
            if isinstance(count, (float, int))
        }
        scalars.update(
            {
                f"generated_failure/{name}": count
                for name, count in generated.get("failure_types", {}).items()
            }
        )
        scalars.update(
            {
                f"generated_failure_detail/{name}": count
                for name, count in generated.get("failure_subtypes", {}).items()
            }
        )
        for name, value in scalars.items():
            self.writer.add_scalar("val/" + name, value, step)
            self.epoch_writer.add_scalar("epoch/val_" + name, value, round(epoch * 1000))
        self.writer.add_scalar("val/epoch", epoch, step)
        log_thermal(self.writer, Path(self.plan.config.thermal_log), step)
