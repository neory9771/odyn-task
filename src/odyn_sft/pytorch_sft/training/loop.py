"""Orchestrate setup, complete optimizer windows, quarter validation and finalization.

Model loading, update math, checkpoint selection, reporting and logging live in
separate modules. This module owns only lifecycle and sampler/epoch progression.
"""

from __future__ import annotations

import fcntl
import json
import os
import signal
import time
from pathlib import Path
from typing import Any

import torch

from ...common.storage import write_json
from ..config import Config
from ..monitoring.training_logs import TrainingLogs
from .checkpoints import TrainingState, latest_checkpoint, restore_checkpoint
from .model_setup import load_training_components
from .run_manifest import write_run_manifest
from .run_setup import prepare_run
from .selection import validate_boundary
from .session import TrainingSession
from .training_summary import finish_run
from .updates import run_update


class StopRequest:
    """Signal handlers set a flag; only safe optimizer boundaries save or exit."""

    def __init__(self) -> None:
        self.requested = False

    def request(self, *_: Any) -> None:
        self.requested = True

    def __call__(self) -> bool:
        return self.requested


def check_gpu_environment() -> None:
    if os.environ.get("ODYN_GPU_SUPERVISED") != "1":
        raise ValueError("Launch CUDA training through odyn-sft train --config CONFIG")
    if (
        int(os.environ.get("WORLD_SIZE", "1")) != 1
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise ValueError("Expose exactly one available CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This profile requires BF16 support")


def train(config: Config, interrupt_after_updates: int | None = None) -> dict[str, Any]:
    """Own the run lock and signal handlers for a supervised single-GPU worker."""
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    stop = StopRequest()
    previous = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        check_gpu_environment()
        with (output / ".training.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                return run_training(config, stop, interrupt_after_updates)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_training(
    config: Config, stop: StopRequest, interrupt_after_updates: int | None
) -> dict[str, Any]:
    plan = prepare_run(config)
    if plan.completed_summary is not None:
        return plan.completed_summary
    components = load_training_components(plan)
    manifest = write_run_manifest(plan, components)
    state: TrainingState = {
        "global_step": 0,
        "epoch": 0,
        "position": 0,
        "validated": [],
        "best_loss": None,
        "nonzero_lr_updates": 0,
    }
    checkpoint = latest_checkpoint(plan.output, plan.fingerprint)
    if checkpoint:
        state = restore_checkpoint(
            checkpoint, components.model, components.optimizer, components.scheduler
        )
    elif plan.continuation:
        state = restore_checkpoint(
            plan.continuation[0], components.model, components.optimizer, components.scheduler
        )
        if plan.continuation[1].get('scheduler_horizon_shortened'):
            from .continuation import reconcile_shortened_schedule

            if state['global_step'] >= plan.planned_steps:
                raise ValueError('Shortened schedule is already exhausted by the parent updates')
            reconcile_shortened_schedule(components.optimizer, components.scheduler)
        state["validated"], state["best_loss"] = [], None
        print(
            json.dumps(
                {
                    "stage": "continuation",
                    "global_step": state["global_step"],
                    "position": state["position"],
                    **plan.continuation[1],
                }
            ),
            flush=True,
        )
    logs = TrainingLogs(plan, manifest, state, resumed=checkpoint is not None)
    session = TrainingSession(
        plan=plan,
        components=components,
        state=state,
        manifest=manifest,
        logs=logs,
        stop_requested=stop,
    )
    try:
        return fit(session, interrupt_after_updates)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            write_json(
                plan.output / "failure.json",
                {"type": type(exc).__name__, "message": str(exc), **state},
            )
        raise
    finally:
        logs.close()


def finish_epoch(session: TrainingSession) -> None:
    if session.state["position"] == len(session.plan.train_data):
        session.state["epoch"] += 1
        session.state["position"] = 0
        session.save()


def fit(session: TrainingSession, interrupt_after_updates: int | None = None) -> dict[str, Any]:
    """Replay pending validation on resume, then commit complete update windows."""
    plan, state = session.plan, session.state
    started = time.monotonic()
    validate_boundary(session)
    finish_epoch(session)
    while state["epoch"] < plan.config.epochs and state["global_step"] < plan.planned_steps:
        validate_boundary(session)
        order = torch.randperm(
            len(plan.train_data),
            generator=torch.Generator().manual_seed(plan.config.seed + state["epoch"]),
        ).tolist()
        for start, end in plan.windows:
            if end <= state["position"]:
                continue
            if start != state["position"]:
                raise ValueError("Sampler position is not a complete optimizer boundary")
            if session.stop_requested():
                session.stop_at_boundary()
            metrics = run_update(session, start, end, order)
            session.logs.update(state, metrics)
            if state["global_step"] % plan.config.save_every_updates == 0:
                session.save()
            if session.stop_requested() or (
                interrupt_after_updates is not None
                and state["global_step"] >= interrupt_after_updates
            ):
                session.stop_at_boundary()
            validate_boundary(session)
            if state["global_step"] >= plan.planned_steps:
                break
        finish_epoch(session)
    session.save()
    return finish_run(session, started)
