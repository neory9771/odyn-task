"""One completion-token-weighted optimizer update, with safe partial-window rollback."""

from __future__ import annotations

import time
from typing import Any

import torch

from ...common.storage import write_json
from ..data import training_batches
from ..monitoring.training_logs import peak_memory
from .adapters import verify_first_update
from .checkpoints import restore_rng, rng_state
from .objective import completion_loss
from .session import TrainingSession


def abandon_window(session: TrainingSession, window_rng: dict[str, Any]) -> None:
    """Discard partial gradients and replay the same dropout/RNG stream on resume."""
    session.components.optimizer.zero_grad(set_to_none=True)
    restore_rng(window_rng)
    session.stop_at_boundary()


def run_update(
    session: TrainingSession, start: int, end: int, order: list[int]
) -> dict[str, float | int]:
    """Commit one whole accumulation window; mutate sampler state only after step()."""
    plan, components, state = session.plan, session.components, session.state
    config, model, optimizer = plan.config, components.model, components.optimizer
    model.train()
    optimizer.zero_grad(set_to_none=True)
    window_rng = rng_state()
    records = [plan.train_data[order[j]] for j in range(start, end)]
    batches = training_batches(
        records, plan.tokenizer.pad_token_id, config.microbatch_size, config.microbatch_token_budget
    )
    denominator = sum(batch["target_tokens"] for batch in batches)
    tick = time.monotonic()
    nll_sum, input_tokens, processed_examples, padded_tokens = 0.0, 0, 0, 0
    for index, batch in enumerate(batches):
        if session.stop_requested():
            abandon_window(session, window_rng)
        write_json(
            plan.output / "live_status.json",
            {
                **state,
                "phase": "accumulating",
                "window_start": start,
                "window_end": end,
                "completed_microbatches": index,
                "planned_steps": plan.planned_steps,
            },
        )
        batch_started = time.monotonic()
        ids, labels = batch["input_ids"].to(session.device), batch["labels"].to(session.device)
        with torch.autocast(
            device_type=session.device.type,
            dtype=torch.bfloat16,
            enabled=session.device.type == "cuda",
        ):
            nll, count, _ = completion_loss(
                model,
                ids,
                labels,
                config.loss_chunk_tokens,
                attention_mask=batch["attention_mask"].to(session.device),
            )
        if count != batch["target_tokens"] or not torch.isfinite(nll):
            raise FloatingPointError("Invalid completion loss or loss mask")
        (nll / denominator).backward()
        nll_sum += float(nll.detach())
        input_tokens += int(batch["attention_mask"].sum())
        processed_examples += int(batch["input_ids"].shape[0])
        padded_tokens += int(batch["input_ids"].numel())
        session.logs.microbatch(
            state,
            start,
            end,
            processed_examples,
            index,
            batch,
            float(nll.detach()) / count,
            time.monotonic() - batch_started,
        )
    if session.stop_requested():
        abandon_window(session, window_rng)
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(
            components.parameters, config.max_grad_norm, error_if_nonfinite=True
        )
    )
    learning_rate_used = optimizer.param_groups[0]["lr"]
    optimizer.step()
    components.scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    if session.device.type == "cuda":
        torch.cuda.synchronize()
    state["global_step"] += 1
    if state["global_step"] == 1:
        verification = verify_first_update(model, learning_rate_used)
        state["first_update_has_learned_delta"] = True
        write_json(
            plan.output / "first_update_verification.json",
            {"run_fingerprint": plan.fingerprint, **verification},
        )
        session.logs.writer.add_scalar("train/first_update_learned_delta_verified", 1, 1)
    state["nonzero_lr_updates"] = state.get("nonzero_lr_updates", 0) + int(learning_rate_used > 0)
    state["position"] = end
    seconds = time.monotonic() - tick
    allocated, reserved = peak_memory(session.device)
    return {
        "loss": nll_sum / denominator,
        "grad_norm": grad_norm,
        "learning_rate": learning_rate_used,
        "next_learning_rate": components.scheduler.get_last_lr()[0],
        "epoch": state["epoch"] + end / len(plan.train_data),
        "update_seconds": seconds,
        "input_tokens_per_second": input_tokens / seconds,
        "completion_tokens": denominator,
        "effective_batch_size": end - start,
        "microbatches": len(batches),
        "max_microbatch_size": max(int(batch["input_ids"].shape[0]) for batch in batches),
        "mean_microbatch_size": (end - start) / len(batches),
        "padding_fraction": 1 - input_tokens / padded_tokens,
        "gradient_clipped": int(grad_norm > config.max_grad_norm),
        "clip_scale": min(1.0, config.max_grad_norm / (grad_norm + 1e-6)),
        "peak_allocated_gib": allocated,
        "peak_reserved_gib": reserved,
    }
