"""Quarter-epoch validation and learned-adapter checkpoint selection."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...common.storage import file_hash, write_json
from ..config import Config, quarter_boundaries
from ..evaluation.evaluator import evaluate, selection_loss_precision
from .adapters import assess_model
from .session import TrainingSession


def selection_key(metrics: dict[str, Any], config: Config) -> list[float]:
    """Smaller is better: negated generated rate first when selected, then loss."""
    if config.selection_metric == "loss":
        return [metrics["loss"]]
    from ...tasks import get_task

    rate = metrics.get(get_task(config.task).selection_rate)
    if rate is None:
        raise ValueError("Generated selection metric missing from validation metrics")
    return [-rate, metrics["loss"]]


def is_better(key: list[float], state: dict[str, Any], config: Config) -> bool:
    best = state.get("best_key")
    if best is None and state["best_loss"] is not None and config.selection_metric == "loss":
        best = [state["best_loss"]]  # Checkpoints written before best_key was recorded.
    return best is None or key < best


def generation_check(session: TrainingSession, tag: str) -> dict[str, Any]:
    """Generate and score a fixed validation subset; full-set loss is measured separately."""
    plan = session.plan
    metrics = evaluate(
        session.components.model,
        plan.tokenizer,
        plan.val_data.subset(plan.config.generation_check_examples),
        replace(plan.config, validation_generation=True, generation_check_examples=None),
        session.device,
        plan.output,
        tag + "-generated",
        session.stop_requested,
        plan.fingerprint,
    )
    if metrics.get("interrupted"):
        session.logs.writer.flush()
        raise SystemExit(75)
    return metrics


def validate_boundary(session: TrainingSession) -> None:
    """Checkpoint before validation; mark a boundary only once it has finished."""
    plan, state, model = session.plan, session.state, session.components.model
    tag = f"epoch-{state['epoch'] + 1}-examples-{state['position']:04d}"
    if (
        state["position"] not in quarter_boundaries(len(plan.train_data))
        or tag in state["validated"]
    ):
        return
    session.save()  # Resume repeats any unfinished validation, without losing the optimizer update.
    metrics = evaluate(
        model,
        plan.tokenizer,
        plan.val_data,
        plan.config,
        session.device,
        plan.output,
        tag,
        session.stop_requested,
        plan.fingerprint,
    )
    if metrics.get("interrupted"):
        session.logs.writer.flush()
        raise SystemExit(75)
    if plan.config.generation_check_examples:
        check = generation_check(session, tag)
        # Subset loss and token accuracy would shadow the full-set values.
        metrics = {
            **metrics,
            **{k: v for k, v in check.items() if k not in metrics},
            "generation_check": {
                "examples": check["examples"],
                "loss": check["loss"],
                "source_indices": plan.val_data.subset(plan.config.generation_check_examples).indices,
            },
        }
    epoch_progress = state["epoch"] + state["position"] / len(plan.train_data)
    session.logs.validation(metrics, state, epoch_progress)
    state["validated"].append(tag)
    assessment = assess_model(model)
    if not assessment["has_learned_delta"]:
        write_json(
            plan.output / "baseline_validation.json",
            {
                "tag": tag,
                "global_step": state["global_step"],
                **metrics,
                "adapter_assessment": assessment,
                "eligible_for_best_adapter": False,
            },
        )
    elif is_better(selection_key(metrics, plan.config), state, plan.config):
        state["best_loss"] = metrics["loss"]
        state["best_key"] = selection_key(metrics, plan.config)
        model.save_pretrained(plan.output / "best_adapter", safe_serialization=True)
        plan.tokenizer.save_pretrained(plan.output / "best_adapter")
        selection = {
            "tag": tag,
            "global_step": state["global_step"],
            **metrics,
            "selection_metric": plan.config.selection_description(),
            "selection_key": state["best_key"],
            "loss_precision": selection_loss_precision(plan.config),
            "validation_source_indices": plan.val_data.indices,
            "adapter_assessment": assessment,
            "run_fingerprint": plan.fingerprint,
            "training_complete": False,
            "files": {
                p.name: file_hash(p)
                for p in (plan.output / "best_adapter").iterdir()
                if p.is_file() and p.name != "selection.json"
            },
        }
        write_json(plan.output / "best_adapter" / "selection.json", selection)
        write_json(
            plan.output / "best_validation.json",
            selection,
        )
    session.save()
    session.logs.writer.flush()
    session.logs.epoch_writer.flush()
