"""Quarter-epoch validation and learned-adapter checkpoint selection."""

from __future__ import annotations

from ...common.storage import file_hash, write_json
from ..config import quarter_boundaries
from ..evaluation.evaluator import evaluate, selection_loss_precision
from .adapters import assess_model
from .session import TrainingSession


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
    elif state["best_loss"] is None or metrics["loss"] < state["best_loss"]:
        state["best_loss"] = metrics["loss"]
        model.save_pretrained(plan.output / "best_adapter", safe_serialization=True)
        plan.tokenizer.save_pretrained(plan.output / "best_adapter")
        selection = {
            "tag": tag,
            "global_step": state["global_step"],
            **metrics,
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
