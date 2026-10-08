"""Write final adapter artifacts and record whether training actually completed."""

from __future__ import annotations

import time
from typing import Any

from ...common.storage import read_json, write_json
from ..config import training_acceptance
from ..monitoring.training_logs import peak_memory
from .adapters import assess_model
from .session import TrainingSession


def finish_run(session: TrainingSession, started: float) -> dict[str, Any]:
    plan, state, model = session.plan, session.state, session.components.model
    acceptance = training_acceptance(state, len(plan.train_data), plan.config.epochs)
    if state["epoch"] == plan.config.epochs and not acceptance["passed"]:
        raise RuntimeError(
            "Training epochs ended without all first-update/validation acceptance checks"
        )
    model.save_pretrained(plan.output / "adapter", safe_serialization=True)
    plan.tokenizer.save_pretrained(plan.output / "adapter")
    summary = {
        **state,
        "planned_steps": plan.planned_steps,
        "completed_max_epochs": state["epoch"] == plan.config.epochs,
        "bounded_development_run": plan.config.max_steps is not None
        or plan.config.train_example_limit is not None,
        "session_seconds": time.monotonic() - started,
        "peak_allocated_gib": peak_memory(session.device)[0],
        "peak_reserved_gib": peak_memory(session.device)[1],
        "tensorboard": str(plan.output / "tensorboard"),
        "tensorboard_microbatches": str(plan.output / "tensorboard_microbatches"),
        "tensorboard_epochs": str(plan.output / "tensorboard_epochs"),
        "llm_api_calls": 0,
        "status": "completed_smoke_run"
        if plan.data_manifest["provisional"] or len(plan.val_data) < 100
        else "completed_training",
        "results_scope": session.manifest["effective"]["results_scope"],
        "generation_validation_enabled": plan.config.validation_generation,
        "test_evaluated": False,
        "adapter_assessment": assess_model(model),
        "run_fingerprint": plan.fingerprint,
        "acceptance_checks": acceptance,
    }
    if (plan.output / "best_adapter" / "selection.json").exists():
        selection = read_json(plan.output / "best_adapter" / "selection.json")
        selection["training_complete"] = summary["completed_max_epochs"]
        write_json(plan.output / "best_adapter" / "selection.json", selection)
        write_json(plan.output / "best_validation.json", selection)
    write_json(plan.output / "training_summary.json", summary)
    return summary
