"""Audited continuation for validation changes or a shortened epoch budget."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.storage import digest, file_hash, read_json
from ..config import Config
from .checkpoints import latest_checkpoint


def continuation_checkpoint(
    config: Config,
    data: dict[str, Any],
    versions: dict[str, str],
    implementation: dict[str, str],
    scoring: dict[str, str],
) -> tuple[Path, dict[str, Any]] | None:
    """Verify immutable parent artifacts; never rewrite their fingerprints.

    Model, optimizer, sample order and tokenized examples must match. A new
    output gets its own fingerprint, with the verified parent recorded in its
    manifest. Old generated-code validation caches are deliberately excluded.
    """
    if config.continuation_from is None:
        return None
    parent = Path(config.continuation_from)
    manifest_path = parent / "run_manifest.json"
    manifest = read_json(manifest_path)
    original = {
        key: manifest[key]
        for key in ("config", "versions", "implementation", "scoring_implementation")
    }
    # The training manifest uses "dataset", whereas its fingerprint uses "data".
    original["data"] = manifest["dataset"]
    if digest(original) != manifest["fingerprint"]:
        raise ValueError("Parent run manifest fingerprint changed")
    if (
        manifest["config"].get("schedule_version", "legacy-zero-first-v1")
        != config.schedule_version
    ):
        raise ValueError("Continuation changes learning-rate schedule semantics; start a new run")
    old = Config(**manifest["config"]).as_dict()
    new = config.as_dict()
    allowed = {
        "prepared",
        "output",
        "thermal_log",
        "max_length",
        "validation_generation",
        "continuation_from",
    }
    shortened = new['epochs'] < old['epochs']
    if shortened:
        allowed.add('epochs')
    if any(digest(old[key]) != digest(value) for key, value in new.items() if key not in allowed):
        raise ValueError("Continuation changes model or optimization settings")
    if {k: v for k, v in data.items() if k != "max_length"} != {
        k: v for k, v in manifest["dataset"].items() if k != "max_length"
    }:
        raise ValueError("Continuation changes tokenized data or source provenance")
    if versions != manifest["versions"] or scoring != manifest["scoring_implementation"]:
        raise ValueError("Continuation changes software versions or scoring implementation")
    allowed_code = {
        "config.py",
        "training/loop.py",
        "evaluation/evaluator.py",
        "training/continuation.py",
    }
    names = set(implementation) | set(manifest["implementation"])
    if any(
        implementation.get(name) != manifest["implementation"].get(name)
        for name in names
        if name not in allowed_code
    ):
        raise ValueError("Continuation changes loss, checkpoint, data or thermal implementation")
    checkpoint = latest_checkpoint(parent, manifest["fingerprint"])
    if checkpoint is None:
        raise ValueError("Continuation requires a complete parent checkpoint")
    if shortened:
        import torch

        state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=False)['state']
        if state['epoch'] >= config.epochs:
            raise ValueError('Requested epoch budget is already complete in the parent checkpoint')
    from .adapters import assess_adapter

    assessment = assess_adapter(checkpoint)
    return checkpoint, {
        "parent_run": str(parent.resolve()),
        "parent_manifest_sha256": file_hash(manifest_path),
        "parent_fingerprint": manifest["fingerprint"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_marker_sha256": file_hash(checkpoint / "complete.json"),
        "changed_settings": {
            key: {"before": old[key], "after": new[key]} for key in allowed if old[key] != new[key]
        },
        "validation_caches_reused": False,
        "optimizer_scheduler_rng_preserved": not shortened,
        "optimizer_rng_and_completed_updates_preserved": True,
        "scheduler_horizon_shortened": shortened,
        "parent_adapter_assessment": assessment,
        "parent_checkpoint_is_base_only": not assessment["has_learned_delta"],
    }


def reconcile_shortened_schedule(optimizer: Any, scheduler: Any) -> None:
    """Apply the new LambdaLR horizon at the restored step without an update.

    LambdaLR.load_state_dict restores counters and the old optimizer LR, but its
    new lambda already refers to the reduced update budget. Reconcile that LR
    before the next update, keeping Adam moments and sampler/RNG state intact.
    """
    rates = [base * fn(scheduler.last_epoch)
             for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)]
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group['lr'] = rate
    scheduler._last_lr = rates
