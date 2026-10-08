"""Load and validate the split configuration (paths, ratios, sizes, coverage minima)."""

from __future__ import annotations

from pathlib import Path

from ...common.storage import read_json
from ...survey_evaluation.package_integrity import SPLITS
from ..comparison_groups import configured_bindings
from ..config import validate_coverage_policy
from ..generation import STYLES
from ..types import DataObject, PathLike


def load_config(path: PathLike) -> DataObject:
    """Resolve all paths relative to the explicit configuration workspace."""
    path = Path(path).resolve()
    config = read_json(path)
    required = {
        "workspace",
        "source",
        "metadata",
        "overlap_corpus",
        "heldout_outcome_blocks",
        "seed",
        "splits",
        "partition_ratios",
        "train_variants",
        "scenario_rules",
    }
    if (
        not isinstance(config, dict)
        or not required <= set(config)
        or set(config) - required - {"study_bindings"}
    ):
        raise ValueError("Invalid two-level config keys")
    if (
        type(config["seed"]) is not int
        or config["seed"] < 0
        or type(config["train_variants"]) is not int
        or not 1 <= config["train_variants"] <= len(STYLES)
    ):
        raise ValueError("Invalid seed or train_variants")
    if set(config["splits"]) != set(SPLITS) or set(config["partition_ratios"]) != set(SPLITS):
        raise ValueError("Exactly train/validation/test splits required")
    for split in SPLITS:
        entry = config["splits"][split]
        if (
            set(entry) != {"size", "coverage_minima", "interaction_minima"}
            or type(entry["size"]) is not int
            or entry["size"] < 1
        ):
            raise ValueError("Invalid split policy: " + split)
        validate_coverage_policy(
            entry["coverage_minima"], entry["interaction_minima"], entry["size"]
        )
        ratio = config["partition_ratios"][split]
        if type(ratio) not in (float, int) or not 0 < ratio < float("inf"):
            raise ValueError("Partition ratios must be finite positive numbers")
    for name in ("workspace", "source", "metadata"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(name + " must be a nonempty path")
    for name in ("overlap_corpus", "heldout_outcome_blocks"):
        if not isinstance(config[name], list) or any(
            not isinstance(x, str) or not x for x in config[name]
        ):
            raise ValueError(name + " must be a list of nonempty strings")
    rules = config["scenario_rules"]
    allowed = {
        "causal_group_variables",
        "media_group_variables",
        "media_outcome_blocks",
        "purchase_outcome_blocks",
        "purchase_values",
    }
    if not isinstance(rules, dict) or set(rules) - allowed:
        raise ValueError("Invalid scenario eligibility rules")
    for name, bindings in rules.items():
        if name == "purchase_values":
            if not isinstance(bindings, dict) or any(
                not isinstance(k, str)
                or not isinstance(v, list)
                or any(not isinstance(x, str) for x in v)
                for k, v in bindings.items()
            ):
                raise ValueError("Invalid purchase_values")
        elif not isinstance(bindings, list) or any(not isinstance(x, str) for x in bindings):
            raise ValueError("Invalid scenario bindings: " + name)
    workspace = (path.parent / config["workspace"]).resolve()
    config["study_bindings"] = configured_bindings(config, workspace)
    for name in ("source", "metadata"):
        config[name] = str((workspace / config[name]).resolve())
    config["overlap_corpus"] = [str((workspace / p).resolve()) for p in config["overlap_corpus"]]
    return config
