"""Validated, portable evaluation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

from ..common.storage import read_json
from .comparison_groups import configured_bindings
from .types import (
    Availability,
    Composition,
    EvaluationConfig,
    EvidenceDirection,
    InferenceDomain,
    PathLike,
)

AXIS_VALUES = {
    "C": set(get_args(Composition)),
    "D": set(get_args(InferenceDomain)),
    "A": set(get_args(Availability)),
    "E": set(get_args(EvidenceDirection)),
    "R": {"select_all", "single", "ordered_scale"},
    "S": {"routing_ambiguity", "denominator_ambiguity"},
    "holdout": {"held_out"},
}


def validate_coverage_policy(minima: dict, interactions: dict, size: int) -> None:
    """Reject unknown axes, pseudo-formats and impossible per-case minimums."""
    if not isinstance(minima, dict) or not isinstance(interactions, dict):
        raise ValueError("Coverage minima and interaction minima must be objects")
    for d, values in minima.items():
        if d not in AXIS_VALUES or not isinstance(values, dict):
            raise ValueError("Invalid coverage axis: " + str(d))
        for v, n in values.items():
            if v not in AXIS_VALUES[d] or type(n) is not int or not 0 <= n <= size:
                raise ValueError(f"Invalid coverage minimum {d}={v}")
    for d in ("C", "A"):
        if sum(minima.get(d, {}).values()) > size:
            raise ValueError(f"Mutually exclusive {d} minima exceed case count")
    for expression, n in interactions.items():
        if not isinstance(expression, str) or type(n) is not int or not 0 <= n <= size:
            raise ValueError("Invalid interaction minimum")
        for token in expression.split("&"):
            parts = token.split("=")
            if (
                len(parts) != 2
                or parts[0] not in AXIS_VALUES
                or parts[1] not in AXIS_VALUES[parts[0]]
            ):
                raise ValueError("Invalid interaction token: " + token)


def load_config(path: PathLike) -> EvaluationConfig:
    """Resolve paths relative to the config workspace, independent of shell cwd."""
    path = Path(path).resolve()
    value = read_json(path)
    required = {
        "workspace",
        "source",
        "metadata",
        "overlap_corpus",
        "heldout_outcome_blocks",
        "size",
        "calibration_size",
        "seed",
        "coverage_minima",
        "interaction_minima",
        "scenario_rules",
    }
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - {"study_bindings"}
    ):
        raise ValueError(
            "Config requires these keys plus optional study_bindings: "
            + ", ".join(sorted(required))
        )
    for key, minimum in [("size", 1), ("calibration_size", 0), ("seed", 0)]:
        if type(value[key]) is not int or value[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    validate_coverage_policy(value["coverage_minima"], value["interaction_minima"], value["size"])
    rules = value["scenario_rules"]
    allowed_rules = {
        "causal_group_variables",
        "media_group_variables",
        "media_outcome_blocks",
        "purchase_outcome_blocks",
        "purchase_values",
    }
    if not isinstance(rules, dict) or set(rules) - allowed_rules:
        raise ValueError("Invalid scenario eligibility rules")
    for name, bindings in rules.items():
        if name == "purchase_values":
            if not isinstance(bindings, dict) or any(
                not isinstance(k, str)
                or not isinstance(v, list)
                or any(not isinstance(x, str) for x in v)
                for k, v in bindings.items()
            ):
                raise ValueError("purchase_values must map variables to response strings")
        elif not isinstance(bindings, list) or any(not isinstance(x, str) for x in bindings):
            raise ValueError(name + " must be a list of binding names")
    for key in ("overlap_corpus", "heldout_outcome_blocks"):
        if not isinstance(value[key], list) or any(
            not isinstance(v, str) or not v for v in value[key]
        ):
            raise ValueError(f"{key} must be a list of nonempty strings")
    for key in ("workspace", "source", "metadata"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{key} must be a nonempty path")
    workspace = (path.parent / value["workspace"]).resolve()
    result = dict(value)
    result["study_bindings"] = configured_bindings(value, workspace)
    result["source"] = str((workspace / value["source"]).resolve())
    result["metadata"] = str((workspace / value["metadata"]).resolve())
    result["overlap_corpus"] = [str((workspace / p).resolve()) for p in value["overlap_corpus"]]
    return result
