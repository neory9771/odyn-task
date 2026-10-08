"""Validated study-specific group definitions supplied as JSON, never code."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..common.storage import read_json
from .types import DataObject, PathLike

DEFAULT_BINDINGS = Path(__file__).parent / "resources" / "survey_2022_bindings.json"


def load_comparison_groups(source: PathLike | DataObject | None = None) -> DataObject:
    """Load a JSON path or inline dict; None loads the packaged 2022 preset.

    Example: compare ages 18–34 with 55+ using the 2022 survey's stored bands::

        {"version": "survey-bindings-1", "study_id": "quant-data-example-2022",
         "groups": [{"variable": "dem_ww_age_dm_l_v1_14072020_tgt_none",
                     "groups": {"18-34": ["18-24", "25-34"],
                                "55+": ["55-64", "65 and above"]}}]}

    Here "18-34" is a group label combining two exact response values; ages
    35–54 belong to neither group. No extra keys are allowed at either level.
    study_id is nonempty text; groups is a nonempty list of definitions. Each
    definition uses an exact string variable ID and at least two group labels,
    each mapping to a nonempty list of exact stored response strings. Responses
    must be unique and disjoint within that definition; "Prefer not to say" is
    excluded. study.py separately checks IDs/values against the catalogue.
    Return a validated deep copy; malformed bindings raise ValueError, while
    file-reading/JSON errors propagate.
    """
    value: Any = (
        read_json(DEFAULT_BINDINGS if source is None else source)
        if not isinstance(source, dict)
        else source
    )
    if not isinstance(value, dict) or set(value) != {"version", "study_id", "groups"}:
        raise ValueError("Study bindings require version, study_id and groups")
    if (
        value["version"] != "survey-bindings-1"
        or not isinstance(value["study_id"], str)
        or not value["study_id"].strip()
    ):
        raise ValueError("Invalid study binding version or study_id")
    if not isinstance(value["groups"], list) or not value["groups"]:
        raise ValueError("Study bindings must contain nonempty group definitions")
    for row in value["groups"]:
        if not isinstance(row, dict) or set(row) != {"variable", "groups"}:
            raise ValueError("A binding requires variable and groups")
        if not isinstance(row["variable"], str) or not row["variable"].strip():
            raise ValueError("Group variable must be an exact nonempty string ID")
        groups = row["groups"]
        if not isinstance(groups, dict) or len(groups) < 2:
            raise ValueError("A binding requires at least two groups")
        seen: set[str] = set()
        for label, values in groups.items():
            if (
                not isinstance(label, str)
                or not label.strip()
                or not isinstance(values, list)
                or not values
            ):
                raise ValueError("Groups require nonempty labels and response lists")
            if any(not isinstance(v, str) or not v or v == "Prefer not to say" for v in values):
                raise ValueError("Groups require stored response strings excluding exact refusal")
            if len(set(values)) != len(values) or seen.intersection(values):
                raise ValueError("Group responses must be unique and disjoint")
            seen.update(values)
    return deepcopy(value)


def configured_bindings(config: DataObject, workspace: Path) -> DataObject:
    """Resolve an optional file against the workspace and freeze its JSON content."""
    source = config.get("study_bindings")
    if isinstance(source, str):
        source = workspace / source
    elif source is not None and not isinstance(source, dict):
        raise ValueError("study_bindings must be a JSON path or inline object")
    return load_comparison_groups(source)
