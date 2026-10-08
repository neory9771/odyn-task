"""Execute one analysis family on the study and phrase its training claims."""

from __future__ import annotations

import pandas as pd

from ...common.storage import digest
from ...survey_api import run_reference
from ..generation import (
    annotate_candidate,
    evidence_direction,
    interval,
    materialize,
    stability,
    validate_reference,
)
from ..types import AnalysisRecord, DataObject

TRAIN_PREFIXES = (
    "",
    "For our planning meeting, I need evidence on this: ",
    "can you check this for me? i think ",
    "Research interpretation requested: ",
    "Our team is presenting the following conclusion. What evidence backs it up or challenges it? ",
    "Here is my working interpretation of the study: ",
)


def training_claim(case: DataObject, variant: int) -> str:
    """Vary client register using claim text only; evidence labels are never read."""
    if variant == 0:
        return case["claim"]
    text = " ".join(c["text"] for c in case["components"])
    if variant == 2:
        text = text.replace("Respondents in ", "People in ").replace(
            "respondents in ", "people in "
        )
    return TRAIN_PREFIXES[variant] + text


def execute_family(
    spec: DataObject, data: pd.DataFrame, metadata: DataObject, seed: int
) -> AnalysisRecord:
    """Claim spec -> program -> API evidence -> E; independent oracle checks it.

    An empty provisional reference supplies only result-blind authoring fields.
    Its E is never exported or used for sampling. Only the API result assigns E.
    """
    row = spec["row"]
    card = row["card"]
    kind = card["answer_type"]
    s = ["routing_ambiguity", "denominator_ambiguity"] if kind == "select_all" else []
    provisional = {
        **spec,
        "counts": [],
        "ci": None,
        "E": "not_applicable",
        "S": s,
        "diagnostic_flags": [],
        "stability": {
            "rule": "pending_execution",
            "stable": False,
            "original": "not_applicable",
            "perturbed_states": [],
            "limitation": "",
        },
    }
    candidate = annotate_candidate(
        {
            "scenario": "single",
            "analyses": [provisional],
            "family": digest([spec["core"]]),
            "scenario_keys": [],
        }
    )
    authored = materialize(candidate, "inventory", seed, metadata)
    evidence = run_reference(authored["program"], data, metadata)
    contrast = next(r for r in evidence["results"] if r["kind"] == "contrast")
    # Separate pandas path checks the API's population/response counts.
    block = [c["id"] for c in metadata["variables"] if c["block_id"] == card["block_id"]]
    base = (
        data[block].notna().any(axis=1)
        if kind == "select_all"
        else data[card["id"]].notna() & ~data[card["id"]].isin(["Prefer not to say"])
    )
    selected = (
        data[card["id"]].eq(True).fillna(False)
        if kind == "select_all"
        else data[card["id"]].isin(card["ordered_values"][-2:])
        if kind == "ordered_scale"
        else data[card["id"]].eq(row["value"]).fillna(False)
    )
    counts = [
        (
            int((base & data[row["group_variable"]].isin(vs)).sum()),
            int((base & data[row["group_variable"]].isin(vs) & selected).sum()),
        )
        for vs in row["groups"].values()
    ]
    ci = interval(counts, spec["threshold"])
    api_e = {
        "consistent": "support",
        "inconsistent": "contradict",
        "no_clear_difference": "inconclusive",
        "unavailable": "unavailable",
    }[contrast["label"]]
    if evidence_direction(ci, spec["direction"]) != api_e:
        raise ValueError("Independent direction oracle disagrees with API")
    boundary = stability(counts, spec["direction"], spec["threshold"])
    flags = []
    if min(n for n, k in counts) < 30:
        flags.append("small_base")
    if ci is None:
        flags.append("suppression")
    if not boundary["stable"]:
        flags.append("boundary_sensitive")
    result = {
        **spec,
        "counts": counts,
        "ci": ci,
        "E": api_e,
        "S": s,
        "diagnostic_flags": flags,
        "stability": boundary,
    }
    check = materialize(
        annotate_candidate(
            {
                "scenario": "single",
                "analyses": [result],
                "family": digest([spec["core"]]),
                "scenario_keys": [],
            }
        ),
        "inventory",
        seed,
        metadata,
    )
    validate_reference(check, evidence)
    return result
