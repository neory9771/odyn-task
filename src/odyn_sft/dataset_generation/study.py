"""Survey bindings and scoring contracts shared by evaluation preparation."""

from __future__ import annotations

from typing import Any

from ..common.storage import digest
from .rubrics import build_rubric
from .types import DataObject, Groups


def group_definitions(
    cards: dict[str, DataObject], bindings: DataObject | None = None
) -> list[tuple[str, Groups]]:
    """Return configured disjoint groups whose exact IDs/codes exist in this study.

    The bundled 2022 preset is the compatibility default. It can cover more
    variables than a supplied catalogue; unavailable definitions are omitted.
    """
    from .comparison_groups import load_comparison_groups

    definitions = load_comparison_groups(bindings)["groups"]
    return [
        (row["variable"], row["groups"])
        for row in definitions
        if row["variable"] in cards
        and all(
            value in cards[row["variable"]]["values"]
            for values in row["groups"].values()
            for value in values
        )
    ]


def measure_key(v: str, kind: str, value: Any) -> DataObject:
    """Canonical response definition; a category share is distinct from top-box."""
    return {
        "variable": v,
        "operation": "top_box_2"
        if kind == "ordered_scale"
        else "selection"
        if kind == "select_all"
        else "value=" + str(value),
    }


def atomic_key(measure: DataObject, gv: str, groups: Groups, estimand: str) -> str:
    """Hash analytical meaning rather than assertion direction or prose."""
    if estimand == "threshold":
        # p(target)-0.5 never uses a comparator. Exclude comparator values and
        # cosmetic group labels, so Suburban/Urban and Suburban/Rural coincide.
        return digest(
            {
                "measure": measure,
                "group_variable": gv,
                "target_values": sorted(next(iter(groups.values()))),
                "estimand": "share_minus_0.5",
            }
        )
    return digest(
        {
            "measure": measure,
            "group_variable": gv,
            "groups": sorted((k, sorted(v)) for k, v in groups.items()),
            "estimand": estimand,
        }
    )


def measurement_phrase(row: DataObject) -> str:
    """Describe the recorded response, avoiding unsupported behavioural proxies."""
    c = row["card"]
    if c["answer_type"] == "select_all":
        return f"record selecting the checkbox item {c['label']!r}"
    if c["answer_type"] == "ordered_scale":
        return f"give one of the top two agreement responses to the item {c['label']!r}"
    return f"answer {row['value']!r} to {c['label']!r}"


def rubric(case: DataObject, evidence: DataObject) -> DataObject:
    """Require every material component while permitting grounded alternatives."""
    targets = [
        "Offer a perspective on the claim, citing relevant supporting, contradicting and inconclusive evidence. Do not declare the whole claim true or false.",
        "Address all requested components, including scope/measurement gaps. Keep recorded responses distinct from constructs, actual behaviour, population prevalence and causal explanations.",
    ]
    numbers = []
    for binding in case["bindings"]:
        estimates = [
            r
            for r in evidence["results"]
            if r["kind"] == "estimate" and r["subclaim"] == binding["subclaim"]
        ]
        contrasts = [
            r
            for r in evidence["results"]
            if r["kind"] == "contrast" and r["subclaim"] == binding["subclaim"]
        ]
        for r in estimates:
            numbers.append(
                {
                    "kind": "estimate",
                    "variable": r["variable"],
                    "operation": r["operation"],
                    "groups": [
                        x for x in r["estimates"] if x["group"] in binding["required_groups"]
                    ],
                }
            )
        numbers += [{"kind": "contrast", "reference": r} for r in contrasts]
    if case["gap_type"]:
        targets.append(
            "Explain the "
            + case["gap_type"]
            + " limitation; do not substitute a related measured quantity as proof of the missing component."
        )
    result = build_rubric(
        case["case_id"],
        "client_1000",
        targets,
        numerical=numbers,
        synthetic=False,
        optional=[
            "Relevant contextual findings are optional, clearly labelled and grounded. Extra reference groups are not mandatory."
        ],
        prohibited=[
            "Whole-claim true/false verdict, including paraphrased verdicts.",
            "Invented numbers, waves, transactions, campaign delivery, weights or causal identification.",
        ],
    )
    result["evidence_labels_are_component_annotations_not_claim_verdicts"] = True
    return result
