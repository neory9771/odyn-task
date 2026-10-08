"""Claim scenarios (what a client asserts beyond the data) and opening styles, with their survey bindings."""

from __future__ import annotations

from ..types import AnalysisRecord, DataObject

SCENARIOS = {
    "causal": (
        "causal_identification",
        "The difference is caused by membership in these groups.",
        "Cross-sectional responses do not identify a causal effect of group membership.",
        "partially_available",
    ),
    "population": (
        "survey_population_inference",
        "These percentages also describe all US consumers.",
        "No documented selection and weighting design identifies US population prevalence.",
        "partially_available",
    ),
    "behaviour": (
        "measurement_construct_validity",
        "These responses establish the corresponding actual purchase rate.",
        "No linked purchase observations validate this response as an actual purchase measure.",
        "partially_available",
    ),
    "media": (
        "causal_identification",
        "The difference is explained by the campaign not reaching this demographic.",
        "No campaign exposure records or causal design identify this explanation.",
        "partially_available",
    ),
    "transport": (
        "transportability",
        "The same pattern applies to consumers outside the United States.",
        "No target-setting observations or transport model justify this extension.",
        "partially_available",
    ),
    "transactions": (
        "measurement_construct_validity",
        "Actual transaction records verify the specified purchase.",
        "The supplied survey has no linked transaction records.",
        "unavailable",
    ),
    "historical": (
        "temporal_comparability",
        "The measured response share increased between 2020 and 2022.",
        "Only the 2022 survey is supplied; historical observations and harmonisation evidence are unavailable.",
        "unavailable",
    ),
}


STYLES = [
    "What does the supplied research show about this claim? ",
    "Give an evidence perspective on the following statement: ",
    "Which findings bear on this claim, and what remains unresolved? ",
    "Assess the research evidence relevant to the following claim: ",
    "Explain the supporting and contradicting findings for this statement: ",
    "Using the supplied survey, discuss the evidence for this claim: ",
]


def scenario_applicable(scenario: str, analysis: AnalysisRecord, rules: DataObject) -> bool:
    """Require verified survey bindings for substantive behavioural/media claims.

    No eligibility is inferred from an opaque variable ID or a vague item label.
    Configured bindings are metadata decisions that still require expert review.
    """
    row = analysis["row"]
    card = row["card"]
    if scenario in {"population", "transport", "historical"}:
        return True
    if scenario == "causal":
        return row["group_variable"] in rules.get("causal_group_variables", [])
    if scenario == "media":
        return row["group_variable"] in rules.get("media_group_variables", []) and card[
            "block_id"
        ] in rules.get("media_outcome_blocks", [])
    if scenario in {"behaviour", "transactions"}:
        if card["block_id"] in rules.get("purchase_outcome_blocks", []):
            return True
        return row["value"] in rules.get("purchase_values", {}).get(card["id"], [])
    return False


def validate_scenario_bindings(rules: DataObject, metadata: DataObject) -> None:
    """Fail on misspelled curation bindings rather than silently changing coverage."""
    cards = {c["id"]: c for c in metadata["variables"]}
    blocks = {c["block_id"] for c in cards.values()}
    for name in ("causal_group_variables", "media_group_variables"):
        if any(v not in cards for v in rules.get(name, [])):
            raise ValueError("Unknown scenario group binding: " + name)
    for name in ("media_outcome_blocks", "purchase_outcome_blocks"):
        if any(v not in blocks for v in rules.get(name, [])):
            raise ValueError("Unknown scenario block binding: " + name)
    for v, values in rules.get("purchase_values", {}).items():
        if v not in cards or any(x not in cards[v]["values"] for x in values):
            raise ValueError("Unknown purchase response binding: " + v)
