"""A small deterministic survey whose response rules are known, so counts can be checked independently."""

import numpy as np
import pandas as pd

from ...common.storage import digest


def fixture():
    """Synthetic data with deliberately known counts, not a resample of client data."""
    i = np.arange(125)
    group = np.where(i < 60, "A", np.where(i < 120, "B", "Tiny"))
    selected = (i < 48) | ((i >= 60) & (i < 78)) | (i == 120)
    data = pd.DataFrame(
        {
            "respondent_id": [f"synthetic_{v}" for v in i],
            "segment": group,
            "selected": pd.array(np.where(selected, True, None), dtype="boolean"),
            "other": pd.array(np.where(~selected, True, None), dtype="boolean"),
            "choice": np.where(selected, "Pass A", "Pass B"),
            "satisfaction": np.where((i < 40) | ((i >= 60) & (i < 80)), "Agree", "Disagree"),
            "awareness": np.where((i < 40) | ((i >= 60) & (i < 80)), "Aware", "Unaware"),
            "nominal": np.where(i % 2 == 0, "Car", "Train"),
            "age_band": np.array(["18-34", "35-54", "55+"])[i % 3],
            "weight": np.where(i < 60, 1.0, 2.0),
        }
    )
    # Five people have no recorded answer in the checkbox block. Single-choice base differs.
    data.loc[120:, ["selected", "other"]] = pd.NA
    cards = []
    for identifier, label, kind, values, ordered, block in [
        ("segment", "Synthetic segment", "single", ["A", "B", "Tiny"], [], "segment"),
        ("selected", "Selected benefit", "select_all", [True], [], "benefits"),
        ("other", "Other benefit", "select_all", [True], [], "benefits"),
        ("choice", "First-choice pass", "single", ["Pass A", "Pass B"], [], "choice"),
        (
            "satisfaction",
            "Satisfaction agreement",
            "ordered_scale",
            ["Disagree", "Neutral", "Agree", "Strongly agree"],
            ["Disagree", "Neutral", "Agree", "Strongly agree"],
            "satisfaction",
        ),
        ("awareness", "Brand awareness", "single", ["Aware", "Unaware"], [], "awareness"),
        ("nominal", "Transport mode (nominal)", "single", ["Car", "Train"], [], "nominal"),
        (
            "age_band",
            "Age band",
            "single",
            ["18-34", "35-54", "55+"],
            ["18-34", "35-54", "55+"],
            "age_band",
        ),
    ]:
        cards.append(
            {
                "id": identifier,
                "label": label,
                "answer_type": kind,
                "values": values,
                "ordered_values": ordered,
                "block_id": block,
                "section": "demographics" if identifier == "age_band" else "synthetic",
                "denominator_rule": "observed_block_respondents"
                if kind == "select_all"
                else "observed_nonmissing_answers",
                "flags": ["unweighted"],
            }
        )
    metadata = {
        "version": "synthetic-use-case-fixture-0.1",
        "study": {
            "label": "Controlled synthetic use-case fixture",
            "n": 125,
            "weights": None,
            "period": "not_applicable",
            "scope": "synthetic development data; not consumer research",
            "limitations": [
                "unweighted",
                "synthetic",
                "true_routing_unknown",
                "no_causal_identification",
            ],
        },
        "variables": cards,
    }
    metadata["hash"] = digest(metadata)
    return data, metadata
