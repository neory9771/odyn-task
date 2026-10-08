"""Rich population/estimand contracts, independent of generated Python spelling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..survey_api import SurveyAPI
from ..survey_api.filters import Filter
from ..survey_api.tracing import mask_hash

VERSION = "population-analysis-contract-2.0.0"


def filter_value(spec: dict[str, Any], api: SurveyAPI) -> Filter:
    op = spec["op"]
    if op == "eq":
        return api.where(spec["variable"], spec["values"])
    if op == "eligible":
        return api.eligible(spec["variable"])
    if op == "not":
        return ~filter_value(spec["arg"], api)
    left, right = (filter_value(arg, api) for arg in spec["args"])
    if op == "and":
        return left & right
    if op == "or":
        return left | right
    raise ValueError("Unknown population expression: " + op)


def expected_groups(required: dict[str, Any], api: SurveyAPI) -> dict[str, Filter]:
    if required.get("group_filters"):
        return api.group(
            {name: filter_value(f, api) for name, f in required["group_filters"].items()}
        )
    return api.group(required["group_variable"], required["groups"])


def semantics(
    contract: dict[str, Any], evidence: dict[str, Any], data: pd.DataFrame, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Check each required measured contrast against respondent hashes and meaning.

    Proxy rationale and causal/synthesis adequacy still require semantic judging.
    Extra nonrequired evidence is allowed. Contrast type prevents an adjusted
    odds ratio or trend from masquerading as a requested share difference.
    """
    api, missing, failures = SurveyAPI(data, metadata), [], []
    traces = evidence.get("analysis_trace", {})
    for required in contract["required_analyses"]:
        groups = expected_groups(required, api)
        among = (
            filter_value(required["population"], api).mask
            if required.get("population")
            else pd.Series(True, index=data.index)
        )
        base = api._analysis_base(required["variable"], required["operation"])
        targets = [(mask_hash(f.mask), mask_hash(base & f.mask & among)) for f in groups.values()]
        matched = False
        for row in evidence["results"]:
            if row["kind"] != "contrast" or row.get("contrast_type") not in {
                "difference",
                "threshold",
            }:
                continue
            trace = traces.get(row.get("estimate_id"))
            if (
                not trace
                or trace["variable"] != required["variable"]
                or trace["operation"] != required["operation"]
            ):
                continue
            observed = {g["group"]: (g["population_hash"], g["base_hash"]) for g in trace["groups"]}
            if required["threshold"] is not None:
                matched = (
                    observed.get(row["a"]) == targets[0]
                    and row["b"] == required["threshold"]
                    and row["expect"] == required["expect"]
                )
            else:
                direction = {">": "<", "<": ">", "!=": "!=", None: None}[required["expect"]]
                matched = (
                    tuple(observed.get(row.get(k)) for k in ("a", "b")) == tuple(targets)
                    and row["expect"] == required["expect"]
                    or tuple(observed.get(row.get(k)) for k in ("b", "a")) == tuple(targets)
                    and row["expect"] == direction
                )
            if matched:
                break
        if not matched:
            missing.append(required["component_id"])
            failures.append(
                {"component_id": required["component_id"], "category": "measured_analysis_mismatch"}
            )
    # API lessons also cover records without a share contrast. Match semantics,
    # not arbitrary evidence IDs or the exact text of comments.
    for target in contract.get("required_records", []):
        matches = [
            r
            for r in evidence["results"]
            if r["kind"] == target["kind"]
            and all(r.get(k) == v for k, v in target.get("fields", {}).items())
        ]
        if target.get("expected_trace"):
            expected = target["expected_trace"]
            signature = lambda t: sorted(
                (g["population_hash"], g["base_hash"]) for g in t["groups"]
            )
            matches = [
                r
                for r in matches
                if r["id"] in traces
                and traces[r["id"]]["variable"] == expected["variable"]
                and traces[r["id"]]["operation"] == expected["operation"]
                and signature(traces[r["id"]]) == signature(expected)
            ]
        if not matches:
            missing.append(target["component_id"])
            failures.append(
                {"component_id": target["component_id"], "category": "missing_required_record"}
            )
    return {
        "passed": not missing,
        "missing_components": missing,
        "failures": failures,
        "note": "Executed measured populations, estimands and required record types; rationale, natural-claim validity and synthesis require reviewed semantic grading.",
    }
