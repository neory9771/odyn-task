"""Native Python process execution and analysis-semantic scoring."""

from __future__ import annotations

import pandas as pd

from ..survey_api import SurveyAPI, run_program
from ..survey_api.tracing import mask_hash
from .types import DataObject


def execute(program: str, data: pd.DataFrame, metadata: DataObject) -> DataObject:
    """Execute candidate Python in a separate process with API population traces."""
    return run_program(program, data, metadata, collect_trace=True)


def check_analysis_semantics(
    contract: DataObject, evidence: DataObject, data: pd.DataFrame, metadata: DataObject
) -> DataObject:
    """Accept equivalent executed populations/estimands, not identical source code."""
    from .meaning import VERSION as meaning_version
    from .meaning import semantics

    if contract.get("version") == meaning_version:
        return semantics(contract, evidence, data, metadata)
    api = SurveyAPI(data, metadata)
    trace = evidence.get("analysis_trace", {})
    missing = []
    for required in contract["required_analyses"]:
        targets = []
        for values in required["groups"].values():
            population = api.where(required["group_variable"], values).mask
            base = api._analysis_base(required["variable"], required["operation"]) & population
            targets.append((mask_hash(population), mask_hash(base)))
        matched = False
        for r in evidence["results"]:
            if r["kind"] != "contrast":
                continue
            t = trace.get(r["estimate_id"])
            if (
                not t
                or t["variable"] != required["variable"]
                or t["operation"] != required["operation"]
            ):
                continue
            groups = {g["group"]: (g["population_hash"], g["base_hash"]) for g in t["groups"]}
            if required["threshold"] is not None:
                matched = (
                    groups.get(r["a"]) == targets[0]
                    and r["b"] == required["threshold"]
                    and r["expect"] == required["expect"]
                )
            else:
                direct = (groups.get(r["a"]), groups.get(r["b"])) == tuple(targets) and r[
                    "expect"
                ] == required["expect"]
                reverse = (groups.get(r["b"]), groups.get(r["a"])) == tuple(targets) and r[
                    "expect"
                ] == (">" if required["expect"] == "<" else "<")
                matched = direct or reverse
            if matched:
                break
        if not matched:
            missing.append(required["component_id"])
    return {
        "passed": not missing,
        "missing_components": missing,
        "note": "Deterministic measured semantics; the judge checks decomposition, gap/proxy meaning and inference scope.",
    }
