"""Combine analyses and scenarios into candidate cases and annotate their benchmark dimensions."""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations

from ...common.storage import digest
from ..types import AnalysisRecord, Candidate, DataObject
from .scenarios import SCENARIOS, scenario_applicable


def annotate_candidate(candidate: Candidate) -> Candidate:
    """Derive primary dimensions from propositions, not scenario allocation weights."""
    analyses, scenario = candidate["analyses"], candidate["scenario"]
    linked = bool(analyses) and scenario in SCENARIOS
    c = "linked" if linked else "multiple" if len(analyses) > 1 else "single"
    domains = {"statistical_inference"} if analyses else set()
    states = {a["E"] for a in analyses}
    availability = "available" if analyses else "unavailable"
    if scenario in SCENARIOS:
        domains.add(SCENARIOS[scenario][0])
        availability = "partially_available" if analyses else "unavailable"
        states.add("not_applicable" if analyses else "unavailable")
    candidate["dimensions"] = {
        "C": [c],
        "D": sorted(domains),
        "A": [availability],
        "E": sorted(states),
        "R": sorted({a["row"]["card"]["answer_type"] for a in analyses}),
        "S": sorted({s for a in analyses for s in a["S"]}),
    }
    return candidate


def candidates(
    analyses: list[AnalysisRecord],
    seed: int,
    blocked: set[str],
    scenario_rules: DataObject | None = None,
) -> list[Candidate]:
    """Enumerate meaningful specifications; unstable analyses are excluded.

    Scenario names choose prose only. Selection uses the derived dimensions.
    Pure absence claims name a concrete item/target and reserve its context core.
    """
    rules = scenario_rules or {}
    result, buckets = [], defaultdict(list)
    eligible = [a for a in analyses if a["core"] not in blocked and a["stability"]["stable"]]
    for a in eligible:
        variant = "suppression" if a["ci"] is None else "single"
        result.append(
            annotate_candidate(
                {
                    "scenario": variant,
                    "analyses": [a],
                    "family": digest([a["core"]]),
                    "scenario_keys": [],
                }
            )
        )
        if variant == "single" and not a["threshold"]:
            r = a["row"]
            buckets[(r["card"]["block_id"], r["group_variable"], digest(r["groups"]))].append(a)
            for scenario in ("causal", "population", "behaviour", "media", "transport"):
                if scenario_applicable(scenario, a, rules):
                    result.append(
                        annotate_candidate(
                            {
                                "scenario": scenario,
                                "analyses": [a],
                                "family": digest([a["core"]]),
                                "scenario_keys": [],
                            }
                        )
                    )
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
        pairs = [
            (a, b)
            for a, b in combinations(rows[:24], 2)
            if a["row"]["card"]["id"] != b["row"]["card"]["id"]
        ]
        rng.shuffle(pairs)
        for a, b in pairs[:60]:
            result.append(
                annotate_candidate(
                    {
                        "scenario": "compound",
                        "analyses": [a, b],
                        "family": digest(sorted([a["core"], b["core"]])),
                        "scenario_keys": [],
                    }
                )
            )
    for context in eligible:
        if not context["threshold"] or context["ci"] is None:
            continue
        for scenario in ("transactions", "historical"):
            if not scenario_applicable(scenario, context, rules):
                continue
            key = digest(
                {
                    "missing_evidence": scenario,
                    "context_core": context["core"],
                    "years": [2020, 2022] if scenario == "historical" else None,
                }
            )
            if key in blocked:
                continue
            result.append(
                annotate_candidate(
                    {
                        "scenario": scenario,
                        "analyses": [],
                        "family": key,
                        "scenario_keys": [key],
                        "context_keys": [context["core"]],
                        "context": context,
                        "target": next(iter(context["row"]["groups"])),
                    }
                )
            )
    rng.shuffle(result)
    return result


def keys(candidate: Candidate) -> list[str]:
    """Resources reserved by a case, including genuine missing-evidence targets."""
    return (
        [a["core"] for a in candidate["analyses"]]
        + candidate["scenario_keys"]
        + candidate.get("context_keys", [])
    )
