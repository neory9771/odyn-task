"""Coverage-constrained sampling over C,D,A,E,R with targeted S minima.

Scenario names are language templates, not allocation buckets. Requirements
count cases containing a tag, so multi-label margins need not sum to size.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np

from .types import Candidate, DataObject

DEFAULT_MINIMA = {
    "C": {"single": 100, "multiple": 100, "linked": 100},
    "D": {
        "statistical_inference": 600,
        "causal_identification": 30,
        "survey_population_inference": 30,
        "measurement_construct_validity": 30,
        "transportability": 30,
        "temporal_comparability": 20,
    },
    "A": {"available": 400, "partially_available": 120, "unavailable": 20},
    "E": {"support": 200, "contradict": 200, "inconclusive": 150, "unavailable": 20},
    "R": {"select_all": 300, "single": 100, "ordered_scale": 50},
    "S": {"routing_ambiguity": 200, "denominator_ambiguity": 200},
    "holdout": {"held_out": 100},
}
DEFAULT_INTERACTIONS = {
    "C=multiple&E=support&E=contradict": 30,
    "C=linked&A=partially_available": 100,
}


def scaled_minima(minima: dict[str, dict[str, int]], fraction: float) -> dict[str, dict[str, int]]:
    """Scale explicit evaluation minima for a smaller development split."""
    return {
        d: {v: math.ceil(n * fraction) for v, n in values.items()} for d, values in minima.items()
    }


def tags(candidate: Candidate, heldout_blocks: list[str]) -> set[str]:
    """Make primary methodological tags independently of the template's name."""
    dimensions = candidate["dimensions"]
    result = {f"{d}={v}" for d in ("C", "D", "A", "E", "R", "S") for v in dimensions[d]}
    blocks = [a["row"]["card"]["block_id"] for a in candidate["analyses"]]
    if blocks and all(b in heldout_blocks for b in blocks):
        result.add("holdout=held_out")
    return result


def requirements(minima: dict[str, dict[str, int]], interactions: dict[str, int]) -> dict[str, int]:
    return {
        **{f"{d}={v}": n for d, values in minima.items() for v, n in values.items()},
        **interactions,
    }


def resources(candidate: Candidate) -> set[str]:
    # Context cores are also reserved: reusing an item/group just as a historical
    # framing must not turn it into a fresh independent analytic family.
    return (
        {a["core"] for a in candidate["analyses"]}
        | set(candidate["scenario_keys"])
        | set(candidate.get("context_keys", []))
    )


def select(
    pool: list[Candidate],
    size: int,
    seed: int,
    minima: dict[str, dict[str, int]],
    interactions: dict[str, int],
    heldout_blocks: list[str],
    forbidden: set[str] | None = None,
) -> tuple[list[Candidate], set[str], list[DataObject]]:
    """Satisfy scarce coverage needs, then balance feasible joint strata.

    Ties are seeded/random. No response direction or evidence label is changed.
    This is a greedy feasibility check, not a proof of globally optimal capacity.
    """
    rng = random.Random(seed)
    need = requirements(minima, interactions)
    feature_sets = [tags(c, heldout_blocks) for c in pool]
    core_sets = [resources(c) for c in pool]
    reserved = set(forbidden or ())
    active = np.array([not (cores & reserved) for cores in core_sets], dtype=bool)
    incidence = {
        r: np.array([i for i, t in enumerate(feature_sets) if set(r.split("&")) <= t], dtype=int)
        for r in need
    }
    core_members = defaultdict(list)
    strata = defaultdict(list)
    for i, c in enumerate(pool):
        for core in core_sets[i]:
            core_members[core].append(i)
        key = tuple(sorted(t for t in feature_sets[i] if not t.startswith(("S=", "holdout="))))
        strata[key].append(i)
    attained, stratum_counts = Counter(), Counter()
    chosen = []
    while len(chosen) < size:
        deficit = {r: n - attained[r] for r, n in need.items() if attained[r] < n}
        if deficit:
            available = {r: incidence[r][active[incidence[r]]] for r in deficit}
            impossible = [r for r in deficit if len(available[r]) < deficit[r]]
            if impossible:
                break
            tightest = min(deficit, key=lambda r: (len(available[r]) / deficit[r], rng.random()))
            options = available[tightest].tolist()
            rng.shuffle(options)
            # Choose the candidate covering the most outstanding needs, with
            # random ties. This is based on annotations, never effect extremity.
            pick = max(
                options,
                key=lambda i: sum(
                    1 / need[r] for r in deficit if set(r.split("&")) <= feature_sets[i]
                ),
            )
        else:
            eligible_strata = [(k, [i for i in ids if active[i]]) for k, ids in strata.items()]
            eligible_strata = [(k, ids) for k, ids in eligible_strata if ids]
            if not eligible_strata:
                break
            rng.shuffle(eligible_strata)
            _, options = min(eligible_strata, key=lambda x: stratum_counts[x[0]])
            pick = rng.choice(options)
        candidate = pool[pick]
        chosen.append(candidate)
        for r in need:
            attained[r] += int(set(r.split("&")) <= feature_sets[pick])
        stratum_key = tuple(
            sorted(t for t in feature_sets[pick] if not t.startswith(("S=", "holdout=")))
        )
        stratum_counts[stratum_key] += 1
        for core in core_sets[pick]:
            active[core_members[core]] = False
        active[pick] = False
        reserved.update(core_sets[pick])
    shortages = [
        {"requirement": r, "minimum": n, "attained": attained[r]}
        for r, n in need.items()
        if attained[r] < n
    ]
    if len(chosen) != size:
        shortages.append({"requirement": "size", "minimum": size, "attained": len(chosen)})
    return chosen, reserved, shortages


def coverage(
    selected: list[Candidate],
    size: int,
    minima: dict[str, dict[str, int]],
    interactions: dict[str, int],
    heldout_blocks: list[str],
) -> DataObject:
    feature_sets = [tags(c, heldout_blocks) for c in selected]
    counts = {
        r: sum(set(r.split("&")) <= t for t in feature_sets)
        for r in requirements(minima, interactions)
    }
    checks = {r: counts[r] >= n for r, n in requirements(minima, interactions).items()}
    checks["size"] = len(selected) == size
    checks["unique_families"] = len({c["family"] for c in selected}) == len(selected)
    return {
        "counts": counts,
        "checks": checks,
        "counting_unit": "cases containing all requested tags",
    }
