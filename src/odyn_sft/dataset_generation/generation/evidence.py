"""Independent interval arithmetic and evidence direction (support/contradict/inconclusive) for a fixed assertion."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ...common.storage import digest
from ..types import Counts, EvidenceDirection, StabilityAudit


def wilson_oracle(k: int, n: int) -> list[float]:
    """Independent check via scipy; the Survey API itself uses statsmodels."""
    from scipy.stats import binomtest

    interval = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return [float(interval.low), float(interval.high)]


def interval(counts: Sequence[tuple[int, int]], threshold: bool = False) -> list[float] | None:
    """95% interval for A−B or A−0.5; None means suppression, not no effect."""
    na, ka = counts[0]
    if na < 10 or (not threshold and counts[1][0] < 10):
        return None
    la, ua = wilson_oracle(ka, na)
    if threshold:
        return [la - 0.5, ua - 0.5]
    nb, kb = counts[1]
    lb, ub = wilson_oracle(kb, nb)
    a, b = ka / na, kb / nb
    return [a - b - math.hypot(a - la, ub - b), a - b + math.hypot(ua - a, b - lb)]


def evidence_direction(ci: list[float] | None, direction: int) -> EvidenceDirection:
    """Assign E for a fixed assertion; evidence direction is not claim validity."""
    if ci is None:
        return "unavailable"
    if ci[0] <= 0 <= ci[1]:
        return "inconclusive"
    return "support" if (ci[0] > 0) == (direction > 0) else "contradict"


def fixed_direction(core: str, seed: int) -> int:
    """Commit an assertion deterministically without inspecting observed results."""
    # Only the core and seed enter this decision; no counts, intervals or labels.
    return (
        1
        if int(digest({"core": core, "seed": seed, "purpose": "claim-direction"})[:16], 16) % 2
        else -1
    )


def stability(counts: Counts, direction: int, threshold: bool = False) -> StabilityAudit:
    """Check local classification sensitivity while keeping observed bases fixed."""
    original = evidence_direction(interval(counts, threshold), direction)
    variants = set()
    for i, (n, k) in enumerate(counts[:1] if threshold else counts):
        for step in (-1, 1):
            if 0 <= k + step <= n:
                perturbed = list(counts)
                perturbed[i] = (n, k + step)
                variants.add(evidence_direction(interval(perturbed, threshold), direction))
    return {
        "rule": "one_response_flip_fixed_observed_base",
        "stable": all(x == original for x in variants),
        "original": original,
        "perturbed_states": sorted(variants),
        "limitation": "Does not test unknown routing, changed bases or measurement error.",
    }
