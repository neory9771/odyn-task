"""Category ordering from codebook provenance or explicit response meaning.

The order is metadata only: response values are never recoded, and ordinal
positions must not be interpreted as equally spaced numerical measurements.
Numeric codes alone do not establish a scale's substantive direction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

REFUSALS = {"Prefer not to say", "Don't know / Prefer not to say"}
COHORTS = ["Generation Z", "Millennials", "Generation X", "Baby boomers", "The Silent Generation"]
FREQUENCY = ["Less than once a year", "Once a year or about once a year", "Multiple times a year"]
COMFORT = [
    "Finding it very difficult on present income",
    "Finding it difficult on present income",
    "Having just enough income not to struggle, but cannot save much",
    "Living comfortably on present income",
    "Living very comfortably on present income",
]
CONSIDERATION = [
    "It's not one I would ever consider",
    "It's one I would be unlikely to consider",
    "It's one I would consider but after most others",
    "It's one I would consider equally with others",
    "It's one I would consider above most others",
    "Its the only one I'd consider",
]
NPS = ["Detractors", "Passives", "Promoters"]


@dataclass(frozen=True)
class CategoryOrdering:
    ordered_values: list[str] | None
    basis: str | None
    unranked_values: list[str]
    flags: tuple[str, ...] = ()


def _income_lower_bound(value: str) -> int | None:
    if value == "No income":
        return 0
    match = re.fullmatch(r"\$([\d,]+)\s*-\s*\$([\d,]+)", value)
    if match:
        lower, upper = (int(part.replace(",", "")) for part in match.groups())
        return lower if 0 < lower <= upper else None
    match = re.fullmatch(r"\$([\d,]+) or more", value)
    return int(match[1].replace(",", "")) if match else None


def category_ordering(
    values: list[str], source_order: str, *, answer_type: str
) -> CategoryOrdering:
    """Return a declared/inferred order while preserving every allowed value.

    Named semantic scales match their response text, not variable IDs. Missing
    response categories stay missing; we never add an unobserved scale point.
    Codebook-declared ordering is retained independently of response format.
    """
    if answer_type == "select_all":
        return CategoryOrdering(None, None, [])
    if answer_type == "rank":
        return CategoryOrdering(None, "unresolved_scale_direction", [])

    substantive = [value for value in values if value not in REFUSALS]

    def result(order: list[str], basis: str, flags: tuple[str, ...] = ()) -> CategoryOrdering:
        return CategoryOrdering(order, basis, [v for v in values if v not in order], flags)

    # Honour the supplied sequence, not an alphabetical or inferred code sort.
    # "No rules" is a valid original answer but has no duration rank.
    if source_order == "known_scale":
        order = [value for value in substantive if value != "No rules"]
        if len(order) >= 2:
            flags: tuple[str, ...] = ()
            if set(order) == {"A", "B", "C1", "C2", "D", "E"}:
                flags = ("ordering_meaning_unverified",)
            if "Between 1 and 2 weeks" in order and "2 weeks or more" in order:
                flags += ("category_boundary_ambiguous",)
            return result(order, "source_known_scale", flags)

    bounds = [_income_lower_bound(value) for value in substantive]
    if len(substantive) >= 2 and all(bound is not None for bound in bounds):
        order = sorted(substantive, key=lambda value: _income_lower_bound(value) or 0)
        return result(order, "income_band_bounds")

    for scale, basis in (
        (COHORTS, "chronological_cohort_younger_to_older"),
        (FREQUENCY, "response_meaning"),
        (COMFORT, "response_meaning"),
        (CONSIDERATION, "response_meaning"),
        (NPS, "response_meaning"),
    ):
        if len(substantive) >= 2 and set(substantive) <= set(scale):
            return result([value for value in scale if value in substantive], basis)

    return CategoryOrdering(None, None, [])
