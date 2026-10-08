"""Shared response interpretation and metadata-derived caveats for prompts."""

from __future__ import annotations

from typing import Any

RESPONSE_METADATA_NOTE: str = (
    "single records one category; select_all records an option selection (True); "
    "ordered_scale denotes an agreement-scale response; rank records rank-coded "
    "responses whose direction is unverified. Response type and category order are "
    "independent. ordered true uses values as the scale sequence; ordered false "
    "means no usable order; an ordered list supplies an explicit sequence or subset. "
    "Orders preserve codebook declarations or "
    "follow interpretable response wording and numeric band bounds. Use the supplied "
    "sequence rather than guessing direction from numeric codes. Responses outside "
    "a supplied order remain valid but are excluded from ordered "
    "calculations; category positions do not imply equal numerical spacing."
)

DENOMINATOR_NOTE: str = (
    "For select_all, the denominator includes respondents with at least one "
    "recorded answer in the item's block. Within that observed block base, blank "
    "options count as unselected; respondents with no recorded block answers are "
    "excluded. Other items use nonmissing answers, excluding the exact response "
    "'Prefer not to say'. top_box further restricts the base to the resolved "
    "ordered response sequence and counts its last k categories as the numerator. Requested "
    "groups and filters restrict each base. Denominators can differ across items "
    "and do not establish true routing or population coverage. Interpret each "
    "estimate using its returned n and denominator rule."
)


def study_caveats(cards: list[dict[str, Any]]) -> str | None:
    """Explain applicable variable warnings once without changing API flags."""
    notes: list[str] = []
    repeated_blocks = sorted(
        {card["block_id"] for card in cards if "duplicate_block" in card.get("flags", [])}
    )
    if repeated_blocks:
        notes.append(
            "Potentially repeated blocks: "
            + ", ".join(repeated_blocks)
            + ". Their relationship is undocumented; matching item labels do not "
            "establish that they are interchangeable measures."
        )
    for card in cards:
        flags = card.get("flags", [])
        if "ordering_meaning_unverified" in flags:
            notes.append(
                f"{card['id']}: category definitions and substantive ordering are "
                "unverified. A supplied code sequence does not establish increasing "
                "occupational status."
            )
        if "category_boundary_ambiguous" in flags:
            notes.append(
                f"{card['id']}: 'Between 1 and 2 weeks' and '2 weeks or more' "
                "overlap at two weeks; the boundary interpretation is undocumented."
            )
    if any("merged_labels" in card.get("flags", []) for card in cards):
        notes.append("Some response spelling and punctuation were normalised.")
    if any("acquiescence_risk" in card.get("flags", []) for card in cards):
        notes.append(
            "Agreement items may encourage agreement; this is a potential risk, not demonstrated bias."
        )
    return " ".join(notes) or None
