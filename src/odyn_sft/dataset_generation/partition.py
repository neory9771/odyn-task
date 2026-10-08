"""Assign atomic analysis families before references, E, cases or prose exist."""

from __future__ import annotations

from ..common.storage import digest
from .generation import fixed_direction
from .study import atomic_key, group_definitions, measure_key
from .types import DataObject


def enumerate_family_specs(
    metadata: DataObject, seed: int, study_bindings: DataObject | None = None
) -> list[DataObject]:
    """Metadata-only inventory: no respondent rows, counts or evidence labels."""
    cards = {c["id"]: c for c in metadata["variables"]}
    configured_groups = group_definitions(cards, study_bindings)
    result, seen = [], set()
    for card in cards.values():
        kind = card["answer_type"]
        if (
            kind not in {"select_all", "single", "ordered_scale"}
            or card["section"] == "demographics"
        ):
            continue
        values = (
            [None] if kind != "single" else [v for v in card["values"] if v != "Prefer not to say"]
        )
        if kind == "single" and (len(values) < 2 or all(str(v).isdigit() for v in values)):
            continue
        if kind == "ordered_scale" and not card.get("ordered_values"):
            continue
        for value in values:
            measure = measure_key(card["id"], kind, value)
            for gv, groups in configured_groups:
                if gv == card["id"]:
                    continue
                for threshold in (False, True):
                    target = dict(list(groups.items())[:1]) if threshold else groups
                    core = atomic_key(
                        measure, gv, target, "threshold" if threshold else "difference"
                    )
                    if core in seen:
                        continue
                    seen.add(core)
                    direction = fixed_direction(core, seed)
                    result.append(
                        {
                            "core": core,
                            "row": {
                                "card": card,
                                "value": value,
                                "measure": measure,
                                "group_variable": gv,
                                "groups": target,
                            },
                            "threshold": threshold,
                            "direction": direction,
                            "direction_commitment": digest(
                                {"core": core, "direction": direction, "seed": seed}
                            ),
                        }
                    )
    return result


def assign_owners(
    specs: list[DataObject],
    ratios: dict[str, float],
    seed: int,
    heldout_blocks: list[str],
    blocked: set[str],
) -> dict[str, str]:
    """Frozen family ownership independent of wording, evidence and selection.

    Held-out outcome blocks are test-only. Other atomic families are assigned
    by a seeded hash; compounds can only use atoms with the same owner.
    """
    owners = {}
    total = sum(ratios.values())
    cuts = (ratios["train"] / total, (ratios["train"] + ratios["validation"]) / total)
    for spec in specs:
        core = spec["core"]
        if core in blocked:
            continue
        if spec["row"]["card"]["block_id"] in heldout_blocks:
            split = "test"
        else:
            u = (
                int(digest({"core": core, "seed": seed, "purpose": "family-partition"})[:16], 16)
                / 2**64
            )
            split = "train" if u < cuts[0] else "validation" if u < cuts[1] else "test"
        owners[core] = split
    return owners
