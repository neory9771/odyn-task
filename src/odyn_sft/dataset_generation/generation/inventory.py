"""Enumerate the group comparisons a study supports, and block cores already used elsewhere."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from ...common.storage import digest, file_hash, read_jsonl
from ..study import atomic_key, group_definitions, measure_key
from ..types import AnalysisRecord, DataObject, PathLike
from .evidence import evidence_direction, fixed_direction, interval, stability


def enumerate_analyses(
    data: pd.DataFrame, metadata: DataObject, seed: int, study_bindings: DataObject | None = None
) -> list[AnalysisRecord]:
    """Define each core and commit its direction before computing its reference."""
    cards = {c["id"]: c for c in metadata["variables"]}
    configured_groups = group_definitions(cards, study_bindings)
    blocks = defaultdict(list)
    for c in cards.values():
        blocks[c["block_id"]].append(c["id"])
    analyses, seen = [], set()
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
        base = (
            data[blocks[card["block_id"]]].notna().any(axis=1)
            if kind == "select_all"
            else data[card["id"]].notna() & ~data[card["id"]].isin(["Prefer not to say"])
        )
        for value in values:
            measure = measure_key(card["id"], kind, value)
            selected = (
                data[card["id"]].eq(True).fillna(False)
                if kind == "select_all"
                else data[card["id"]].isin(card["ordered_values"][-2:])
                if kind == "ordered_scale"
                else data[card["id"]].eq(value).fillna(False)
            )
            for gv, groups in configured_groups:
                if gv == card["id"]:
                    continue
                group_masks = [data[gv].isin(vs) for vs in groups.values()]
                for threshold in (False, True):
                    core = atomic_key(
                        measure, gv, groups, "threshold" if threshold else "difference"
                    )
                    if core in seen:
                        continue
                    seen.add(core)
                    direction = fixed_direction(core, seed)
                    commitment = digest({"core": core, "direction": direction, "seed": seed})
                    # Reference work starts only after the result-blind commitment above.
                    counts = [
                        (int((base & mask).sum()), int((base & mask & selected).sum()))
                        for mask in (group_masks[:1] if threshold else group_masks)
                    ]
                    if counts[0][0] == 0 or (not threshold and counts[1][0] == 0):
                        continue
                    ci = interval(counts, threshold)
                    s = []
                    flags = []
                    if kind == "select_all":
                        s += ["routing_ambiguity", "denominator_ambiguity"]
                    if min(n for n, k in (counts[:1] if threshold else counts)) < 30:
                        flags.append("small_base")
                    if ci is None:
                        flags.append("suppression")
                    audit = stability(counts, direction, threshold)
                    if not audit["stable"]:
                        flags.append("boundary_sensitive")
                    analyses.append(
                        {
                            "core": core,
                            "row": {
                                "card": card,
                                "value": value,
                                "measure": measure,
                                "group_variable": gv,
                                "groups": dict(list(groups.items())[:1]) if threshold else groups,
                            },
                            "threshold": threshold,
                            "direction": direction,
                            "direction_commitment": commitment,
                            "counts": counts,
                            "ci": ci,
                            "E": evidence_direction(ci, direction),
                            "S": s,
                            "diagnostic_flags": flags,
                            "stability": audit,
                        }
                    )
    return analyses


def blocked_cores(
    paths: Iterable[PathLike], metadata: DataObject
) -> tuple[set[str], list[DataObject]]:
    """Audit pilot specs and generated bindings, including old development drafts."""
    cards = {c["id"]: c for c in metadata["variables"]}
    blocked, reports = set(), []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Overlap corpus missing: {path}")
        rows = read_jsonl(path)
        parsed = 0
        for c in rows:
            blocked.update(c.get("scenario_keys", []))
            blocked.update(c.get("context_keys", []))
            if "bindings" in c and "group_variable" in c:
                for b in c["bindings"]:
                    measure = {"variable": b["variable"], "operation": b["operation"]}
                    threshold = b.get("threshold") is not None
                    groups = (
                        {b["required_groups"][0]: c["groups"][b["required_groups"][0]]}
                        if threshold
                        else c["groups"]
                    )
                    blocked.add(
                        atomic_key(
                            measure,
                            c["group_variable"],
                            groups,
                            "threshold" if threshold else "difference",
                        )
                    )
                parsed += 1
            elif c.get("scenario_keys") and not c.get("bindings"):
                parsed += 1
            elif "spec" in c and "outcomes" in c["spec"]:
                spec = c["spec"]
                for v in spec["outcomes"]:
                    if cards[v]["answer_type"] == "single":
                        raise ValueError(
                            f"Cannot recover single-choice response binding in {path}: {c['case_id']}"
                        )
                    blocked.add(
                        atomic_key(
                            measure_key(v, cards[v]["answer_type"], None),
                            spec["group_variable"],
                            spec["groups"],
                            "threshold" if spec.get("threshold") else "difference",
                        )
                    )
                parsed += 1
            else:
                raise ValueError(
                    f"Unsupported overlap record schema in {path}; supply an adapter, not silent skipping."
                )
        reports.append(
            {
                "path": str(path.resolve()),
                "sha256": file_hash(path),
                "records": len(rows),
                "parsed": parsed,
            }
        )
    return blocked, reports
