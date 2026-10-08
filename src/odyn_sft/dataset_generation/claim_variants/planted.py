"""Planted meaning changes on template claims, used to validate the meaning verifier.

Each change alters exactly one meaning element (direction, groups, threshold, a part,
causality, population, measure). The verifier must reject nearly all of them.
"""
from __future__ import annotations

import random
import re

from ..types import DataObject


def _swap(text: str, a: str, b: str) -> str:
    token = "\0SWAP\0"
    pattern_a, pattern_b = (re.compile(rf"(?<!\w){re.escape(x)}(?!\w)") for x in (a, b))
    return pattern_b.sub(a, pattern_a.sub(token, text)).replace(token, b)


def perturb(kind: str, claim: str, sig: DataObject, variables: list[DataObject], rng: random.Random) -> str | None:
    parts = sig["parts"]
    if kind == "flip_direction":
        flipped = _swap(_swap(claim, "more often", "less often"), "More than half", "Fewer than half")
        flipped = _swap(flipped, "more than half", "fewer than half")
        return flipped if flipped != claim else None
    if kind == "swap_groups":
        diff = [p for p in parts if p["comparison"]["type"] == "difference" and str(p["a"]) in claim and str(p["b"]) in claim]
        return _swap(claim, str(diff[0]["a"]), str(diff[0]["b"])) if diff else None
    if kind == "change_threshold":
        changed = re.sub(r"\b(half)\b", "a third", claim, count=1)
        return changed if changed != claim else None
    if kind == "drop_part":
        sentences = re.split(r"(?<=[.?!])\s+", claim.strip())
        return " ".join(sentences[:-1]) if len(sentences) > 1 and len(parts) + len(sig["gaps"]) > 1 else None
    if kind == "add_causal":
        return None if sig["causal_clause"] else claim.rstrip() + " This is because of differences in their media exposure."
    if kind == "drop_population":
        if not any(p["population"] for p in parts):
            return None
        kept = [s for s in re.split(r"(?<=[.?!])\s+", claim.strip()) if not re.search(r"\brestrict|\bonly among|\bamong people", s, re.I)]
        dropped = " ".join(kept)
        return dropped if dropped != claim.strip() else None
    if kind == "swap_measure":
        quoted = re.findall(r"'([^']+)'", claim)
        labels = {v["label"] for v in variables}
        current = next((q for q in quoted if q in labels), None)
        if not current:
            return None
        others = sorted(labels - {current})
        return claim.replace(f"'{current}'", f"'{rng.choice(others)}'", 1) if others else None
    raise ValueError(kind)


PERTURBATIONS = ["flip_direction", "swap_groups", "change_threshold", "drop_part", "add_causal",
                 "drop_population", "swap_measure"]
