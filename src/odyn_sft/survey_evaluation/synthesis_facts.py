"""Code-built facts and yes/no expectations for judging step-2 perspectives.

Facts come only from the executed evidence pack. Directions describe each finding
relative to its claim part, never the truth of the claim. See
STEP2_SYNTHESIS_EVAL_SPEC.md sections 3 and 4.
"""
from __future__ import annotations

import re
from typing import Any

from ..survey_api.constants import EXPECT_SYMBOLS
from ..survey_api.service import SurveyAPI
from ..survey_api.statistics import holm_label
from .types import DataObject

VERSION = "synthesis-facts-1.0.1"

DIRECTIONS = {"consistent": "supports", "inconsistent": "contradicts", "no_clear_difference": "inconclusive",
              "not_tested": "descriptive", "unavailable": "unavailable"}
GAP_LABELS = {"not_measured", "proxy_variable", "caveat"}
SUPPRESSED_FLAGS = {"suppressed", "empty_segment"}
CAUSAL_CONTRASTS = {"association", "adjusted_difference"}
CAUSAL_WORDS = re.compile(r"\b(because|caus\w*|due to|driven by|leads? to|result(?:s|ed)? in|explain\w*)\b", re.I)


def _label_holm(contrast: DataObject) -> str | None:
    """API >= 0.4.0 stores label_holm; older packs get the same rule recomputed."""
    if "label_holm" in contrast:
        return contrast["label_holm"]
    expect = EXPECT_SYMBOLS.get(contrast.get("expect"), contrast.get("expect"))
    return holm_label(SurveyAPI._direction(contrast), contrast.get("p_holm"), expect)


def _part_description(contrast: DataObject, estimate: DataObject | None) -> str:
    """Readable claimed proposition rebuilt from the recorded contrast."""
    measure = (estimate or {}).get("variable_label") or contrast.get("variables") or "the measure"
    operation = (estimate or {}).get("operation", "")
    expect = contrast.get("expect")
    kind = contrast.get("contrast_type")
    if kind == "threshold":
        if operation == "mean_score":  # Threshold is a scale position, not a share.
            side = {">": "above", "<": "below"}.get(expect, "compared with")
            return f"mean score of {contrast['a']} on {measure!r} {side} {contrast['b']}"
        side = {">": "more than", "<": "fewer than"}.get(expect, "compared with")
        return f"{side} {contrast['b']:.0%} of {contrast['a']} on {measure!r} ({operation})"
    if kind == "association":
        x, y = contrast.get("variable_labels") or contrast.get("variables")
        return f"association between {x!r} and {y!r} (claimed {expect or 'unspecified'})"
    if kind == "trend":
        return f"trend across {', '.join(contrast.get('order') or [])} on {measure!r} (claimed {expect or 'unspecified'})"
    side = {">": "higher than", "<": "lower than", "!=": "different from"}.get(expect, "compared with")
    scale = " (odds ratio, adjusted for controls)" if kind == "adjusted_difference" else ""
    return f"{contrast['a']} {side} {contrast['b']} on {measure!r} ({operation}){scale}"


def build_facts(evidence: DataObject, claim: str, part_texts: dict[str, str] | None = None) -> DataObject:
    """Facts per claim part (subclaim) plus study-level scope, from one evidence pack."""
    results = evidence["results"]
    by_id = {r["id"]: r for r in results}
    part_texts = part_texts or {}
    parts: dict[str, DataObject] = {}

    def part(subclaim: str) -> DataObject:
        if subclaim not in parts:
            text = next((r.get("text") for r in results if r["kind"] == "subclaim" and r["subclaim"] == subclaim), None)
            parts[subclaim] = {"part_id": subclaim, "text": part_texts.get(subclaim), "subclaim_text": text,
                               "findings": [], "descriptive": [], "suppressed": [], "gaps": []}
        return parts[subclaim]

    contrasted = set()
    for r in results:
        if r["kind"] != "contrast":
            continue
        estimate = by_id.get(r.get("estimate_id"))
        contrasted.add(r.get("estimate_id"))
        label_holm = _label_holm(r)
        direction = DIRECTIONS.get(r["label"], "unavailable")
        p = part(r["subclaim"])
        if p["text"] is None:
            p["text"] = _part_description(r, estimate)
        p["findings"].append({
            "evidence_id": r["id"], "estimate_id": r.get("estimate_id"), "contrast_type": r.get("contrast_type"),
            "a": r.get("a"), "b": r.get("b"), "expect": r.get("expect"), "value": r.get("value"),
            "value_scale": r.get("value_scale", "difference"), "ci": r.get("ci"),
            "p_value": r.get("p_value"), "p_holm": r.get("p_holm"), "label": r["label"], "label_holm": label_holm,
            "direction": direction, "reason": r.get("reason"),
            "robust_after_adjustment": label_holm == r["label"] if direction in {"supports", "contradicts"} else None,
            "estimate_rows": [{k: row.get(k) for k in ("group", "n", "numerator", "value", "ci")}
                              for row in (estimate or {}).get("estimates", [])]})
    for r in results:
        if r["kind"] == "estimate":
            for row in r["estimates"]:
                if row["value"] is None and SUPPRESSED_FLAGS & set(row.get("flags", [])):
                    part(r["subclaim"])["suppressed"].append({"evidence_id": r["id"], "group": row["group"], "n": row["n"]})
            if r["id"] not in contrasted:
                part(r["subclaim"])["descriptive"].append({
                    "evidence_id": r["id"], "variable_label": r.get("variable_label"), "operation": r.get("operation"),
                    "rows": [{k: row.get(k) for k in ("group", "n", "numerator", "value", "ci")} for row in r["estimates"]]})
        elif r["kind"] == "note" and r.get("label") in GAP_LABELS:
            part(r["subclaim"])["gaps"].append({"evidence_id": r["id"], "type": r["label"], "concept": r.get("concept"),
                                                "reason": r.get("reason"), "variables": r.get("variables")})
    for p in parts.values():
        if p["text"] is None:
            gap = next((g for g in p["gaps"] if g["type"] != "caveat"), None)
            p["text"] = gap["concept"] if gap else p["subclaim_text"]
    findings = [f for p in parts.values() for f in p["findings"]]
    causal = (bool(CAUSAL_WORDS.search(claim))
              or any(g["type"] == "not_measured" and "causal" in (g["concept"] or "") for p in parts.values() for g in p["gaps"])
              or any(f["contrast_type"] in CAUSAL_CONTRASTS for f in findings))
    return {"version": VERSION, "claim": claim, "study": evidence.get("study", {}), "api_version": evidence.get("api_version"),
            "parts": list(parts.values()), "causal_relevant": causal}


# Items that only require something to be absent: a pass may have nothing to quote.
ABSENCE_TYPES = {"E5_no_verdict", "E7_suppression"}


def _item(item_id: str, kind: str, statement: str, critical: bool, **refs: Any) -> DataObject:
    return {"id": item_id, "type": kind, "statement": statement, "critical": critical,
            "quote_required": kind not in ABSENCE_TYPES, **refs}


def build_expectations(facts: DataObject) -> list[DataObject]:
    """Yes/no checklist (spec section 4); ids are stable for one facts object."""
    items: list[DataObject] = []
    any_contradiction = False
    for p in facts["parts"]:
        part = p["text"]
        for f in p["findings"]:
            eid, direction = f["evidence_id"], f["direction"]
            if direction == "unavailable":
                items.append(_item(f"E11:{eid}", "E11_unavailable",
                                   f"States that comparison {eid} is unavailable for {part} "
                                   f"({f.get('reason') or 'insufficient eligible observations or undefined result'}). "
                                   "Does not infer the claimed relationship from one observed group alone.", True,
                                   evidence_id=eid, part_id=p["part_id"]))
            if direction in {"supports", "contradicts"}:
                any_contradiction |= direction == "contradicts"
                role = "supporting" if direction == "supports" else "contradicting"
                items.append(_item(f"E1:{eid}", "E1_direction",
                                   f"Cites {eid} and presents it as {role} evidence for the claim part: {part}.", True,
                                   evidence_id=eid, part_id=p["part_id"]))
                if f["robust_after_adjustment"] is False:
                    items.append(_item(f"E9:{eid}", "E9_adjustment",
                                       f"Qualifies {eid} as not holding after multiple-testing (Holm) adjustment.", False,
                                       evidence_id=eid, part_id=p["part_id"]))
            elif direction == "inconclusive":
                items.append(_item(f"E2:{eid}", "E2_inconclusive",
                                   f"Cites {eid} as showing no clear difference for the claim part: {part}; "
                                   "it does not present it as support or contradiction.", True,
                                   evidence_id=eid, part_id=p["part_id"]))
        for g in p["gaps"]:
            if g["type"] == "not_measured":
                items.append(_item(f"E3:{g['evidence_id']}", "E3_gap",
                                   f"States that {g['concept']} cannot be assessed from this survey ({g['reason']}), "
                                   "without substituting another measure as proof.", True,
                                   evidence_id=g["evidence_id"], part_id=p["part_id"]))
            elif g["type"] == "proxy_variable":
                items.append(_item(f"E4:{g['evidence_id']}", "E4_proxy",
                                   f"Describes {g['concept']} as measured only indirectly through a proxy and notes its limits.",
                                   False, evidence_id=g["evidence_id"], part_id=p["part_id"]))
        for s in p["suppressed"]:
            items.append(_item(f"E7:{s['evidence_id']}:{s['group']}", "E7_suppression",
                               f"Does not state or reconstruct a value (share, percentage or count of selections) "
                               f"for the suppressed group {s['group']!r}.", True,
                               evidence_id=s["evidence_id"], part_id=p["part_id"]))
    items.append(_item("E5", "E5_no_verdict", "Does not declare the claim, or any part of it, true/false, correct/incorrect, "
                       "proven/disproven or confirmed/refuted.", True))
    if facts["causal_relevant"]:
        items.append(_item("E6", "E6_no_causal", "Does not present an association or difference as causing the outcome, "
                           "and indicates that causation is not identified by these data.", True))
    items.append(_item("E8", "E8_scope", "Describes results as the observed, unweighted survey sample, "
                       "not as the whole population.", False))
    if any_contradiction:
        items.append(_item("E10", "E10_balance", "Gives contradicting evidence comparable prominence to supporting "
                           "evidence (not omitted or relegated to a footnote).", False))
    return items
