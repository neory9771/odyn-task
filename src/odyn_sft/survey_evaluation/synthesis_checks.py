"""Deterministic checks D1-D4 for step-2 perspectives: citations, numbers, suppression, required IDs.

Prose wording varies; cited IDs and numbers do not. Number matching is heuristic:
unmatched numbers are reported with their sentence rather than silently ignored.
"""
from __future__ import annotations

import math
import re

from .types import DataObject

VERSION = "synthesis-checks-1.2.2"

ID_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\.r\d+\b")
# Signed decimal or scientific number, optional thousands separators, optional unit.
NUMBER = re.compile(
    r"(?P<cmp>[<>≤≥]=?\s*)?(?P<num>[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-−]?\d+(?:\.\d+)?(?:[eE][-+−]?\d+)?)"
    r"\s*(?P<unit>%|per ?cent\b|percentage points?\b|pp\b|points?\b)?", re.I)
# "-0.6 to 12.4 percentage points": the first number takes the unit written after the second.
RANGE_UNIT = re.compile(r"\s*(?:to|-|–|—|and)\s*[-−]?\d[\d,]*(?:\.\d+)?\s*(%|per ?cent\b|percentage points?\b|pp\b|points?\b)", re.I)
SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
YEAR = re.compile(r"^(19|20)\d{2}$")
CONFIDENCE_LEVEL = re.compile(r"^\s*(?:%\s*)?(?:confidence|interval|CI|credible)", re.I)


def cited_ids(answer: str) -> list[str]:
    return ID_PATTERN.findall(answer)


def _evidence_values(evidence: DataObject) -> tuple[set[int], list[float], list[float], list[float]]:
    """(counts, shares/differences on the 0-1 scale, other reals, p-values) recorded in the pack."""
    counts: set[int] = set()
    shares: list[float] = []
    reals: list[float] = []
    pvalues: list[float] = []
    if isinstance(evidence.get("study", {}).get("n"), int):
        counts.add(evidence["study"]["n"])
    for r in evidence["results"]:
        if r["kind"] == "estimate":
            scores = r.get("operation") == "mean_score"
            for row in r["estimates"]:
                for key in ("n", "numerator"):
                    if isinstance(row.get(key), int):
                        counts.add(row[key])
                values = [v for v in [row.get("value"), *(row.get("ci") or []), row.get("sd")] if v is not None]
                (reals if scores else shares).extend(values)
        elif r["kind"] == "contrast":
            values = [v for v in [r.get("value"), *(r.get("ci") or []), r.get("b") if isinstance(r.get("b"), float) else None,
                                  r.get("crude_odds_ratio"), r.get("cohens_h"), r.get("standardized_mean_difference")]
                      if isinstance(v, (int, float)) and not isinstance(v, bool)]
            scale = r.get("value_scale") in {"odds_ratio"} or r.get("contrast_type") in {"association", "trend"}
            (reals if scale else shares).extend(values)
            if r.get("contrast_type") in {"trend", "association"}:  # Slopes/rho can also be shown as shares.
                shares.extend(values)
            pvalues.extend(v for v in (r.get("p_value"), r.get("p_holm"), r.get("breslow_day_p")) if v is not None)
            if isinstance(r.get("n"), int):
                counts.add(r["n"])
        elif r["kind"] == "segment_sizes":
            counts.update(v for v in r["segments"].values() if isinstance(v, int))
    return counts, shares, reals, pvalues


def _label_numbers(value: object) -> set[str]:
    """Numbers inside evidence text (group names such as "18-24" or "55+", labels, notes).

    Repeating a label is not a numerical claim, so these numbers are exempt from D2.
    """
    found: set[str] = set()
    if isinstance(value, str):
        found.update(m["num"].replace(",", "").lstrip("-−") for m in NUMBER.finditer(value))
    elif isinstance(value, dict):
        for v in value.values():
            found |= _label_numbers(v)
    elif isinstance(value, list):
        for v in value:
            found |= _label_numbers(v)
    return found


def _decimals(text: str) -> int:
    mantissa = text.lower().split("e")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def _close(shown: float, actual: float, decimals: int) -> bool:
    return abs(abs(shown) - abs(actual)) <= 0.5 * 10 ** -decimals + 1e-9


def _number_grounded(match: re.Match[str], values: tuple[set[int], list[float], list[float], list[float]],
                     range_unit: str | None = None) -> bool:
    counts, shares, reals, pvalues = values
    raw = match["num"].replace(",", "").replace("−", "-")
    shown = float(raw)
    unit = (match["unit"] or range_unit or "").lower()
    decimals = _decimals(raw)
    if match["cmp"]:  # "p < 0.001": grounded if some recorded p-value satisfies the bound.
        op = match["cmp"].strip()
        return any((p < shown if op.startswith(("<", "≤")) else p > shown) for p in pvalues + shares + reals)
    if "e" in raw.lower():  # Scientific notation: compare at shown significant figures.
        return any(math.isclose(shown, p, rel_tol=0.5 * 10 ** -decimals + 1e-12) for p in pvalues + reals + shares)
    if unit:  # Percentages and percentage points refer to 0-1 shares/differences.
        return any(_close(shown, 100 * v, decimals) for v in shares)
    if decimals == 0 and abs(shown) in counts:
        return True
    # A unitless number above 1 cannot be a 0-1 share; read it as a percentage, as in
    # ranges where only the last number carries the unit ("12.3 to 19.1 points").
    return any(_close(shown, v, decimals) for v in shares + reals + pvalues) or (
        abs(shown) > 1 and any(_close(shown, 100 * v, decimals) for v in shares))


def _exempt(match: re.Match[str], text: str, claim_numbers: set[str]) -> bool:
    raw = match["num"].replace(",", "")
    following = text[match.end("num"):match.end("num") + 20]
    if YEAR.match(raw) and not match["unit"]:
        return True
    if raw in claim_numbers or raw.lstrip("-−") in claim_numbers:
        return True
    if raw == "95" and CONFIDENCE_LEVEL.match(following):
        return True
    # Small bare integers are usually wording ("top 2", "3 parts"), not results.
    return not match["unit"] and not match["cmp"] and "." not in raw and float(raw) <= 10


def _numbers(text: str) -> list[re.Match[str]]:
    stripped = ID_PATTERN.sub(lambda m: " " * len(m.group()), text)
    return list(NUMBER.finditer(stripped))


def check_answer(answer: str, evidence: DataObject, facts: DataObject) -> DataObject:
    """Run D1-D4; each check reports pass/fail plus the offending details."""
    valid_ids = {r["id"] for r in evidence["results"]}
    cited = cited_ids(answer)
    d1 = sorted(set(cited) - valid_ids)

    values = _evidence_values(evidence)
    # Unsigned: "18-24" in the claim must exempt "18–24" or "24" in the answer.
    claim_numbers = (_label_numbers(facts["claim"]) | _label_numbers(evidence["results"])
                     | _label_numbers(evidence.get("study", {})))
    ungrounded = []
    for sentence in filter(None, (s.strip() for s in SENTENCE.split(answer))):
        stripped = ID_PATTERN.sub(lambda m: " " * len(m.group()), sentence)
        for m in _numbers(sentence):
            ranged = None if m["unit"] else RANGE_UNIT.match(stripped, m.end())
            if not _exempt(m, stripped, claim_numbers) and not _number_grounded(m, values, ranged and ranged.group(1)):
                ungrounded.append({"number": m.group().strip(), "sentence": sentence})

    suppressed_hits = []
    attribution_review = []
    reported_groups = {
        str(row['group']) for record in evidence['results'] if record['kind'] == 'estimate'
        for row in record['estimates'] if row.get('value') is not None
    }
    for part in facts["parts"]:
        for s in part["suppressed"]:
            group = re.compile(rf"(?<!\w){re.escape(str(s['group']))}(?!\w)", re.I)
            for sentence in SENTENCE.split(answer):
                if not group.search(sentence):
                    continue
                # Co-occurrence does not attribute a reported number to a
                # suppressed group. Mixed-group sentences need the critical E7
                # semantic check, which always exists for suppressed groups.
                mixed_groups = [label for label in reported_groups if label != str(s['group'])
                                and re.search(rf'(?<!\w){re.escape(label)}(?!\w)', sentence, re.I)]
                for m in _numbers(sentence):
                    raw = m["num"].replace(",", "")
                    is_base = not m["unit"] and "." not in raw and float(raw) == s["n"]
                    if raw == '95' and CONFIDENCE_LEVEL.match(sentence[m.end('num'):m.end('num') + 20]):
                        continue
                    if (m["unit"] or "." in raw) and not is_base:
                        hit = {"group": s["group"], "number": m.group().strip(), "sentence": sentence.strip()}
                        if mixed_groups:
                            attribution_review.append(hit)
                        else:
                            suppressed_hits.append(hit)

    required = {f["evidence_id"] for p in facts["parts"] for f in p["findings"]
                if f["direction"] in {"supports", "contradicts", "inconclusive"}}
    d4 = sorted(required - set(cited))
    checks = {"D1_citations_valid": {"passed": not d1, "invalid_ids": d1},
              "D2_numbers_grounded": {"passed": not ungrounded, "ungrounded": ungrounded},
              "D3_no_suppressed_values": {"passed": not suppressed_hits, "hits": suppressed_hits,
                                         "requires_E7_semantic_review": attribution_review},
              "D4_required_ids_cited": {"passed": not d4, "missing": d4}}
    return {"version": VERSION, "passed": all(c["passed"] for c in checks.values()), "checks": checks}
