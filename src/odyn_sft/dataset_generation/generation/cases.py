"""Render a selected candidate as a case, verify its reference evidence and write the reference perspective."""

from __future__ import annotations

import numpy as np

from ...common.storage import digest
from ..study import measurement_phrase
from ..types import Candidate, DataObject
from .candidates import annotate_candidate, keys
from .evidence import wilson_oracle
from .scenarios import SCENARIOS, STYLES


def materialize(candidate: Candidate, split: str, seed: int, metadata: DataObject) -> DataObject:
    """Render a selected claim and its proposition/relation inference contract."""
    candidate = annotate_candidate(candidate)
    scenario = candidate["scenario"]
    components, bindings, code = [], [], []
    analyses = candidate["analyses"]
    if analyses:
        first = analyses[0]["row"]
        code.append(f"g = group(var({first['group_variable']!r}), {first['groups']!r})")
    for i, analysis in enumerate(analyses, 1):
        row, direction = analysis["row"], analysis["direction"]
        labels = list(row["groups"])
        a, b = labels[0], labels[1] if len(labels) > 1 else None
        expect = ">" if direction > 0 else "<"
        phrase = measurement_phrase(row)
        if analysis["threshold"]:
            text = (
                f"{'More' if direction > 0 else 'Fewer'} than half of respondents in {a} {phrase}."
            )
        else:
            text = f"Respondents in {a} {phrase} {'more often' if direction > 0 else 'less often'} than respondents in {b}."
        c = row["card"]
        call = (
            f"top_box(var({c['id']!r}), k=2, by=g)"
            if c["answer_type"] == "ordered_scale"
            else f"proportion(var({c['id']!r}), by=g)"
            if c["answer_type"] == "select_all"
            else f"proportion(var({c['id']!r}), value={row['value']!r}, by=g)"
        )
        sid = f"c{i}"
        code += [
            f"with subclaim({sid!r}, {text!r}):",
            f"    e{i} = {call}",
            f"    compare(e{i}, {a!r}, {0.5 if analysis['threshold'] else repr(b)}, expect={expect!r})",
        ]
        components.append(
            {
                "id": sid,
                "text": text,
                "D": ["statistical_inference"],
                "A": "available",
                "E": analysis["E"],
                "R": c["answer_type"],
                "S": analysis["S"],
                "diagnostic_flags": analysis.get("diagnostic_flags", []),
                "analytical_core": analysis["core"],
                "direction": direction,
                "direction_commitment": analysis["direction_commitment"],
                "reference_counts": analysis["counts"],
                "reference_interval": analysis["ci"],
                "boundary_audit": analysis["stability"],
                "estimand": "observed_group_share_minus_half"
                if analysis["threshold"]
                else "observed_group_share_difference",
                "scope": "Observed unweighted 2022 analysis base, not population prevalence.",
            }
        )
        bindings.append(
            {
                "subclaim": sid,
                "variable": c["id"],
                "operation": row["measure"]["operation"],
                "required_groups": [a] if analysis["threshold"] else [a, b],
                "threshold": 0.5 if analysis["threshold"] else None,
                "expect": expect,
            }
        )
    if scenario in SCENARIOS:
        # A concerns all evidence needed for the inference. An observed
        # association is only partial evidence for the causal explanation.
        domain, text, reason, availability = SCENARIOS[scenario]
        if not analyses:
            context = candidate["context"]["row"]
            target = candidate["target"]
            if scenario == "historical":
                text = f"Among {target}, the share who {measurement_phrase(context)} increased between 2020 and 2022."
            else:
                text = f"Verified transaction records establish that respondents in {target} made the purchase described by the recorded item {context['card']['label']!r}"
                if context["value"] is not None:
                    text += f" with response category {context['value']!r}"
                text += "."
        code += [
            f"with subclaim('limit', {text!r}):",
            f"    not_measured({scenario!r}, {reason!r})",
        ]
        components.append(
            {
                "id": "limit",
                "text": text,
                "D": [domain],
                "A": availability,
                "E": "not_applicable" if availability != "unavailable" else "unavailable",
                "R": None,
                "S": [],
                "diagnostic_flags": [],
                "required_limitation": reason,
                "identification_supported": False,
                "api_note_convention": "not_measured records lack of justified inference; it does not always mean raw endpoints are absent.",
            }
        )
    code.append(
        "caveat('Report the observed sample and relevant limits; do not give a whole-claim true/false verdict. Unadjusted interval labels do not establish Holm-adjusted significance.')"
    )
    family = candidate["family"]
    template = int(digest({"family": family, "seed": seed, "purpose": "wording"})[:8], 16) % len(
        STYLES
    )
    composition = candidate["dimensions"]["C"][0]
    # Mixed is derived from component E values. It is not a primary case class.
    case = {
        "case_id": "rev2_" + family[:20],
        "family_id": family,
        "split": split,
        "scenario": scenario,
        "C": composition,
        "components": components,
        "relations": [
            {
                "from": [c["id"] for c in components[:-1]],
                "to": "limit",
                "domain": SCENARIOS[scenario][0],
                "justified": False,
            }
        ]
        if composition == "linked"
        else [],
        "D": sorted({d for c in components for d in c["D"]}),
        "A": candidate["dimensions"]["A"][0],
        "E": [c["E"] for c in components],
        "R": sorted({c["R"] for c in components if c["R"] is not None}),
        "S": sorted({s for c in components for s in c["S"]}),
        "diagnostic_flags": sorted({f for c in components for f in c["diagnostic_flags"]}),
        "claim": STYLES[template] + " ".join(c["text"] for c in components),
        "template_family": template,
        "program": "\n".join(code) + "\n",
        "bindings": bindings,
        "component_keys": keys(candidate),
        "gap_type": scenario if scenario in SCENARIOS else None,
        "scenario_keys": candidate["scenario_keys"],
        "context_keys": candidate.get("context_keys", []),
        "outcome_blocks": sorted({a["row"]["card"]["block_id"] for a in analyses}),
        "draft_status": "pending_expert_and_language_review",
        "study_id": "quant_us_2022",
    }
    if analyses:
        case.update(
            group_variable=analyses[0]["row"]["group_variable"], groups=analyses[0]["row"]["groups"]
        )
    if candidate.get("context"):
        context = candidate["context"]["row"]
        case["context_measure"] = {
            "variable": context["card"]["id"],
            "label": context["card"]["label"],
            "operation": context["measure"]["operation"],
            "group_variable": context["group_variable"],
            "groups": context["groups"],
            "source_period": "2022",
            "requested_prior_period": "2020" if scenario == "historical" else None,
        }
    measured_e = {a["E"] for a in analyses}
    case["derived_tags"] = (["mixed"] if {"support", "contradict"} <= measured_e else []) + (
        ["partially_answerable"]
        if analyses and scenario in SCENARIOS
        else ["outside_survey"]
        if not analyses
        else []
    )
    return case


def validate_reference(case: DataObject, evidence: DataObject) -> int:
    """Compare executed API evidence with independent counts/intervals and E."""
    mapping = {
        "consistent": "support",
        "inconsistent": "contradict",
        "no_clear_difference": "inconclusive",
        "unavailable": "unavailable",
    }
    checked = 0
    for component, binding in zip(case["components"], case["bindings"]):
        estimates = [
            r
            for r in evidence["results"]
            if r["kind"] == "estimate" and r["subclaim"] == component["id"]
        ]
        contrasts = [
            r
            for r in evidence["results"]
            if r["kind"] == "contrast" and r["subclaim"] == component["id"]
        ]
        if len(estimates) != 1 or len(contrasts) != 1:
            raise ValueError("Reference component missing: " + case["case_id"])
        rows = estimates[0]["estimates"]
        for observed, (n, k) in zip(rows, component["reference_counts"]):
            if n < 10:
                if observed["value"] is not None:
                    raise ValueError("Suppression mismatch")
            else:
                if observed["n"] != n or observed["numerator"] != k:
                    raise ValueError("Independent count oracle mismatch")
                if not np.allclose(observed["ci"], wilson_oracle(k, n), atol=1e-12, rtol=0):
                    raise ValueError("Independent proportion interval mismatch")
            checked += 1
        r = contrasts[0]
        if mapping.get(r["label"]) != component["E"]:
            raise ValueError(f"Evidence direction mismatch: {r['label']} vs {component['E']}")
        if component["reference_interval"] is not None and not np.allclose(
            r["ci"], component["reference_interval"], atol=1e-12, rtol=0
        ):
            raise ValueError("Independent contrast interval mismatch")
    return checked


def reference_perspective(case: DataObject) -> str:
    """Render actual evidence, including suppressed bases, without a truth verdict."""
    paragraphs = []
    for r in case["evidence"]["results"]:
        if r["kind"] == "estimate":
            details = []
            for x in r["estimates"]:
                if x["value"] is None:
                    details.append(
                        f"{x['group']}: observed base n={x['n']}; share and numerator suppressed"
                    )
                else:
                    details.append(
                        f"{x['group']}: {x['numerator']}/{x['n']} ({100 * x['value']:.1f}%; 95% CI {100 * x['ci'][0]:.1f}–{100 * x['ci'][1]:.1f}%)"
                    )
            paragraphs.append(f"[{r['id']}] {r['variable_label']}: " + "; ".join(details) + ".")
        elif r["kind"] == "contrast":
            if r["ci"] is None:
                paragraphs.append(
                    f"[{r['id']}] This comparison is unavailable because a required base is suppressed or empty."
                )
            else:
                state = {
                    "consistent": "supports the specified statistical component",
                    "inconsistent": "runs against the specified statistical component",
                    "no_clear_difference": "does not resolve the specified direction",
                }[r["label"]]
                paragraphs.append(
                    f"[{r['id']}] The unadjusted comparison {state}: difference {100 * r['value']:.1f} percentage points, 95% CI {100 * r['ci'][0]:.1f}–{100 * r['ci'][1]:.1f}; two-sided p={r['p_value']:.6g}, Holm-adjusted p={r['p_holm']:.6g}. The interval classification is not a multiplicity-adjusted significance claim."
                )
        elif r.get("label") == "not_measured":
            paragraphs.append(f"[{r['id']}] {r['reason']}")
    if "boundary_sensitive" in case["diagnostic_flags"]:
        paragraphs.append(
            "The component classification changes under the documented one-response perturbation; interpret its direction as boundary-sensitive."
        )
    paragraphs.append(
        "These findings describe the observed unweighted 2022 analysis bases. Relevant routing, population and identification limits remain; this is an evidence perspective for the user's judgement."
    )
    return "\n\n".join(paragraphs)
