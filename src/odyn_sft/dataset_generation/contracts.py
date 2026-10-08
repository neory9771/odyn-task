"""Authoritative two-level gold contracts; code/prose are executable exemplars."""

from __future__ import annotations

from .types import DataObject

CONTRACT_VERSION = "two-level-contract-1.0.0"


def analysis_contract(case: DataObject) -> DataObject:
    """Describe required analysis meaning without requiring literal code equality."""
    requirements = []
    for binding in case["bindings"]:
        component = next(c for c in case["components"] if c["id"] == binding["subclaim"])
        requirements.append(
            {
                "component_id": component["id"],
                "variable": binding["variable"],
                "operation": binding["operation"],
                "group_variable": case["group_variable"],
                "groups": {label: case["groups"][label] for label in binding["required_groups"]},
                "threshold": binding["threshold"],
                "expect": binding["expect"],
                "estimand": component["estimand"],
                "scope": component["scope"],
                "acceptable_equivalence": "Same measured response and respondent filters; cosmetic names and equivalent API composition may differ. Reversed group differences with reversed direction are equivalent.",
            }
        )
    limitations = [
        {
            "component_id": c["id"],
            "text": c["text"],
            "domains": c["D"],
            "availability": c["A"],
            "reason": c["required_limitation"],
        }
        for c in case["components"]
        if "required_limitation" in c
    ]
    return {
        "version": CONTRACT_VERSION,
        "level": "analysis",
        "claim_components": case["components"],
        "relations": case["relations"],
        "required_analyses": requirements,
        "required_gaps": limitations,
        "proxy_policy": "Use a proxy only if justified; label the difference from the requested construct. A related measure is not proof of an unavailable outcome.",
        "reference_program": case["program"],
        "reference_evidence": case["evidence"],
        "criteria": [
            "coverage",
            "measure_population",
            "estimand_direction",
            "gaps_proxy",
            "execution",
        ],
        "code_exact_match_required": False,
    }


def synthesis_contract(case: DataObject) -> DataObject:
    """Create the findings/limits contract after API execution has assigned E."""
    contrast_by_component = {
        r["subclaim"]: r for r in case["evidence"]["results"] if r["kind"] == "contrast"
    }
    estimate_by_id = {r["id"]: r for r in case["evidence"]["results"] if r["kind"] == "estimate"}
    findings = []
    for component in case["components"]:
        contrast = contrast_by_component.get(component["id"])
        item = {
            "component_id": component["id"],
            "text": component["text"],
            "E": component["E"],
            "D": component["D"],
            "A": component["A"],
            "S": component["S"],
            "diagnostic_flags": component["diagnostic_flags"],
        }
        if contrast:
            estimate = estimate_by_id[contrast["estimate_id"]]
            binding = next(b for b in case["bindings"] if b["subclaim"] == component["id"])
            item.update(
                contrast=contrast,
                estimate={
                    **estimate,
                    "estimates": [
                        x for x in estimate["estimates"] if x["group"] in binding["required_groups"]
                    ],
                },
            )
        else:
            item["required_limitation"] = component["required_limitation"]
        findings.append(item)
    return {
        "version": CONTRACT_VERSION,
        "level": "synthesis",
        "required_findings": findings,
        "relations": case["relations"],
        "scope": "Observed analysis bases of the supplied study; population, causal and behavioural extensions require their own evidence.",
        "prohibited": [
            "Invented statistics or studies",
            "Suppressed numerators reconstructed in prose",
            "No clear difference described as equivalence",
            "Unadjusted CI label described as Holm-confirmed significance",
            "Unjustified causal, population, behavioural, transport or temporal conclusions",
            "Whole-claim true/false verdict",
        ],
        "criteria": ["coverage", "grounding", "inference", "scope_stress", "perspective"],
        "reference_prose_exact_match_required": False,
        "candidate_conditioning": "For end-to-end synthesis, ground numbers in candidate execution, never fill missing candidate findings from private gold. Acknowledge analysis gaps; Level 1 scores their cause.",
    }
