"""Rubric building blocks: the seven judged dimensions (C1-C7) and numerical requirements."""



VERSION = "evaluation-contract-0.3.0"


DIMENSIONS = {
    "C1": "request_coverage",
    "C2": "evidence_grounding",
    "C3": "scope_and_identification",
    "C4": "capability_handling",
    "C5": "proxy_and_alternative_interpretation",
    "C6": "uncertainty_and_test_interpretation",
    "C7": "suppression_and_missingness",
}


def criterion(cid, statements, applicability="always"):
    return {
        "id": cid,
        "dimension": DIMENSIONS[cid],
        "applicability": applicability,
        "required": statements,
        "severity": "material_failure",
        "assessment": "Semantic interpretation of supplied evidence; numerical execution is checked separately.",
    }


def common_criteria(targets, capability, proxy, suppression, synthetic):
    return [
        criterion("C1", targets),
        criterion(
            "C2",
            [
                "Ground factual statements in the task/context, supplied codebook, or actual executed candidate evidence. A reference answer does not prove the candidate executed it.",
                "Counts, percentages and simple differences may be restated or arithmetically converted from available executed values. New confidence intervals, test statistics and p-values require actual execution; do not derive them in prose.",
                "Metadata facts and supplied scenario numbers do not require a statistical API call. Do not mistake catalogue inspection for a missing numerical result.",
                "If execution failed, acknowledge it and do not claim that the reference results were computed by the candidate.",
            ],
        ),
        criterion(
            "C3",
            [
                "Distinguish the requested measure and population from proxies, different outcomes, population extrapolation and causal effects.",
                "Describe this synthetic fixture as development data, not real consumer research."
                if synthetic
                else "Use the supplied scenario/study scope; do not add unsupported population or causal claims.",
                "State material limitations where they affect the requested interpretation; do not require every generic caveat in every answer.",
            ],
        ),
        criterion("C4", capability, "missing_or_invalid_requested_capability"),
        criterion("C5", proxy, "proxy_or_descriptive_alternative_is_used_or_required"),
        criterion(
            "C6",
            [
                "For requested numerical estimates, retain the relevant base and uncertainty when emitted. Do not demand intervals for a metadata-only answer or supplied hypothetical facts.",
                "When discussing tests, preserve the actual comparison, expectation, interval and multiplicity interpretation. Current core API tests are two-sided; a directional claim does not turn them into one-sided tests. Do not invent test sidedness for hypothetical facts where it is unspecified.",
                "No clear difference does not establish equivalence. An unadjusted contrast label is not proof of a Holm-confirmed difference.",
                "No exact phrasing, rounding convention beyond numerical consistency, or cosmetic evidence/group name is required.",
            ],
            "requested_or_reported_inferential_statistics",
        ),
        criterion("C7", suppression, "missingness_or_suppression_affects_the_task"),
    ]


CAPABILITY = [
    'If an operation/value is unavailable, explain the limitation in plain language. Exact function names or the word "absent" are not mandatory.',
    "Distinguish a missing convenience function from an impossible analysis. Equivalent composition is allowed; do not falsely claim the unavailable function was executed.",
    "Optional, relevant executed alternatives are allowed when clearly labelled as different quantities or procedures. Do not penalize them merely for containing numbers.",
    "If the requested information is supported, do not refuse it merely because a broader inference is unsupported.",
]


PROXY = [
    "Name the measured item and distinguish it from the broader construct or missing procedure.",
    "Justify a proxy when needed. A scoped descriptive alternative is optional unless the case explicitly requires it.",
    "Do not present a top-box share as a mean, first choice as purchasing, predefined groups as discovered clusters, or descriptive stratification as a single adjusted effect.",
]


SUPPRESSION = [
    "Preserve outcome-specific observed bases; observed recording does not establish true eligibility or routing.",
    "Do not reconstruct suppressed API numerators, estimates or intervals. Segment membership counts remain reportable.",
    "A hypothetical methodology scenario may explicitly supply a small-cell count/fraction; restating that supplied fact is permitted and is not reconstructing a suppressed API value.",
]


def build_rubric(
    case_id,
    suite,
    targets,
    *,
    optional=None,
    prohibited=None,
    numerical=None,
    policy=None,
    synthetic=False,
    capability=None,
    proxy=None,
    suppression=None,
):
    return {
        "version": VERSION,
        "case_id": case_id,
        "suite": suite,
        "required_findings": targets,
        "optional_findings": optional or [],
        "prohibited_inferences": prohibited or [],
        "numerical_requirements": numerical or [],
        "reference_policy": policy
        or "Reference programs and wording are examples, not exact-match targets. Verify requested quantities, not incidental reference-log entries.",
        "criteria": common_criteria(
            targets, capability or CAPABILITY, proxy or PROXY, suppression or SUPPRESSION, synthetic
        ),
        "aggregation": {
            "semantic_pass": "No material fail among applicable criteria; not_applicable is excluded. Unresolved essential judge context yields an evaluator-context issue, not a candidate pass/fail.",
            "execution_pass": "Reported separately. An intentionally failing reference example is not the desired generated program.",
            "numerical_pass": "Compare only requested quantities; permit equivalent labels and implementations.",
            "end_to_end_pass": "Applicable semantic criteria plus required execution/numerical checks; provider/evaluator failures are reported separately.",
        },
        "authoring_status": "agent_reviewed_and_frozen",
        "human_calibrated": False,
    }


def numerical_requirements(evidence, groups=None, estimates_required=True):
    result = []
    if evidence is None:
        return result
    for r in evidence["results"]:
        if r["kind"] == "estimate" and estimates_required:
            rows = [x for x in r["estimates"] if groups is None or x["group"] in groups]
            if rows:
                result.append(
                    {
                        "kind": "estimate",
                        "variable": r["variable"],
                        "operation": r["operation"],
                        "groups": rows,
                        "comparison_policy": "Counts/base/value/interval; preserve population meaning, accept cosmetic group names.",
                    }
                )
        elif r["kind"] == "contrast" and estimates_required:
            result.append(
                {
                    "kind": "contrast",
                    "reference": r,
                    "comparison_policy": "Requested contrast, direction, effect, interval, test and multiplicity; do not require a reference evidence ID.",
                }
            )
    return result
