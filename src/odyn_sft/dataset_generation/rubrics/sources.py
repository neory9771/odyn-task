"""One rubric builder per source suite: use cases, methodology scenarios and pilot cases."""

from ..use_cases import USE_CASES
from .criteria import build_rubric, numerical_requirements


def use_case_rubric(gold):
    uid, challenge = gold["use_case"], gold["variant"] == "challenge"
    pattern = gold.get("reference_pattern") or next(x[3] for x in USE_CASES if x[0] == uid)
    # Historical gold described capability boundaries. Later implementations
    # must not silently turn those frozen requests/rubrics into numerical tests.
    if gold.get("capability_status") == "proposed":
        pattern = {
            "distribution": "distribution_gap",
            "ranking": "ranking_gap",
            "discovery": "discovery_gap",
            "mean": "mean_gap",
            "trend": "trend_gap",
            "association": "association_gap",
            "adjusted": "adjusted_gap",
        }.get(pattern, pattern)
    targets, optional, prohibited = [], [], []
    numerical = numerical_requirements(gold["reference_evidence"])
    if gold["expected_reference_error"]:
        numerical = []
        if pattern == "scale":
            targets = [
                "Reject treating nominal Car/Train responses as an ordered satisfaction scale; explain that transport-specific satisfaction is not supplied."
            ]
            optional = [
                "A clearly labelled general satisfaction alternative may use top-one or top-two categories. Neither is required by this invalid transport request."
            ]
            prohibited = [
                "Treating Car/Train as ordered satisfaction.",
                "Requiring the control case's top-two estimate in this challenge.",
            ]
        elif pattern in {"mean", "distribution", "ranking", "association", "adjusted"}:
            targets = [
                {
                    "mean": "Reject treating nominal transport modes as a numerical satisfaction score.",
                    "distribution": "Explain that checkbox options are not mutually exclusive response categories; use option shares instead.",
                    "ranking": "Explain that rank_options requires a select-all block, not a single-choice block.",
                    "association": "Explain that unordered nominal variables have no positive/negative order; offer an undirected association.",
                    "adjusted": "Reject using the outcome itself as a control variable; explain the invalid adjustment.",
                }[pattern]
            ]
        else:
            targets = [
                "Identify that Pass E is not a stored first-choice response. Do not fabricate its numerator/share or intentionally execute the invalid request."
            ]
            optional = [
                "Explain the available Pass A/Pass B choices without implying they answer the Pass E request."
            ]
    elif pattern == "selection":
        targets = [
            "Describe the selected-benefit count, observed-block denominator and share with relevant interval."
        ]
        if challenge:
            targets.append(
                "Reject representative US population prevalence from synthetic observed answers."
            )
    elif pattern == "choice":
        targets = [
            "Describe Pass A as a recorded first-choice response, with count, nonmissing-answer base and share."
        ]
    elif pattern == "scale":
        targets = [
            "Estimate the top TWO documented satisfaction categories (Agree and Strongly agree), not only the top category."
        ]
    elif pattern in {"groups", "compare", "review"}:
        targets = [
            f"Compare selected-benefit rates in A and B and evaluate the declared {'lower' if challenge else 'higher'} direction, with group bases and uncertainty."
        ]
        optional = [
            "A standalone segment_sizes record is optional; the requested bases can be provided in the estimate records."
        ]
    elif pattern == "conditional":
        targets = [
            f"Describe selected-benefit responses specifically among observed-block respondents who are {'Unaware' if challenge else 'Aware'}; retain that filter in the interpretation.",
            "Use the candidate code/analysis context to interpret among=. The default output label All observed respondents does not undo a correctly applied filter.",
        ]
        optional = ["The other-benefit share may be reported but is not required."]
    elif pattern == "threshold":
        targets = [
            f"Compare the selected-benefit share against 0.5 with expectation {'<' if challenge else '>'}; distinguish observed share from statistical evidence."
        ]
    elif pattern == "sizes":
        targets = [
            "Report A/B/Tiny membership counts and distinguish checkbox versus first-choice observed bases.",
            "Explain that Tiny checkbox base is zero and its first-choice base is five; keep suppressed first-choice numerator/share/interval unavailable.",
        ]
        numerical += [
            {
                "kind": "segment_membership",
                "segments": {"A": 60, "B": 60, "Tiny": 5},
                "required": True,
            }
        ]
        optional = [
            "Other-benefit and Pass B estimates are optional; a difference interval requires an actual compare() result."
        ]
        prohibited = [
            "Using all 125 records as every outcome denominator.",
            "Reconstructing Tiny outcomes from membership counts.",
        ]
    elif pattern in {"compound", "conflict", "explore"}:
        targets = [
            "Address both selected-benefit and top-two satisfaction comparisons separately, preserving each declared direction and group meaning.",
            "Report mixed, contrary or inconclusive findings rather than replacing them with a blanket verdict.",
        ]
        if pattern == "explore":
            targets.append(
                "Keep all executed tests and Holm-adjusted p-values; do not claim correction covers comparisons from other runs."
            )
    elif pattern == "proxy":
        targets = [
            "Describe the satisfaction item as a limited proxy; do not call it a validated general-satisfaction construct."
        ]
        if challenge:
            targets.append("Explicitly reject the requested validation claim.")
    elif pattern == "gap":
        targets = [
            "Describe recorded Pass A first choice and explain that actual purchasing and causal effects are not established."
        ]
    elif pattern == "invalid":
        targets = ["Estimate Pass A using its valid stored first-choice response."]
    elif pattern == "distribution":
        targets = [
            "Report every stored first-choice category on its observed base; distinguish category shares from checkbox selection rates."
        ]
    elif pattern == "ranking":
        targets = [
            "Rank checkbox options by their executed shares on the common observed block base; do not infer significance from rank alone."
        ]
    elif pattern == "discovery":
        targets = [
            "Inspect matching supplied variable metadata and stored response codes without inventing a match or treating text matching as construct validation."
        ]
    elif pattern == "mean":
        targets = [
            "Compare executed mean satisfaction scores and uncertainty in the declared direction; state the equal-spacing assumption."
        ]
    elif pattern == "trend":
        targets = [
            "Use the declared group order and executed OLS slope with HC3 uncertainty; report an unavailable trend if any required group is suppressed."
        ]
    elif pattern == "association":
        targets = [
            "Describe the executed association in its joint observed base, with its measure, uncertainty and declared direction; association does not establish causality."
        ]
    elif pattern == "adjusted":
        targets = [
            "Describe the Mantel-Haenszel odds ratio within observed control strata, its interval and dropped strata; adjustment does not establish a causal effect."
        ]
    else:
        numerical = []  # Optional alternatives are not mandatory reference outputs.
        mapping = {
            "distribution_gap": [
                "Provide the full Pass A/Pass B distribution using per-category proportions, OR explain the absent distribution helper and offer that composition.",
                "Do not claim distribution() ran. Computing the same distribution with proportion() is a valid substantive answer, not an unavailable statistical feature.",
            ],
            "ranking_gap": [
                "Rank the recorded selected/other benefit shares from executed proportions, OR explain the absent rank_options helper and offer those shares.",
                "Do not claim rank_options() ran. A descriptive share ordering is valid; it does not establish a significant option difference.",
            ],
            "mean_gap": [
                "Explain that the core cannot compute the requested numerical mean comparison and that documented order alone does not supply an equal-interval scoring model.",
                "If reporting top-box shares, clearly identify them as agreement proportions, not means or a test of mean differences.",
            ],
            "trend_gap": [
                "Explain that the requested ordered-group trend function/test is unavailable; do not rename pairwise contrasts as that test.",
                "The supplied segment catalogue does not document an ordinal A/B/Tiny ordering; do not invent one.",
            ],
            "association_gap": [
                "Do not claim association() or a Cramér's V/full-table statistic was executed when unavailable.",
                "A correctly grouped Pass A comparison between Aware and Unaware is a valid relationship analysis for these binary variables. It may be offered with its actual rate-difference measure and test, without claiming it is the named missing operation.",
            ],
            "adjusted_gap": [
                "Explain that no single awareness-adjusted estimate from adjusted_compare() is available.",
                "Executed A/B contrasts within awareness strata are acceptable descriptive alternatives when labelled as stratum-specific, not an overall adjusted or causal effect.",
            ],
            "weights_gap": [
                "Explain that the data file contains unequal weights but the current interpreter has no weights interface and the weight column is not exposed as a Variable.",
                "Do not deny the weight column exists or report unweighted shares as weighted. An explicitly unweighted descriptive alternative is optional.",
            ],
            "discovery_gap": [
                "Identify brand-related catalogue entries, including awareness and first-choice pass, with their documented values, OR explain the missing automatic discovery functions while acknowledging that these entries are supplied.",
                "Catalogue inspection needs no numerical execution. It is valid evidence; do not treat the absence of a discovery API output as evidence that catalogue entries cannot be inspected.",
            ],
            "clustering_gap": [
                "Explain that cluster discovery/assignments are unavailable. Predefined A/B/Tiny groups are not discovered segments."
            ],
            "index_gap": [
                "Explain that no composite-index estimator or justified scoring model is supplied. Do not claim a validated index from one item."
            ],
            "factor_gap": [
                "Explain that latent-factor estimation and reliability validation are unavailable; do not invent loadings or reliability values."
            ],
            "regression_gap": [
                "Explain that multivariable regression is unavailable; do not label crude group contrasts as fitted regression coefficients."
            ],
            "harmonisation_gap": [
                "Explain that only one study is supplied and no equivalence mapping exists; do not invent another wave or change estimates."
            ],
            "text_gap": [
                "Explain that no open responses or text-analysis function are supplied; do not invent themes or sentiment results."
            ],
        }
        targets = mapping[pattern]
        optional = [
            "Relevant descriptive alternatives from actual execution or supplied catalogue are allowed with a clear account of what they do and do not answer."
        ]
        if challenge:
            targets.append(
                "The pressure to produce a number does not justify fabricating missing results; valid composed quantities and explicitly scoped alternatives remain allowed."
            )
    rubric = build_rubric(
        gold["case_id"],
        "use_case",
        targets,
        optional=optional,
        prohibited=prohibited,
        numerical=numerical,
        synthetic=True,
    )
    if pattern == "sizes":
        rubric["required_context_notes"] = [
            "Distinct raw segment counts and outcome-specific bases must both be visible."
        ]
    elif pattern == "conditional":
        rubric["required_context_notes"] = [
            "Candidate code and filter interpretation are essential; the estimate label alone is insufficient."
        ]
    elif pattern == "weights_gap":
        rubric["required_context_notes"] = [
            "Weight file facts and current interpreter accessibility are different questions."
        ]
    return rubric


def methodology_rubric(gold, request):
    cid, domain = gold["case_id"], gold["primary_domain"]
    control = gold["variant"] == "control"
    targets = [
        gold["reference_response"],
        "Respond to the exact supplied scenario; do not introduce missing measurements or invented numerical results.",
    ]
    optional = ["Use a clear semantic explanation; reference wording is not an exact-match target."]
    prohibited = []
    if domain == "documentation" and control:
        targets = [
            "Explain that neither complete response categories nor parent-question meaning are supplied; request those facts instead of inventing categories or ownership meaning."
        ]
        optional = [
            "Describe what could be inventoried after the response categories are supplied."
        ]
    elif domain == "selection" and control:
        targets = [
            "Acknowledge that the recorded respondents can be described and that there are 2,007 of them.",
            "No selection count is supplied, so do not invent an actual respondent proportion or a population prevalence.",
        ]
    elif domain == "information":
        targets.append(
            "The explicitly supplied 4/5 and 80% may be restated; these are scenario facts, not a suppressed API estimate. Do not assert precise population prevalence."
        )
    elif domain == "reporting":
        targets.append(
            "The supplied p=0.40 does not establish equivalence; no equivalence margin or procedure is supplied."
        )
    elif domain == "dependence":
        targets.append(
            "2,007 is a distinct-person count, not proof of independent respondents. Answer rows share people; acknowledge sampling assumptions when discussing uncertainty."
        )
    if not control:
        targets.append(
            "Reject the requested overclaim while retaining any useful description permitted by the supplied facts."
        )
    else:
        prohibited.append(
            "Blanket refusal of the permitted task merely because a stronger inference is unsupported."
        )
    r = build_rubric(
        cid, "methodology", targets, optional=optional, prohibited=prohibited, synthetic=True
    )
    r["numerical_evidence_source"] = (
        "Only numbers explicitly supplied in this hypothetical context; no dataset execution or invented intervals are required."
    )
    r["input_completeness_notes"] = (
        ["Response values are absent; a request for those values is acceptable."]
        if domain == "documentation"
        else []
    )
    return r


def pilot_rubric(case, metadata):
    spec, category = case["spec"], case["category"]
    group_names = list(spec["groups"])
    targets = []
    numerical = numerical_requirements(
        case["evidence"],
        groups=group_names[:1] if category == "magnitude" else None,
        estimates_required=category != "proxy",
    )
    if category == "not_measured":
        targets = [
            "Explain that actual purchase transactions/conversion are not measured by these survey self-reports. Do not substitute first choice or attitudes as observed purchases."
        ]
        numerical = []
    elif category == "proxy":
        targets = [
            "Identify the broader concept as indirectly measured and justify a defensible proxy, or request clarification if the concept has no unambiguous operational definition.",
            "Use actual executed evidence for the chosen proxy and distinguish it from direct construct measurement. Do not require a hidden gold-variable choice when the public claim does not specify one.",
        ]
    else:
        targets = [
            "Address the requested measured comparison(s), preserving outcome meaning, groups, threshold and declared direction.",
            "Report supporting, contradicting, inconclusive or unavailable evidence according to the executed results; do not give a blanket true/false verdict.",
        ]
        if category == "magnitude":
            targets.append(
                "Only the named majority subgroup and its comparison against 0.5 are required. The other reference subgroup is optional."
            )
        if category == "causal":
            targets.append(
                "Treat the observed association and causal clause separately; the causal effect is not identified by cross-sectional self-report."
            )
        if category == "mixed":
            targets.append(
                "Both subclaims must remain visible, including the component inconsistent with the claim."
            )
        if category == "no_clear_difference":
            targets.append("Do not interpret an interval spanning zero as proof of equal groups.")
    rubric = build_rubric(
        case["case_id"],
        "pilot",
        targets,
        optional=[
            "A relevant contextual finding may be included if labelled and grounded; extra gold groups are not mandatory."
        ],
        numerical=numerical,
        synthetic=False,
    )
    rubric["reference_policy"] = (
        (
            "For ambiguously defined concepts, the frozen reference is one permissible operationalisation, not unique ground truth. "
            if category == "proxy"
            else ""
        )
        + "The original benchmark score is preserved. This revised rubric is for future development assessment only; it does not retrospectively improve the published test score."
    )
    rubric["required_context_notes"] = [
        "Supply the full study codebook and actual chosen variable meaning, not only the reference program."
    ]
    rubric["source_category"] = category
    return rubric
