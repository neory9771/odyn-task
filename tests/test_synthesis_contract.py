"""Portable synthesis facts and deterministic grounding checks."""

import json

import pytest

from odyn_sft.survey_evaluation.synthesis_checks import check_answer, cited_ids
from odyn_sft.survey_evaluation.synthesis_facts import build_expectations, build_facts

CLAIM = (
    "Women know the brand more often than men. Most people in Florida know the brand. "
    "Low awareness among young people is because our media misses them."
)


def evidence(**contrast_overrides):
    """API-0.3.0-shaped pack: no label_holm, so it must be recomputed from p_holm."""
    contrast = {
        "id": "c1.r3",
        "subclaim": "c1",
        "kind": "contrast",
        "estimate_id": "c1.r2",
        "a": "Women",
        "b": "Men",
        "expect": ">",
        "contrast_type": "difference",
        "value": 0.1573,
        "ci": [0.1229, 0.1911],
        "p_value": 0.03,
        "p_holm": 0.06,
        "label": "consistent",
        "test": "two_proportion_z",
        "flags": ["unweighted"],
    }
    contrast.update(contrast_overrides)
    return {
        "api_version": "0.3.0-core",
        "study": {"label": "Awareness study", "n": 3046},
        "results": [
            {
                "id": "c1.r1",
                "subclaim": "c1",
                "kind": "subclaim",
                "text": "Recorded evidence for component 1",
            },
            {
                "id": "c1.r2",
                "subclaim": "c1",
                "kind": "estimate",
                "variable": 20,
                "variable_label": "Brand awareness",
                "operation": "selection",
                "estimates": [
                    {
                        "group": "Women",
                        "n": 1598,
                        "numerator": 748,
                        "value": 0.46808,
                        "ci": [0.44373, 0.49260],
                        "flags": [],
                    },
                    {
                        "group": "Men",
                        "n": 1448,
                        "numerator": 450,
                        "value": 0.31077,
                        "ci": [0.28746, 0.33509],
                        "flags": [],
                    },
                    {
                        "group": "Nonbinary",
                        "n": 7,
                        "numerator": None,
                        "value": None,
                        "ci": None,
                        "flags": ["small_n", "suppressed"],
                    },
                ],
            },
            contrast,
            {
                "id": "c2.r1",
                "subclaim": "c2",
                "kind": "subclaim",
                "text": "Recorded evidence for component 2",
            },
            {
                "id": "c2.r2",
                "subclaim": "c2",
                "kind": "estimate",
                "variable": 21,
                "variable_label": "Brand awareness",
                "operation": "selection",
                "estimates": [
                    {
                        "group": "Florida",
                        "n": 400,
                        "numerator": 210,
                        "value": 0.525,
                        "ci": [0.476, 0.574],
                        "flags": [],
                    }
                ],
            },
            {
                "id": "c2.r3",
                "subclaim": "c2",
                "kind": "contrast",
                "estimate_id": "c2.r2",
                "a": "Florida",
                "b": 0.5,
                "expect": ">",
                "contrast_type": "threshold",
                "value": 0.025,
                "ci": [-0.024, 0.074],
                "p_value": 0.32,
                "p_holm": 0.32,
                "label": "no_clear_difference",
                "flags": ["unweighted"],
            },
            {
                "id": "gap.r1",
                "subclaim": "gap",
                "kind": "subclaim",
                "text": "causal effect",
            },
            {
                "id": "gap.r2",
                "subclaim": "gap",
                "kind": "note",
                "concept": "causal effect",
                "reason": "No identifying causal design is supplied.",
                "label": "not_measured",
            },
            {
                "id": "scope.r1",
                "subclaim": "scope",
                "kind": "note",
                "concept": "scope",
                "reason": "Observed unweighted self-report.",
                "label": "caveat",
            },
        ],
    }


GOOD = (
    "[c1.r3] In this sample, 46.8% of women (n=1,598) and 31.1% of men know the brand, a 15.7 percentage point "
    "gap (95% interval 12.3 to 19.1 points). This supports the first part, although it does not hold after "
    "multiple-testing adjustment. [c2.r3] In Florida 52.5% know the brand, which is not clearly different from "
    "half (interval -2.4 to 7.4 points). Whether the media causes low awareness among young people cannot be "
    "assessed [gap.r2]. These are observed, unweighted survey results."
)


def test_facts_map_labels_to_directions_and_recompute_holm_for_old_packs():
    facts = build_facts(evidence(), CLAIM)
    parts = {p["part_id"]: p for p in facts["parts"]}
    first = parts["c1"]["findings"][0]
    assert (
        first["direction"] == "supports"
        and first["label_holm"] == "no_clear_difference"
    )
    assert first["robust_after_adjustment"] is False
    assert parts["c2"]["findings"][0]["direction"] == "inconclusive"
    assert parts["c2"]["text"].startswith("more than 50% of Florida")
    assert parts["c1"]["suppressed"] == [
        {"evidence_id": "c1.r2", "group": "Nonbinary", "n": 7}
    ]
    assert (
        parts["gap"]["gaps"][0]["concept"] == "causal effect"
        and facts["causal_relevant"]
    )


def test_expectations_cover_directions_gaps_suppression_and_case_rules():
    items = {i["id"]: i for i in build_expectations(build_facts(evidence(), CLAIM))}
    assert set(items) == {
        "E1:c1.r3",
        "E9:c1.r3",
        "E2:c2.r3",
        "E3:gap.r2",
        "E7:c1.r2:Nonbinary",
        "E5",
        "E6",
        "E8",
    }
    assert "supporting evidence" in items["E1:c1.r3"]["statement"]
    assert items["E1:c1.r3"]["critical"] and not items["E9:c1.r3"]["critical"]
    # Women lower than men although the claim expects higher; survives Holm, so no E9.
    contradicted = evidence(
        label="inconsistent", value=-0.1573, ci=[-0.1911, -0.1229], p_holm=0.01
    )
    ids = {i["id"] for i in build_expectations(build_facts(contradicted, CLAIM))}
    assert "E10" in ids and "E1:c1.r3" in ids and "E9:c1.r3" not in ids


def test_causal_item_only_when_claim_or_evidence_is_causal():
    pack = evidence()
    pack["results"] = [r for r in pack["results"] if r["subclaim"] != "gap"]
    plain = {
        i["id"]
        for i in build_expectations(
            build_facts(pack, "Women know the brand more often than men.")
        )
    }
    causal = {
        i["id"]
        for i in build_expectations(
            build_facts(pack, "Women know the brand because of our ads.")
        )
    }
    assert "E6" not in plain and "E6" in causal


def test_label_holm_from_new_packs_is_used_directly():
    facts = build_facts(evidence(label_holm="consistent"), CLAIM)
    assert facts["parts"][0]["findings"][0]["robust_after_adjustment"] is True


def test_rounded_grounded_prose_passes_all_checks():
    pack = evidence()
    result = check_answer(GOOD, pack, build_facts(pack, CLAIM))
    assert result["passed"], json.dumps(result, indent=1)


@pytest.mark.parametrize(
    "answer, failing",
    [
        (
            GOOD.replace("[c2.r3]", "[c9.r3]"),
            {"D1_citations_valid", "D4_required_ids_cited"},
        ),
        (GOOD.replace("52.5%", "61.0%"), {"D2_numbers_grounded"}),
        (GOOD.replace("[c1.r3] ", ""), {"D4_required_ids_cited"}),
        (
            GOOD + " Nonbinary respondents were at 40.0%.",
            {"D2_numbers_grounded", "D3_no_suppressed_values"},
        ),
    ],
)
def test_each_deterministic_check_catches_its_error(answer, failing):
    pack = evidence()
    checks = check_answer(answer, pack, build_facts(pack, CLAIM))["checks"]
    assert {name for name, c in checks.items() if not c["passed"]} == failing


def test_suppressed_base_years_claim_numbers_and_confidence_level_are_exempt():
    pack = evidence()
    answer = (
        GOOD
        + " Nonbinary respondents (n=7) are too few to report. The 2022 wave used a 95% confidence level."
    )
    assert check_answer(answer, pack, build_facts(pack, CLAIM))["passed"]


def test_mixed_group_numbers_need_semantic_attribution_not_cooccurrence():
    pack = evidence()
    answer = GOOD + ' Nonbinary has n=7 and no reported share, whereas Women have 46.8%.'
    result = check_answer(answer, pack, build_facts(pack, CLAIM))
    check = result['checks']['D3_no_suppressed_values']
    assert check['passed'] and check['requires_E7_semantic_review']
    assert any(i['type'] == 'E7_suppression' and i['critical']
               for i in build_expectations(build_facts(pack, CLAIM)))


def test_explicit_suppressed_share_still_fails():
    pack = evidence()
    answer = GOOD + ' Nonbinary has a share of 46.8%.'
    result = check_answer(answer, pack, build_facts(pack, CLAIM))
    assert not result['checks']['D3_no_suppressed_values']['passed']


def test_ambiguous_attribution_cannot_bypass_critical_suppression_judge():
    from odyn_sft.survey_evaluation.synthesis_judge import score
    pack = evidence()
    facts = build_facts(pack, CLAIM)
    items = build_expectations(facts)
    checks = check_answer(GOOD + ' Nonbinary has 46.8%, like Women.', pack, facts)
    judged = {'valid': True, 'results': {
        item['id']: {'passed': item['type'] != 'E7_suppression'} for item in items}}
    assert not score(judged, items, checks)['case_passed']


def test_ids_are_not_read_as_numbers():
    assert cited_ids("see [c1.r3, gap.r2] and scope.r1") == [
        "c1.r3",
        "gap.r2",
        "scope.r1",
    ]


def test_documented_study_age_is_not_an_invented_result():
    pack = evidence()
    pack["study"]["scope"] = "Adults aged 18 and over in Westmark"
    result = check_answer(
        GOOD + " The study includes adults aged 18 and over.",
        pack,
        build_facts(pack, CLAIM),
    )
    assert result["checks"]["D2_numbers_grounded"]["passed"]


def test_unavailable_comparison_requires_explicit_acknowledgement():
    pack = evidence(
        label="unavailable", value=None, ci=None, reason="Empty comparison group"
    )
    items = {i["id"]: i for i in build_expectations(build_facts(pack, CLAIM))}
    assert items["E11:c1.r3"]["critical"]
    assert "one observed group alone" in items["E11:c1.r3"]["statement"]
