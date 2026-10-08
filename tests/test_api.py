import pandas as pd
import pytest

from odyn_sft.survey_api import (
    AnalysisError,
    SurveyAPI,
    contrast_label,
    difference_ci,
    run_program,
    wilson,
)


@pytest.fixture
def survey():
    data = pd.DataFrame(
        {
            "x": pd.array([True] * 5 + [None] * 15, dtype="boolean"),
            "y": pd.array([None] * 5 + [True] * 5 + [None] * 10, dtype="boolean"),
            "age": ["young"] * 10 + ["old"] * 10,
        }
    )
    cards = [
        {
            "id": v,
            "label": v,
            "block_id": "selection",
            "answer_type": "select_all",
            "values": [True],
            "ordered_values": None,
            "flags": ["unweighted", "eligibility_unknown"],
            "denominator_rule": "observed_block_respondents",
        }
        for v in ["x", "y"]
    ]
    cards.append(
        {
            "id": "age",
            "label": "age",
            "block_id": "age",
            "answer_type": "single",
            "values": ["young", "old"],
            "ordered_values": None,
            "flags": ["unweighted"],
            "denominator_rule": "answered_item",
        }
    )
    metadata = {"hash": "fixture", "variables": cards, "study": {"label": "toy"}}
    return data, metadata


def test_selection_base_and_complement(survey):
    api = SurveyAPI(*survey)
    result = api.proportion("x")["estimates"][0]
    assert result["value"] == 0.5 and result["n"] == 10
    absent = ~api.where("x", True)
    assert absent.mask.sum() == 5  # not all 15 nulls
    assert absent.universe.sum() == 10


def test_cross_variable_or_and_negation_use_common_observed_base(survey):
    data, metadata = survey
    data = data.copy()
    data.loc[[3, 4], "age"] = None
    api = SurveyAPI(data, metadata)
    selected = api.where("x", True)
    older = api.where("age", "old")
    # Ten older respondents are observed for age but outside the checkbox base.
    # OR therefore retains only three selectors observed for both variables.
    union = selected | older
    assert union.mask[union.mask].index.tolist() == [0, 1, 2]
    assert union.universe[union.universe].index.tolist() == [0, 1, 2, 5, 6, 7, 8, 9]
    assert (selected & older).mask.sum() == 0
    assert (~union).mask[(~union).mask].index.tolist() == [5, 6, 7, 8, 9]


def test_holm_spans_subclaims_excludes_suppression_and_includes_no_expectation(survey):
    api = SurveyAPI(*survey)
    with api.subclaim("c1", "Descriptive comparison without a direction"):
        estimate = api.proportion("x")
        exploratory = api.compare(estimate, "All observed respondents", 0.5)
    with api.subclaim("c2", "Directional comparison and insufficient base"):
        directional = api.compare(estimate, "All observed respondents", 0.9, expect="<")
        small = api.proportion("y", by=api.group({"selected": api.where("x", True)}))
        unavailable = api.compare(small, "selected", 0.5, expect=">")
    assert exploratory["label"] == "not_tested" and exploratory["p_value"] == 1
    assert "p_holm" not in exploratory  # Adjustment waits for final assembly.
    assert unavailable["p_value"] is None and unavailable["label"] == "unavailable"
    pack = api.pack()
    contrasts = {r["id"]: r for r in pack["results"] if r["kind"] == "contrast"}
    assert pack["test_count"] == 2
    assert contrasts[directional["id"]]["p_holm"] == pytest.approx(2 * directional["p_value"])
    assert contrasts[exploratory["id"]]["p_holm"] == 1
    assert contrasts[unavailable["id"]]["p_holm"] is None


def test_suppression_hides_numerator_and_percentage(survey):
    api = SurveyAPI(*survey)
    group = api.group({"picked_x": api.where("x", True)})
    row = api.proportion("y", by=group)["estimates"][0]
    assert row["n"] == 5
    assert row["value"] is None and row["numerator"] is None and row["ci"] is None


def test_threshold_ci_and_labels(survey):
    api = SurveyAPI(*survey)
    est = api.proportion("x")
    r = api.compare(est, "All observed respondents", 0.5, expect=">")
    assert r["label"] == "no_clear_difference" and r["value"] == 0
    assert r["test"] == "wilson_score"
    assert contrast_label([0.1, 0.2], ">") == "consistent"
    assert contrast_label([0.1, 0.2], "<") == "inconsistent"


def test_refusal_cannot_be_selected(survey):
    data, metadata = survey
    metadata["variables"][2]["values"].append("Prefer not to say")
    data.loc[0, "age"] = "Prefer not to say"
    api = SurveyAPI(data, metadata)
    with pytest.raises(AnalysisError, match="RefusalExcluded"):
        api.proportion("age", "Prefer not to say")
    with pytest.raises(AnalysisError, match="RefusalExcluded"):
        api.where("age", ["young", "Prefer not to say"])
    assert list(api.group("age")) == ["young", "old"]  # default groups skip the refusal


def test_holm_label_downgrades_unadjusted_labels(survey):
    api = SurveyAPI(*survey)
    api.append("contrast", contrast_type="difference", value=0.1, expect=">",
               ci=[0.01, 0.2], p_value=0.03, label="consistent")
    api.append("contrast", contrast_type="difference", value=-0.1, expect=">",
               ci=[-0.2, -0.1], p_value=0.04, label="inconsistent")
    first, second = api.pack()["results"]
    assert first["label"] == "consistent" and first["p_holm"] == pytest.approx(0.06)
    assert first["label_holm"] == second["label_holm"] == "no_clear_difference"


def test_reference_intervals():
    ci = wilson(1045, 2007)
    assert ci == pytest.approx([0.498817, 0.542489], abs=0.0001)
    assert difference_ci(163, 258, 37, 289) == pytest.approx(
        [0.4293578, 0.5695578], abs=0.00001
    )


def test_holm_uses_every_test(survey):
    api = SurveyAPI(*survey)
    for p in [0.01, 0.03, 0.04]:
        api.append("contrast", p_value=p)
    pack = api.pack()
    assert [r["p_holm"] for r in pack["results"]] == pytest.approx([0.03, 0.06, 0.06])


@pytest.mark.parametrize("program", ["var(", "return 1", "break"])
def test_python_syntax_errors(program, survey):
    with pytest.raises(AnalysisError, match="InvalidSyntax"):
        run_program(program, *survey)


def test_normal_python_is_supported(survey):
    pack = run_program(
        "import math\ndef analyse():\n    for name in [str(n) for n in range(2)]:\n        caveat(name)\n    e = proportion('x')\n    compare(e, 'All observed respondents', math.sqrt(0.25), expect='>')\nif __name__ == '__main__':\n    analyse()",
        *survey,
    )
    assert pack["test_count"] == 1 and len(pack["results"]) == 4


def test_native_execution_and_overlap(survey):
    pack = run_program(
        "with subclaim('c1', 'selection'):\n    e = proportion(var('x'))\n    compare(e, 'All observed respondents', 0.5, expect='>')",
        *survey,
    )
    assert (
        pack["test_count"] == 1
        and pack["results"][-1]["label"] == "no_clear_difference"
    )
    with pytest.raises(AnalysisError, match="OverlappingSegments"):
        run_program("g = group(var('age'), {'a': ['young'], 'b': ['young']})", *survey)
