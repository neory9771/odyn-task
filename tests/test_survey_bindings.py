"""Study configuration isolation and prompt relocation checks without paid calls."""

import json

import pandas as pd
import pytest

from odyn_sft.dataset_generation.comparison_groups import load_comparison_groups
from odyn_sft.dataset_generation.config import load_config
from odyn_sft.dataset_generation.dataset import load_config as load_dataset_config
from odyn_sft.dataset_generation.generation import enumerate_analyses
from odyn_sft.dataset_generation.partition import enumerate_family_specs
from odyn_sft.dataset_generation.provenance import implementation_hashes
from odyn_sft.dataset_generation.study import group_definitions
from odyn_sft.prompts.survey import PROMPT_VERSION, analysis_input


def bindings():
    return {
        "version": "survey-bindings-1",
        "study_id": "new-study",
        "groups": [{"variable": "segment", "groups": {"A group": ["A"], "B group": ["B"]}}],
    }


def metadata():
    return {
        "study": {},
        "variables": [
            {
                "id": "segment",
                "values": ["A", "B"],
                "answer_type": "single",
                "section": "demographics",
                "block_id": "segment-block",
            },
            {
                "id": "response",
                "values": ["Yes", "No"],
                "answer_type": "single",
                "section": "outcomes",
                "block_id": "response-block",
            },
        ],
    }


def test_new_study_controls_family_and_observed_analysis_generation():
    study = metadata()
    cards = {c["id"]: c for c in study["variables"]}
    assert group_definitions(cards) == []  # The 2022 preset cannot invent bindings.
    configured = bindings()
    assert group_definitions(cards, configured) == [("segment", configured["groups"][0]["groups"])]
    families = enumerate_family_specs(study, 1, configured)
    data = pd.DataFrame(
        {
            "segment": ["A"] * 40 + ["B"] * 40,
            "response": ["Yes"] * 30 + ["No"] * 10 + ["Yes"] * 10 + ["No"] * 30,
        }
    )
    analyses = enumerate_analyses(data, study, 1, configured)
    assert len(families) == len(analyses) == 4
    assert {f["row"]["group_variable"] for f in families + analyses} == {"segment"}
    assert next(a for a in analyses if not a["threshold"] and a["row"]["value"] == "Yes")[
        "counts"
    ] == [(40, 30), (40, 10)]
    assert group_definitions(cards, configured)  # Does not consume or mutate bindings.


@pytest.mark.parametrize(
    "broken",
    [
        {"version": "bad", "study_id": "new-study", "groups": []},
        {"version": "survey-bindings-1", "study_id": "new-study", "groups": []},
        {
            "version": "survey-bindings-1",
            "study_id": "new-study",
            "groups": [{"variable": "segment", "groups": {"A": ["A"], "B": ["A"]}}],
        },
        {
            "version": "survey-bindings-1",
            "study_id": "new-study",
            "groups": [{"variable": "segment", "groups": {"A": ["Prefer not to say"], "B": ["B"]}}],
        },
    ],
)
def test_invalid_study_bindings_rejected(broken):
    with pytest.raises(ValueError):
        load_comparison_groups(broken)


@pytest.mark.parametrize("loader", [load_config, load_dataset_config])
@pytest.mark.parametrize("inline", [False, True])
def test_configuration_resolves_and_freezes_custom_binding_content(tmp_path, loader, inline):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binding_file = workspace / "bindings.json"
    binding_file.write_text(json.dumps(bindings()))
    config = {
        "workspace": "workspace",
        "source": "data.parquet",
        "metadata": "metadata.json",
        "overlap_corpus": [],
        "heldout_outcome_blocks": [],
        "seed": 1,
        "scenario_rules": {},
        "study_bindings": bindings() if inline else "bindings.json",
    }
    if loader is load_config:
        config.update(size=10, calibration_size=1, coverage_minima={}, interaction_minima={})
    else:
        config.update(
            train_variants=1,
            partition_ratios={"train": 0.6, "validation": 0.2, "test": 0.2},
            splits={
                s: {"size": 10, "coverage_minima": {}, "interaction_minima": {}}
                for s in ["train", "validation", "test"]
            },
        )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = loader(path)
    assert result["study_bindings"] == bindings()
    binding_file.write_text("{}")
    assert result["study_bindings"] == bindings()  # Frozen plan data survives source edits.
    assert result["source"] == str(workspace / "data.parquet")


def test_packaged_preset_and_relocated_prompts_are_available():
    preset = load_comparison_groups()
    assert len(preset["groups"]) == 24
    isolated = load_comparison_groups(preset)
    isolated["groups"].clear()
    assert preset["groups"]
    hashes = implementation_hashes()
    assert "dataset_generation/resources/survey_2022_bindings.json" in hashes
    assert "prompts/survey.py" in hashes and "dataset_generation/dataset/planning.py" in hashes
    assert PROMPT_VERSION == "two-call-prompts-1.16.0"
    assert callable(analysis_input)
