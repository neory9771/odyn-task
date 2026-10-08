"""The public dataset CLI routes real artifacts and preserves generation gates."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_synthesis_suite import inputs

from odyn_sft.dataset_generation.cli import main


@pytest.mark.parametrize("domain", ["metadata", "evaluation", "splits", "synthesis", "variants", "use-cases", "rubrics", "synthetic"])
def test_help_is_lightweight(domain):
    code = """
import sys
from odyn_sft.dataset_generation.cli import build_parser
parser = build_parser()
assert 'pandas' not in sys.modules
assert 'tiktoken' not in sys.modules
assert 'openai' not in sys.modules
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
parser.parse_args([DOMAIN, '--help'])
"""
    code = code.replace("DOMAIN", repr(domain))
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: odyn-dataset" in result.stdout


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["splits"],
        ["splits", "generate", "--plan", "p", "--out", "o"],
        ["evaluation", "generate", "--plan", "p", "--out", "o"],
        [
            "synthesis",
            "assemble",
            "--targets-dir",
            "t",
            "--source-suite",
            "s",
            "--out",
            "o",
            "--max-example-tokens",
            "10",
        ],
        [
            "synthesis",
            "assemble",
            "--targets-dir",
            "t",
            "--source-suite",
            "s",
            "--out",
            "o",
            "--test-limit",
            "0",
        ],
    ],
)
def test_invalid_arguments_never_dispatch(monkeypatch, args):
    from odyn_sft.dataset_generation import cli

    monkeypatch.setattr(
        cli, "_dispatch", lambda _: pytest.fail("invalid arguments dispatched")
    )
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2


def test_synthesis_assembly_and_validation(tmp_path, capsys):
    targets, source = inputs(tmp_path)
    output = tmp_path / "assembled"
    assert (
        main(
            [
                "synthesis",
                "assemble",
                "--targets-dir",
                str(targets),
                "--source-suite",
                str(source),
                "--out",
                str(output),
                "--test-limit",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert report["target_policy"]["test"] == "source_draft_reference_not_writer_gated"
    assert main(["synthesis", "validate", "--suite", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["integrity_valid"] is True
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "synthesis",
                "assemble",
                "--targets-dir",
                str(targets),
                "--source-suite",
                str(source),
                "--out",
                str(output),
            ]
        )
    assert exc.value.code == 1


def test_evaluation_config_mapping(monkeypatch, tmp_path, capsys):
    from odyn_sft.dataset_generation import config, generation

    resolved = dict(
        source="survey",
        metadata="metadata",
        size=1000,
        calibration_size=121,
        seed=42,
        overlap_corpus=["prior"],
        coverage_minima={"A": {}},
        interaction_minima={},
        heldout_outcome_blocks=["block"],
        scenario_rules={},
        study_bindings={"groups": []},
    )
    monkeypatch.setattr(config, "load_config", lambda _: resolved)
    seen = {}

    def audit(**kwargs):
        seen.update(kwargs)
        return {"feasible": False, "large_pool": ["not printed"]}

    monkeypatch.setattr(generation, "audit", audit)
    assert (
        main(["evaluation", "audit", "--config", "config.json", "--out", str(tmp_path)])
        == 0
    )
    assert seen["metadata_path"] == "metadata"
    assert seen["overlaps"] == ["prior"]
    assert seen["heldout_blocks"] == ["block"]
    assert seen["study_bindings"] == resolved["study_bindings"]
    report = json.loads(capsys.readouterr().out)
    assert report["feasible"] is False
    assert "large_pool" not in report


@pytest.mark.parametrize("domain", ["evaluation", "splits"])
def test_generate_forwards_review_acknowledgement(monkeypatch, domain):
    from odyn_sft.dataset_generation import dataset, generation

    seen = {}

    def generate(plan, output, *, acknowledge_plan):
        seen.update(plan=plan, output=output, acknowledged=acknowledge_plan)
        return {}

    monkeypatch.setattr(
        generation if domain == "evaluation" else dataset,
        "generate" if domain == "evaluation" else "generate_splits",
        generate,
    )
    assert (
        main(
            [domain, "generate", "--plan", "p.json", "--out", "o", "--acknowledge-plan"]
        )
        == 0
    )
    assert seen == {"plan": Path("p.json"), "output": Path("o"), "acknowledged": True}


def test_metadata_output_is_immutable(monkeypatch, tmp_path):
    from odyn_sft.dataset_generation import metadata

    monkeypatch.setattr(metadata, "build_metadata", lambda _: {"study": "fixture"})
    output = tmp_path / "metadata.json"
    assert main(["metadata", "build", "--source", "survey", "--out", str(output)]) == 0
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["metadata", "build", "--source", "survey", "--out", str(output)])
    assert exc.value.code == 1
    assert output.read_bytes() == before


def test_synthetic_help_is_lightweight():
    code = """
import sys
from odyn_sft.dataset_generation.cli import build_parser
parser = build_parser()
assert 'tiktoken' not in sys.modules
assert 'pandas' not in sys.modules
assert 'torch' not in sys.modules
parser.parse_args(['synthetic', '--help'])
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "validate-curriculum" in result.stdout
    assert "filter" in result.stdout


def test_synthetic_single_study_cli(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    output = tmp_path / "generated"
    assert main(["synthetic", "spec", "--domain", "brand_tracking",
                 "--study-id", "demo", "--seed", "7", "--population-size",
                 "20000", "--output", str(spec)]) == 0
    assert spec.exists()
    capsys.readouterr()
    assert main(["synthetic", "study", "--spec", str(spec),
                 "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["study_id"] == "demo"
    assert report["respondents"] > 0
    assert (output / "study" / "respondents.parquet").exists()
    assert (output / "hidden").exists()
    with pytest.raises(SystemExit) as exc:
        main(["synthetic", "spec", "--domain", "brand_tracking",
              "--study-id", "demo", "--seed", "7", "--output", str(spec)])
    assert exc.value.code == 1
