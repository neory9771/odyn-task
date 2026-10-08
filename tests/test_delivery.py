"""Delivery contracts that do not load/download model weights or require a GPU."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from odyn_sft.run_management.cli import main
from odyn_sft.run_management.supervisor import gpu_selector, read_gpu
from odyn_sft.pytorch_sft.data import validate_source_suite

ROOT = Path(__file__).resolve().parents[1]


def test_config_resolution_is_independent_of_working_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = ROOT / "configs/mistral-bf16.json"
    assert main(["config", "--config", str(path)]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["source_suite"] == str(ROOT / "data/suite")
    assert config["output"] == str(ROOT / "runs/mistral-bf16/train")
    assert config["quantization"] == "none"
    assert config["system_prompt"] and config["wandb_entity"] is None


@pytest.mark.parametrize("argv", [
    ["prepare"], ["probe", "--batch", "0", "--out", "unused"],
    ["validate", "--interrupt-after-updates", "1"],
    ["train", "--interrupt-after-updates", "0"],
    ["train", "--batch", "1"],
])
def test_invalid_commands_fail_before_gpu_or_network_work(argv):
    if argv[0] != "prepare":
        argv = [*argv, "--config", str(ROOT / "configs/mistral-bf16.json")]
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2


def test_packaged_provenance_has_no_original_workspace_dependency():
    from odyn_sft.survey_evaluation.provenance import implementation_hashes
    result = implementation_hashes()
    assert len(result) > 20 and all(len(value) == 64 for value in result.values())
    source = (ROOT / "src/odyn_sft/pytorch_sft/training/run_setup.py").read_text()
    assert "/home/ryadh/" not in source and 'scripts/launch_sft_local.py' not in source


@pytest.mark.parametrize("version, target", [
    ("synthetic-client-sft-1", "validate_training_suite"),
    ("synthetic-smallest100-sft-1", "validate_small_training_suite"),
    ("other", "validate_splits_package"),
])
def test_prepare_and_final_evaluation_share_suite_validation(tmp_path, monkeypatch, version, target):
    (tmp_path / "manifest.json").write_text(json.dumps({"version": version}))
    module = "survey_evaluation.package_integrity" if version == "other" else "tasks.sft1_suite"
    monkeypatch.setattr(f"odyn_sft.{module}.{target}", lambda path: {"validator": target})
    assert validate_source_suite(tmp_path) == {"validator": target}


def test_gpu_monitor_targets_selected_physical_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-example")
    assert gpu_selector() == "GPU-example"
    def check(argv, **kwargs):
        assert argv[:3] == ["nvidia-smi", "-i", "GPU-example"]
        return "78, 5000, 100, 350\n"
    monkeypatch.setattr("odyn_sft.run_management.supervisor.subprocess.check_output", check)
    assert read_gpu(gpu_selector())["temperature_c"] == 78
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(ValueError, match="one GPU"):
        gpu_selector()


def test_current_contract_is_unchanged_and_legacy_versions_are_removed():
    import hashlib

    from odyn_sft.prompts.authoring import contract_manifest
    manifest = contract_manifest()
    encoded = json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    # Frozen digest of the current manifest before compatibility code was removed.
    assert hashlib.sha256(encoded).hexdigest() == '5affa18fdb6e869d7400aea94000771a9e12c9dd064d7354d2c6512f1f9883a1'
    with pytest.raises(ValueError, match="supports only"):
        contract_manifest("survey-api-baseline-prompt-0.1.0")
    assert not (ROOT / "src/odyn_sft/legacy_api_baseline_contract.py").exists()


def test_core_fingerprints_cover_nested_modules_and_supervisor_override(tmp_path, monkeypatch):
    from odyn_sft.common.storage import file_hash
    from odyn_sft.pytorch_sft.provenance import implementation_hashes
    supervisor = tmp_path / "supervisor.py"
    supervisor.write_text("# dedicated worker supervisor\n")
    monkeypatch.setenv("ODYN_SUPERVISOR_SOURCE", str(supervisor))
    hashes = implementation_hashes()
    for name in ("data/preparation.py", "data/tokenization.py", "training/loop.py",
                 "training/objective.py", "evaluation/evaluator.py", "evaluation/final.py",
                 "monitoring/training_logs.py", "config.py", "provenance.py",
                 "run_management/cli.py", "run_management/supervisor.py",
                 "run_management/probe.py"):
        assert name in hashes and len(hashes[name]) == 64
    assert hashes["supervisor"] == file_hash(supervisor)
    assert {"training/__init__.py", "data/__init__.py", "evaluation/__init__.py",
            "monitoring/__init__.py"} <= hashes.keys()
