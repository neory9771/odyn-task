"""Both delivered Mistral tasks support isolated BF16 and NF4 runs."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename,task,quantization,optimizer", [
    ("mistral-bf16.json", "sft1", "none", "adamw"),
    ("mistral-nf4.json", "sft1", "nf4", "adamw_8bit"),
    ("mistral-sft2-bf16.json", "sft2", "none", "adamw"),
    ("mistral-sft2-nf4.json", "sft2", "nf4", "adamw_8bit"),
])
def test_mistral_profile(filename, task, quantization, optimizer):
    from odyn_sft.pytorch_sft.config import load_config
    from odyn_sft.tasks import get_task

    config = load_config(ROOT / "configs" / filename)
    assert config.task == task
    assert get_task(task) is not None
    assert config.model == "mistralai/Mistral-Small-24B-Instruct-2501"
    assert config.revision == "9527884be6e5616bdd54de542f9ae13384489724"
    assert config.quantization == quantization
    assert config.optimizer == optimizer
    assert config.system_prompt
    assert ("executed analysis" in config.system_prompt) == (task == "sft2")
    profiles = [load_config(path) for path in (ROOT / "configs").glob("mistral*.json")]
    for field in ("prepared", "output", "thermal_log"):
        paths = [str(getattr(profile, field)) for profile in profiles]
        assert len(paths) == len(set(paths))
