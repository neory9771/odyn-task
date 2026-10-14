"""Exercise the real epoch driver and update rollback with tiny CPU adapters."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from odyn_sft.common.storage import read_json
from odyn_sft.pytorch_sft.config import load_config, lr_multiplier, update_windows
from odyn_sft.pytorch_sft.data import TokenDataset
from odyn_sft.pytorch_sft.monitoring.training_logs import TrainingLogs
from odyn_sft.pytorch_sft.training.checkpoints import (
    latest_checkpoint,
    restore_checkpoint,
)
from odyn_sft.pytorch_sft.training.loop import fit
from odyn_sft.pytorch_sft.training.model_setup import (
    MemoryBudget,
    TrainingComponents,
    create_optimizer,
    prepare_base_for_lora,
)
from odyn_sft.pytorch_sft.training.run_manifest import write_run_manifest
from odyn_sft.pytorch_sft.training.run_setup import RunPlan
from odyn_sft.pytorch_sft.training.session import TrainingSession
from odyn_sft.pytorch_sft.training.updates import run_update

ROOT = Path(__file__).resolve().parents[1]


class SavedTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer_config.json").write_text("{}")


def make_session(directory, monkeypatch, *, resume=False, stop=lambda: False, **overrides):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(19)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = directory / "prepared"
    prepared.mkdir(exist_ok=True)
    for split, count in [("train", 8), ("validation", 2)]:
        ids = np.random.default_rng(31).integers(3, 64, (count, 12), dtype=np.int32)
        ids.tofile(prepared / f"{split}.tokens.bin")
        np.save(prepared / f"{split}.offsets.npy", np.arange(count + 1, dtype=np.int64) * 12)
        np.save(prepared / f"{split}.prompts.npy", np.full(count, 6, dtype=np.int32))
    config = replace(
        load_config(ROOT / "configs/mistral-sft2-bf16.json"),
        output=str(directory),
        prepared=str(prepared),
        thermal_log=str(directory / "gpu.jsonl"),
        microbatch_size=1,
        gradient_accumulation=2,
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.1,
        max_length=128,
        loss_chunk_tokens=4,
        **overrides,
    )
    train, val = TokenDataset(prepared, "train"), TokenDataset(prepared, "validation")
    plan = RunPlan(
        config,
        directory,
        SavedTokenizer(),
        train,
        val,
        update_windows(8, 2),
        12,
        1,
        {"provisional": True},
        {},
        {},
        {},
        "offline-fixture",
        None,
        None,
    )
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    )
    model = prepare_base_for_lora(model, config, quantized=False)
    lora = LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.1, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer, options = create_optimizer(config, [p for _, p in named])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, 12, 1)
    )
    memory = MemoryBudget(torch.device("cpu"), 0, 0, 0, 0)
    components = TrainingComponents(
        model, [p for _, p in named], named, optimizer, scheduler, options, lora, None, memory
    )
    monkeypatch.setattr(
        "odyn_sft.pytorch_sft.training.run_manifest.subprocess.check_output",
        lambda *a, **kw: "CPU correctness fixture",
    )
    manifest = write_run_manifest(plan, components)
    state = {
        "global_step": 0,
        "epoch": 0,
        "position": 0,
        "validated": [],
        "best_loss": None,
        "nonzero_lr_updates": 0,
    }
    if resume:
        checkpoint = latest_checkpoint(directory, plan.fingerprint)
        state = restore_checkpoint(checkpoint, model, optimizer, scheduler)
    logs = TrainingLogs(plan, manifest, state, resumed=resume)
    return TrainingSession(plan, components, state, manifest, logs, stop)


def trainable_weights(session):
    return {name: p.detach().clone() for name, p in session.components.named_trainable}


def test_real_epoch_driver_resumes_pending_validation_and_matches_uninterrupted(
    tmp_path, monkeypatch
):
    full = make_session(tmp_path / "full", monkeypatch)
    try:
        summary = fit(full)
        expected = trainable_weights(full)
        expected_rng = torch.get_rng_state()
        expected_scheduler = full.components.scheduler.state_dict()
        assert summary["acceptance_checks"]["passed"]
        assert summary["global_step"] == 12 and summary["epoch"] == 3
        assert len(summary["validated"]) == 12
    finally:
        full.logs.close()
    interrupted = make_session(tmp_path / "resumed", monkeypatch)
    try:
        with pytest.raises(SystemExit) as error:
            fit(interrupted, interrupt_after_updates=2)
        assert error.value.code == 75
        assert interrupted.state["position"] == 4
        assert "epoch-1-examples-0004" not in interrupted.state["validated"]
    finally:
        interrupted.logs.close()
    resumed = make_session(tmp_path / "resumed", monkeypatch, resume=True)
    try:
        summary = fit(resumed)
        assert summary["acceptance_checks"]["passed"] and len(summary["validated"]) == 12
        assert resumed.components.scheduler.state_dict() == expected_scheduler
        assert torch.equal(torch.get_rng_state(), expected_rng)
        for name, value in trainable_weights(resumed).items():
            assert torch.equal(value, expected[name])
        selection = read_json(resumed.plan.output / "best_adapter/selection.json")
        assert selection["training_complete"]
    finally:
        resumed.logs.close()


def test_partial_window_stop_restores_rng_and_never_commits_gradients(tmp_path, monkeypatch):
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls == 2  # Stop between the two microbatches of one update.

    session = make_session(tmp_path / "partial", monkeypatch, stop=stop)
    expected = trainable_weights(session)
    rng = torch.get_rng_state().clone()
    scheduler = session.components.scheduler.state_dict()
    try:
        with pytest.raises(SystemExit) as error:
            run_update(session, 0, 2, list(range(8)))
        assert error.value.code == 75
        assert session.state["global_step"] == session.state["position"] == 0
        assert torch.equal(torch.get_rng_state(), rng)
        assert session.components.scheduler.state_dict() == scheduler
        assert not session.components.optimizer.state  # No Adam moment update occurred.
        for name, p in session.components.named_trainable:
            assert p.grad is None and torch.equal(p, expected[name])
        assert latest_checkpoint(session.plan.output, session.plan.fingerprint) is not None
    finally:
        session.logs.close()


def test_bootstrap_uses_real_setup_and_model_helpers_without_gpu_download(tmp_path, monkeypatch):
    """Mock hardware/loading, while exercising the extracted setup wiring itself."""
    from odyn_sft.common.storage import digest
    from odyn_sft.pytorch_sft.training.loop import StopRequest, run_training

    initial = make_session(tmp_path / "bootstrap", monkeypatch)
    config = initial.plan.config
    initial.logs.close()
    (Path(config.output) / "run_manifest.json").unlink()
    tokenizer = SavedTokenizer()
    tokenizer.chat_template = "offline native template"
    monkeypatch.setattr(
        "odyn_sft.pytorch_sft.training.run_setup.validate_prepared",
        lambda _: {"provisional": True, "chat_template_hash": digest(tokenizer.chat_template)},
    )
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    from transformers import Qwen2Config, Qwen2ForCausalLM

    def loaded_model(*args, **kwargs):
        assert kwargs["local_files_only"] is True and kwargs["quantization_config"] is None
        return Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
        ).to(torch.bfloat16)

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", loaded_model)
    monkeypatch.setattr(
        "odyn_sft.pytorch_sft.training.model_setup.configure_memory",
        lambda _: MemoryBudget(torch.device("cpu"), 0, 0, 0, 0),
    )
    with pytest.raises(SystemExit) as error:
        run_training(config, StopRequest(), interrupt_after_updates=1)
    assert error.value.code == 75
    manifest = read_json(Path(config.output) / "run_manifest.json")
    assert manifest["effective"]["total_optimizer_updates"] == 12
    assert manifest["quantization"]["quant_method"] == "none"
    assert read_json(Path(config.output) / "first_update_verification.json")["passed"]


def test_generated_selection_prefers_generated_rate_over_lower_loss(tmp_path, monkeypatch):
    """Loss rises while the generated rate improves: only generated selection follows quality."""
    calls = {"loss": 0, "generated": 0}

    def fake_evaluate(model, tokenizer, dataset, config, device, output, tag, stop, fingerprint):
        if config.validation_generation:
            calls["generated"] += 1
            assert tag.endswith("-generated") and len(dataset) == 1
            return {"loss": 9.0, "examples": 1, "generated_output_rate": min(calls["generated"], 6) / 6}
        calls["loss"] += 1
        return {"loss": 1.0 + calls["loss"] / 100, "examples": len(dataset)}

    monkeypatch.setattr("odyn_sft.pytorch_sft.training.selection.evaluate", fake_evaluate)
    session = make_session(
        tmp_path, monkeypatch, generation_check_examples=1, selection_metric="generated"
    )
    try:
        summary = fit(session)
    finally:
        session.logs.close()
    assert summary["acceptance_checks"]["passed"] and calls == {"loss": 12, "generated": 12}
    selection = read_json(tmp_path / "best_adapter/selection.json")
    # First boundary reaching rate 1.0 wins; later ties lose on their higher full-set loss.
    assert selection["tag"] == "epoch-2-examples-0004"
    assert selection["loss"] == pytest.approx(1.06)
    assert selection["selection_key"] == [-1.0, pytest.approx(1.06)]
    assert selection["generation_check"]["examples"] == 1


def test_generated_selection_requires_generation_and_subset_is_fixed(tmp_path):
    config = load_config(ROOT / "configs/mistral-sft2-bf16.json")
    with pytest.raises(ValueError, match="Generated selection needs"):
        replace(config, validation_generation=False, selection_metric="generated")
    with pytest.raises(ValueError, match="only to loss-only"):
        replace(config, validation_generation=True, generation_check_examples=4)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    np.zeros(10 * 4, dtype=np.int32).tofile(prepared / "validation.tokens.bin")
    np.save(prepared / "validation.offsets.npy", np.arange(11, dtype=np.int64) * 4)
    np.save(prepared / "validation.prompts.npy", np.full(10, 2, dtype=np.int32))
    data = TokenDataset(prepared, "validation")
    assert data.subset(4).indices == [0, 2, 5, 7] == data.subset(4).indices
    assert data.subset(20).indices == data.indices and len(data) == 10
