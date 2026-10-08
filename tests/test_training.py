"""CPU correctness checks for the custom loss, accumulation schedule and resume."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from odyn_sft.pytorch_sft.config import (
    load_config,
    lr_multiplier,
    quarter_boundaries,
    update_windows,
)


def test_full_experiment_quarters_and_explicit_parameters():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/mistral-nf4.json")
    windows = update_windows(1000, config.gradient_accumulation)
    assert len(windows) == 64
    assert quarter_boundaries(1000) == [250, 500, 750, 1000]
    assert all(b in {end for _, end in windows} for b in quarter_boundaries(1000))
    assert sum(end - start for start, end in windows) == 1000
    assert config.learning_rate == 2e-4 and config.epochs == 3
    assert config.lora_rank == 16 and config.lora_alpha == 32
    assert config.target_modules == "all-linear" and config.quantization == "nf4"
    assert (
        config.train_example_limit
        is config.validation_example_limit
        is config.max_steps
        is None
    )


def test_short_accumulation_windows_cover_every_example_once():
    windows = update_windows(11, 4)
    assert [i for a, b in windows for i in range(a, b)] == list(range(11))
    assert set(quarter_boundaries(11)) <= {end for _, end in windows}
    assert lr_multiplier(0, 100, 3) == pytest.approx(1/3)
    assert lr_multiplier(3, 100, 3) == 1
    assert lr_multiplier(100, 100, 3) == 0
    assert lr_multiplier(1000, 100, 3) == 0


def tiny_model():
    torch = pytest.importorskip("torch")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.set_num_threads(1)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.0,
        )
    )


def test_greedy_validation_overrides_sampling_model_defaults():
    from types import SimpleNamespace

    from odyn_sft.pytorch_sft.evaluation.evaluator import generation_kwargs

    model = tiny_model()
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.6
    model.generation_config.transformers_version = "4.57.3"
    options = generation_kwargs(SimpleNamespace(pad_token_id=0, eos_token_id=1), 7)
    resolved, _ = model._prepare_generation_config(**options)
    assert resolved.do_sample is False
    assert resolved.num_beams == 1 and resolved.max_new_tokens == 7
    assert resolved.eos_token_id == 1 and resolved.pad_token_id == 0


def test_completion_projection_matches_standard_causal_loss_and_gradients():
    torch = pytest.importorskip("torch")
    from odyn_sft.pytorch_sft.training.objective import completion_loss

    torch.manual_seed(19)
    model = tiny_model().eval()
    reference = copy.deepcopy(model)
    ids = torch.randint(0, 64, (1, 13))
    labels = ids.clone()
    labels[:, :8] = -100
    loss, count, _ = completion_loss(model, ids, labels, chunk_tokens=2)
    assert count == 5
    expected = reference(input_ids=ids, labels=labels, use_cache=False).loss
    assert torch.allclose(loss / count, expected, atol=1e-6)
    (loss / count).backward()
    expected.backward()
    for actual, target in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual.grad, target.grad, atol=1e-6, rtol=1e-5)
    with torch.inference_mode():
        val_loss, val_count, correct = completion_loss(model, ids, labels, 2)
    assert val_count == count and 0 <= correct <= count
    assert torch.allclose(val_loss, loss.detach(), atol=1e-6)


def test_accumulation_is_weighted_by_completion_tokens():
    torch = pytest.importorskip("torch")
    from odyn_sft.pytorch_sft.training.objective import completion_loss

    torch.manual_seed(41)
    model = tiny_model().eval()
    a = torch.randint(0, 64, (1, 9))
    b = torch.randint(0, 64, (1, 9))
    labels_a, labels_b = a.clone(), b.clone()
    labels_a[:, :7], labels_b[:, :3] = -100, -100
    loss_a, n_a, _ = completion_loss(model, a, labels_a)
    loss_b, n_b, _ = completion_loss(model, b, labels_b)
    expected = model(
        input_ids=torch.cat([a, b]), labels=torch.cat([labels_a, labels_b])
    ).loss
    assert (n_a, n_b) == (2, 6)
    assert torch.allclose((loss_a + loss_b) / (n_a + n_b), expected, atol=1e-6)


def test_atomic_checkpoint_resume_matches_uninterrupted_optimizer_and_rng(tmp_path):
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model

    from odyn_sft.pytorch_sft.training.checkpoints import (
        latest_checkpoint,
        restore_checkpoint,
        save_checkpoint,
    )
    from odyn_sft.pytorch_sft.training.objective import completion_loss

    def fresh():
        torch.manual_seed(83)
        model = get_peft_model(
            tiny_model(),
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.1,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            ),
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=2e-4
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: lr_multiplier(s, 4, 1)
        )
        return model, optimizer, scheduler

    def update(model, optimizer, scheduler):
        ids = torch.randint(0, 64, (1, 10))
        labels = ids.clone()
        labels[:, :6] = -100
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, count, _ = completion_loss(model, ids, labels, 2)
        (loss / count).backward()
        optimizer.step()
        scheduler.step()

    model, optimizer, scheduler = fresh()
    for _ in range(4):
        update(model, optimizer, scheduler)
    expected = {
        k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad
    }
    expected_rng = torch.get_rng_state()
    expected_scheduler = scheduler.state_dict()
    model, optimizer, scheduler = fresh()
    for _ in range(2):
        update(model, optimizer, scheduler)
    state = {"global_step": 2, "epoch": 0, "position": 4, "validated": []}
    save_checkpoint(tmp_path, model, optimizer, scheduler, state, "frozen")
    (tmp_path / "checkpoint-999-incomplete").mkdir()
    checkpoint = latest_checkpoint(tmp_path, "frozen")
    assert checkpoint is not None
    model, optimizer, scheduler = fresh()
    restored = restore_checkpoint(checkpoint, model, optimizer, scheduler)
    assert restored == state
    for _ in range(2):
        update(model, optimizer, scheduler)
    assert scheduler.state_dict() == expected_scheduler
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in model.named_parameters():
        if value.requires_grad:
            assert torch.equal(value, expected[name])
    with pytest.raises(ValueError, match="mismatch"):
        latest_checkpoint(tmp_path, "changed")
    with (checkpoint / "training_state.pt").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        latest_checkpoint(tmp_path, "frozen")


def test_memory_mapped_dataset_masks_prompt_and_keeps_longest_smoke_examples(tmp_path):
    pytest.importorskip("torch")
    from odyn_sft.pytorch_sft.data import TokenDataset

    np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32).tofile(
        tmp_path / "train.tokens.bin"
    )
    np.save(tmp_path / "train.offsets.npy", np.array([0, 3, 7]))
    np.save(tmp_path / "train.prompts.npy", np.array([2, 2]))
    dataset = TokenDataset(tmp_path, "train", limit=1)
    row = dataset[0]
    assert row["source_index"] == 1 and row["target_tokens"] == 2
    assert row["labels"].tolist() == [[-100, -100, 6, 7]]


def test_validation_resumes_completed_cases_without_regenerating(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from dataclasses import replace
    from types import SimpleNamespace

    import pandas as pd

    from odyn_sft.common.storage import write_json, write_jsonl
    from odyn_sft.pytorch_sft.evaluation.evaluator import evaluate

    write_json(tmp_path / "metadata.json", {})
    write_jsonl(
        tmp_path / "private_gold.jsonl",
        [
            {
                "split": "validation",
                "record_id": "v0",
                "analysis_contract": {"required_analyses": []},
            }
        ],
    )
    write_jsonl(tmp_path / "validation_lineage.jsonl", [{"record_id": "v0"}])
    monkeypatch.setattr(
        "odyn_sft.survey_evaluation.study_data.load_inputs",
        lambda *_: (pd.DataFrame(), {}),
    )
    monkeypatch.setattr(
        "odyn_sft.survey_evaluation.execution.execute",
        lambda *_: {"results": [], "notes": []},
    )
    root = Path(__file__).resolve().parents[1]
    config = replace(
        load_config(root / "configs/mistral-nf4.json"),
        source_suite=str(tmp_path),
        max_length=128,
        generation_max_new_tokens=5,
        validation_generation=True,
    )
    np.arange(1, 10, dtype=np.int32).tofile(tmp_path / "validation.tokens.bin")
    np.save(tmp_path / "validation.offsets.npy", np.array([0, 9]))
    np.save(tmp_path / "validation.prompts.npy", np.array([6]))
    from odyn_sft.pytorch_sft.data import TokenDataset

    dataset = TokenDataset(tmp_path, "validation")
    model = tiny_model()
    calls = []

    def generate(**kwargs):
        calls.append(1)
        return torch.cat([kwargs["input_ids"], torch.tensor([[21, 22]])], dim=1)

    monkeypatch.setattr(model, "generate", generate)
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=22,
        decode=lambda *_args, **_kw: "analysis = pack()",
    )
    args = (
        model,
        tokenizer,
        dataset,
        config,
        torch.device("cpu"),
        tmp_path,
        "epoch-1",
        lambda: False,
        "frozen",
    )
    first = evaluate(*args)
    second = evaluate(*args)
    assert calls == [1]
    for key in (
        "loss",
        "perplexity",
        "teacher_forced_token_accuracy",
        "execution_valid_rate",
        "completion_tokens",
    ):
        assert first[key] == second[key]
    assert first["completion_tokens"] == 3 and first["execution_valid_rate"] == 1.0
    with pytest.raises(ValueError, match="cache.*mismatch"):
        evaluate(*args[:-1], "changed")


def test_partial_generation_resumes_prefix_and_reuses_loss(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from dataclasses import replace
    from types import SimpleNamespace

    import pandas as pd

    from odyn_sft.common.storage import read_json, write_json, write_jsonl
    from odyn_sft.pytorch_sft.data import TokenDataset
    from odyn_sft.pytorch_sft.evaluation.evaluator import evaluate

    write_json(tmp_path / "metadata.json", {})
    write_jsonl(
        tmp_path / "private_gold.jsonl",
        [
            {
                "split": "validation",
                "record_id": "v0",
                "analysis_contract": {"required_analyses": []},
            }
        ],
    )
    write_jsonl(tmp_path / "validation_lineage.jsonl", [{"record_id": "v0"}])
    monkeypatch.setattr(
        "odyn_sft.survey_evaluation.study_data.load_inputs",
        lambda *_: (pd.DataFrame(), {}),
    )
    monkeypatch.setattr(
        "odyn_sft.survey_evaluation.execution.execute",
        lambda *_: {"results": [], "notes": []},
    )
    root = Path(__file__).resolve().parents[1]
    config = replace(
        load_config(root / "configs/mistral-nf4.json"),
        source_suite=str(tmp_path),
        max_length=128,
        generation_max_new_tokens=5,
        validation_generation=True,
    )
    np.arange(1, 10, dtype=np.int32).tofile(tmp_path / "validation.tokens.bin")
    np.save(tmp_path / "validation.offsets.npy", np.array([0, 9]))
    np.save(tmp_path / "validation.prompts.npy", np.array([6]))
    dataset, model = TokenDataset(tmp_path, "validation"), tiny_model()
    stop, generation_inputs, loss_calls = [False], [], []

    def loss(*_args):
        loss_calls.append(1)
        return torch.tensor(2.0), 3, 1

    monkeypatch.setattr("odyn_sft.pytorch_sft.evaluation.evaluator.completion_loss", loss)

    def generate(**kwargs):
        generation_inputs.append(kwargs["input_ids"].tolist()[0])
        if len(generation_inputs) == 1:
            stop[0], token = True, 21
        else:
            token = 22
        return torch.cat([kwargs["input_ids"], torch.tensor([[token]])], dim=1)

    monkeypatch.setattr(model, "generate", generate)
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=22,
        decode=lambda *_args, **_kw: "analysis = pack()",
    )
    args = (
        model,
        tokenizer,
        dataset,
        config,
        torch.device("cpu"),
        tmp_path,
        "epoch-1",
        lambda: stop[0],
        "frozen",
    )
    assert evaluate(*args)["interrupted"]
    partial = read_json(tmp_path / "validation/epoch-1/partial-0000.json")
    assert partial["new_tokens"] == [21]
    stop[0] = False
    final = evaluate(*args)
    assert final["examples"] == 1 and final["generated_tokens"] == 2
    assert loss_calls == [1] and generation_inputs[1] == generation_inputs[0] + [21]
    assert not (tmp_path / "validation/epoch-1/partial-0000.json").exists()


def test_loss_only_validation_never_generates_and_resumes_cached_losses(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import math
    from dataclasses import replace

    from odyn_sft.pytorch_sft.data import TokenDataset
    from odyn_sft.pytorch_sft.evaluation.evaluator import evaluate

    np.arange(1, 19, dtype=np.int32).tofile(tmp_path/'validation.tokens.bin')
    np.save(tmp_path/'validation.offsets.npy', np.array([0, 9, 18]))
    np.save(tmp_path/'validation.prompts.npy', np.array([6, 5]))
    dataset, model = TokenDataset(tmp_path, 'validation'), tiny_model().eval()
    # Standard causal loss provides an independent reference for token weighting.
    expected = 0.0
    with torch.inference_mode():
        for row in (dataset[0], dataset[1]):
            expected += float(model(input_ids=row['input_ids'], labels=row['labels'],
                                    use_cache=False).loss)*row['target_tokens']
    expected /= 7
    def fail(*args, **kwargs):
        raise AssertionError('Loss-only evaluation must not generate or execute code')
    monkeypatch.setattr(model, 'generate', fail)
    monkeypatch.setattr('odyn_sft.survey_evaluation.execution.execute', fail)
    root = Path(__file__).resolve().parents[1]
    config = replace(load_config(root/'configs/mistral-nf4.json'),
                     validation_generation=False, max_length=18)
    args = (model, None, dataset, config, torch.device('cpu'), tmp_path,
            'epoch-1', lambda:False, 'frozen')
    first = evaluate(*args)
    assert first['loss'] == pytest.approx(expected, abs=1e-6)
    assert first['perplexity'] == pytest.approx(math.exp(expected), rel=1e-6)
    assert first['completion_tokens'] == 7 and first['examples'] == 2
    assert 'execution_valid_rate' not in first and 'syntax_valid_rate' not in first
    monkeypatch.setattr('odyn_sft.pytorch_sft.evaluation.evaluator.completion_loss', fail)
    assert evaluate(*args)['loss'] == first['loss']
    with pytest.raises(ValueError, match='cache.*mismatch'):
        evaluate(*args[:-1], 'changed')


def test_continuation_verifies_parent_and_rejects_training_changes(tmp_path):
    import json
    from dataclasses import replace

    from odyn_sft.common.storage import digest, file_hash, write_json
    from odyn_sft.pytorch_sft.training.continuation import continuation_checkpoint

    root = Path(__file__).resolve().parents[1]
    old = load_config(root/'configs/mistral-nf4.json')
    data = {'max_length':old.max_length, 'files':{'train.tokens.bin':'same'}}
    versions, scoring = {'torch':'fixed'}, {'api.py':'same'}
    before = {'training/objective.py':'same', 'training/loop.py':'before', 'evaluation/evaluator.py':'before'}
    after = {**before, 'training/loop.py':'after', 'evaluation/evaluator.py':'after', 'training/continuation.py':'new'}
    raw_config = old.as_dict()
    # These settings did not exist in the original parent manifest.
    del raw_config['continuation_from'], raw_config['validation_generation']
    fingerprint = digest({'config':raw_config, 'data':data, 'versions':versions,
                          'implementation':before, 'scoring_implementation':scoring})
    write_json(tmp_path/'run_manifest.json', {
        'config':raw_config, 'dataset':data, 'versions':versions,
        'implementation':before, 'scoring_implementation':scoring,
        'fingerprint':fingerprint})
    checkpoint = tmp_path/'checkpoint-000001-test'
    checkpoint.mkdir()
    (checkpoint/'training_state.pt').write_bytes(b'parent checkpoint')
    import torch
    from safetensors.torch import save_file
    save_file({'layer.lora_B.weight':torch.zeros(2, 2)}, str(checkpoint/'adapter_model.safetensors'))
    write_json(checkpoint/'complete.json', {
        'fingerprint':fingerprint, 'global_step':1, 'committed_at_ns':1,
        'files':{name:file_hash(checkpoint/name) for name in ('training_state.pt', 'adapter_model.safetensors')}})
    config = replace(old, continuation_from=str(tmp_path), output=str(tmp_path/'child'),
                     validation_generation=False, max_length=14263)
    changed_data = {**data, 'max_length':14263}
    path, provenance = continuation_checkpoint(config, changed_data, versions, after, scoring)
    assert path == checkpoint and provenance['optimizer_scheduler_rng_preserved']
    assert not provenance['validation_caches_reused']
    assert provenance['parent_checkpoint_is_base_only']
    with pytest.raises(ValueError, match='optimization settings'):
        continuation_checkpoint(replace(config, learning_rate=1e-4), changed_data,
                                versions, after, scoring)
    with pytest.raises(ValueError, match='tokenized data'):
        continuation_checkpoint(config, {**changed_data, 'files':{}}, versions, after, scoring)
    with pytest.raises(ValueError, match='loss, checkpoint'):
        continuation_checkpoint(config, changed_data, versions,
                                {**after, 'training/objective.py':'changed'}, scoring)
    torch.save({'state': {'epoch': 0, 'global_step': 1}}, checkpoint/'training_state.pt')
    marker = json.loads((checkpoint/'complete.json').read_text())
    marker['files']['training_state.pt'] = file_hash(checkpoint/'training_state.pt')
    write_json(checkpoint/'complete.json', marker)
    _, shortened = continuation_checkpoint(replace(config, epochs=1), changed_data, versions, after, scoring)
    assert shortened['scheduler_horizon_shortened']
    assert shortened['optimizer_rng_and_completed_updates_preserved']
    assert shortened['changed_settings']['epochs'] == {'before': old.epochs, 'after': 1}
    torch.save({'state': {'epoch': 1, 'global_step': 32}}, checkpoint/'training_state.pt')
    marker['files']['training_state.pt'] = file_hash(checkpoint/'training_state.pt')
    write_json(checkpoint/'complete.json', marker)
    with pytest.raises(ValueError, match='already complete'):
        continuation_checkpoint(replace(config, epochs=1), changed_data, versions, after, scoring)
    (checkpoint/'training_state.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt checkpoint'):
        continuation_checkpoint(config, changed_data, versions, after, scoring)


@pytest.mark.parametrize('total,warmup', [(12, 1), (12, 0), (100, 3), (1, 1)])
def test_first_optimizer_update_changes_adapter(total, warmup):
    import torch
    from peft import LoraConfig, get_peft_model

    from odyn_sft.pytorch_sft.training.adapters import assess_model, verify_first_update
    from odyn_sft.pytorch_sft.training.objective import completion_loss

    torch.manual_seed(71)
    model = get_peft_model(tiny_model(), LoraConfig(r=2, lora_alpha=4,
        target_modules='all-linear', task_type='CAUSAL_LM'))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:lr_multiplier(s, total, warmup))
    assert optimizer.param_groups[0]['lr'] > 0
    assert not assess_model(model)['has_learned_delta']
    with pytest.raises(RuntimeError, match='First optimizer update'):
        verify_first_update(model, optimizer.param_groups[0]['lr'])
    ids = torch.randint(0, 64, (1, 10))
    labels = ids.clone()
    labels[:, :6] = -100
    nll, count, _ = completion_loss(model, ids, labels)
    (nll/count).backward()
    optimizer.step()
    used_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    assert assess_model(model)['has_learned_delta']
    assert verify_first_update(model, used_lr)['passed']
    with pytest.raises(RuntimeError, match='First optimizer update'):
        verify_first_update(model, 0)


def test_encoding_supervises_eos_without_trailing_template_newline():
    from types import SimpleNamespace

    from odyn_sft.pytorch_sft.data import encode_row

    class Tokenizer:
        eos_token_id = 9
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if len(messages) == 1 else [1, 2, 7, 9, 198]
        def decode(self, tokens, **kwargs):
            return '\n' if tokens == [198] else ''
    result = encode_row({'instruction':'a','input':'b','output':'c'},
                                 Tokenizer(), SimpleNamespace(max_length=10, enable_thinking=False, system_prompt=None))
    assert result['input_ids'] == [1, 2, 7, 9]
    assert result['completion_mask'] == [0, 0, 1, 1]


def test_actual_batch_sizes_at_quarter_boundaries():
    windows = update_windows(10, 8)
    assert [end-start for start,end in windows] == [3, 2, 3, 2]
    assert len(windows)*3 == 12


@pytest.mark.parametrize('microbatch, accumulation, examples, updates, boundaries', [
    (1, 8, 10, 12, (3, 5, 8, 10)),
    (8, 2, 100, 24, (25, 50, 75, 100)),
])
def test_final_adapter_gate_rejects_incomplete_and_base_only_runs(
        tmp_path, microbatch, accumulation, examples, updates, boundaries):
    from dataclasses import replace

    import torch
    from safetensors.torch import save_file

    from odyn_sft.common.storage import file_hash, write_json
    from odyn_sft.pytorch_sft.evaluation.final import selected_adapter

    config = replace(load_config(Path(__file__).resolve().parents[1]/'configs/mistral-nf4.json'),
                     output=str(tmp_path), gradient_accumulation=accumulation, microbatch_size=microbatch)
    with pytest.raises(ValueError, match='incomplete'):
        selected_adapter(config)
    write_json(tmp_path/'run_manifest.json', {'config':config.as_dict(), 'fingerprint':'frozen',
                                            'effective':{'train_examples':examples}})
    summary = {'completed_max_epochs':False, 'epoch':0, 'global_step':2, 'planned_steps':updates}
    write_json(tmp_path/'training_summary.json', summary)
    with pytest.raises(ValueError, match='every requested epoch'):
        selected_adapter(config)
    summary.update(completed_max_epochs=True, epoch=3, global_step=updates, run_fingerprint='frozen',
                   first_update_has_learned_delta=True,
                   validated=[f'epoch-{epoch}-examples-{boundary:04d}'
                              for epoch in (1, 2, 3) for boundary in boundaries])
    write_json(tmp_path/'training_summary.json', summary)
    adapter = tmp_path/'best_adapter'
    adapter.mkdir()
    write_json(adapter/'adapter_config.json', {})
    def export(value):
        save_file({'layer.lora_B.weight':torch.full((2, 2), value)}, str(adapter/'adapter_model.safetensors'))
        write_json(adapter/'selection.json', {'training_complete':True, 'run_fingerprint':'frozen',
            'global_step':6, 'files':{n:file_hash(adapter/n) for n in ('adapter_model.safetensors','adapter_config.json')}})
    export(0.0)
    with pytest.raises(ValueError, match='base-only'):
        selected_adapter(config)
    export(0.01)
    path, selection = selected_adapter(config)
    assert path == adapter and selection['verified_adapter_assessment']['has_learned_delta']
    (adapter/'adapter_model.safetensors').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        selected_adapter(config)


def test_held_out_generation_uses_prompt_only_and_test_semantics(tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import pandas as pd
    import torch

    from odyn_sft.common.storage import read_json, write_jsonl
    from odyn_sft.pytorch_sft.data import TokenDataset
    from odyn_sft.pytorch_sft.evaluation.evaluator import evaluate

    config = replace(load_config(Path(__file__).resolve().parents[1]/'configs/mistral-nf4.json'),
                     source_suite=str(tmp_path), max_length=128, generation_max_new_tokens=5, validation_generation=True)
    write_jsonl(tmp_path/'private_gold.jsonl', [
        {'split':'validation','record_id':'val-only', 'analysis_contract':{'required_analyses':[]}},
        {'split':'test','record_id':'held-out', 'analysis_contract':{'required_analyses':['measured']}}])
    write_jsonl(tmp_path/'test_lineage.jsonl', [{'record_id':'held-out'}])
    monkeypatch.setattr('odyn_sft.tasks.sft1.read_json', lambda _: {})
    monkeypatch.setattr('odyn_sft.survey_evaluation.study_data.load_inputs', lambda *_:(pd.DataFrame(), {}))
    monkeypatch.setattr('odyn_sft.survey_evaluation.execution.execute', lambda *_:{'results':[], 'notes':[]})
    checked = []
    def semantics(contract, *args):
        checked.append(contract)
        return {'passed':True}
    monkeypatch.setattr('odyn_sft.survey_evaluation.execution.check_analysis_semantics', semantics)
    np.arange(1, 10, dtype=np.int32).tofile(tmp_path/'test.tokens.bin')
    np.save(tmp_path/'test.offsets.npy', np.array([0, 9]))
    np.save(tmp_path/'test.prompts.npy', np.array([6]))
    model = tiny_model()
    def generate(**kwargs):
        assert kwargs['input_ids'].tolist() == [[1, 2, 3, 4, 5, 6]]
        return torch.cat([kwargs['input_ids'], torch.tensor([[22]])], dim=1)
    monkeypatch.setattr(model, 'generate', generate)
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=22,
                                decode=lambda *_args, **_kwargs:'analysis = pack()')
    metrics = evaluate(model, tokenizer, TokenDataset(tmp_path, 'test'), config,
                       torch.device('cpu'), tmp_path, 'final', lambda:False, 'frozen', split='test')
    assert metrics['execution_valid_rate'] == metrics['measured_semantic_pass_rate'] == 1
    assert metrics['generated_outcomes'] == {
        'successful': 1, 'failed': 0, 'failure_types': {}, 'exception_types': {},
        'failure_subtypes': {}}
    assert metrics['generated_success_rate'] == 1
    assert checked == [{'required_analyses':['measured']}]
    record = read_json(tmp_path/'test/final/case-0000.json')
    assert record['record_id'] == 'held-out'
    assert not (tmp_path/'validation').exists()


def test_generated_outcomes_separate_success_and_failure_types():
    from odyn_sft.tasks.sft1 import generated_outcome

    case = {'analysis_contract': {'required_analyses': ['target']}}
    base = {'truncated': False, 'syntax_valid': True, 'execution_valid': True,
            'measured_semantics_pass': True, 'error': None}
    assert generated_outcome(base, case) == 'successful'
    assert generated_outcome({**base, 'truncated': True}, case) == 'truncated'
    assert generated_outcome({**base, 'syntax_valid': False,
                              'error': {'type': 'SyntaxError'}}, case) == 'syntax_error'
    assert generated_outcome({**base, 'syntax_valid': False, 'error': None}, case) == 'empty_or_invalid_output'
    assert generated_outcome({**base, 'execution_valid': False,
                              'error': {'type': 'AnalysisError'}}, case) == 'execution_error'
    assert generated_outcome({**base, 'measured_semantics_pass': False}, case) == 'semantic_failure'
    assert generated_outcome({**base, 'measured_semantics_pass': None}, case) == 'semantic_check_unavailable'


def test_legacy_schedule_cannot_silently_continue(tmp_path):
    from dataclasses import replace

    from odyn_sft.common.storage import digest, write_json
    from odyn_sft.pytorch_sft.training.continuation import continuation_checkpoint

    config = replace(load_config(Path(__file__).resolve().parents[1]/'configs/mistral-nf4.json'),
                     continuation_from=str(tmp_path))
    old = config.as_dict()
    del old['schedule_version']
    original = {'config':old, 'data':{}, 'versions':{}, 'implementation':{}, 'scoring_implementation':{}}
    write_json(tmp_path/'run_manifest.json', {**original, 'dataset':{}, 'fingerprint':digest(original)})
    with pytest.raises(ValueError, match='learning-rate schedule semantics'):
        continuation_checkpoint(config, {}, {}, {}, {})


def test_loss_comparison_explains_precision_and_example_scope():
    from odyn_sft.pytorch_sft.evaluation.evaluator import loss_precision_comparison

    selection = {'loss':1.5, 'validation_source_indices':[0, 1]}
    result = loss_precision_comparison(selection, {'loss':1.7}, [0, 1])
    assert result['selection_loss'] == 1.5 and result['final_validation_loss'] == 1.7
    assert result['final_minus_selection_loss'] == pytest.approx(0.2)
    assert result['selection_precision']['norm_weights'] == 'fp32'
    assert result['final_precision']['norm_weights'] == 'bf16'
    assert result['same_validation_examples'] is True
    scoped = loss_precision_comparison(selection, {'loss':1.7}, [0, 1, 2])
    assert scoped['same_validation_examples'] is False
    assert 'also reflects scope' in scoped['note']


@pytest.mark.parametrize('base_precision', ['NF4', 'bf16'])
def test_final_loss_report_retains_actual_base_precision(base_precision):
    from odyn_sft.pytorch_sft.evaluation.evaluator import loss_precision_comparison
    selection = {'loss': 1.5, 'validation_source_indices': [0],
                 'loss_precision': {'base_linear_weights': base_precision,
                                    'norm_weights': 'bf16', 'lora_weights': 'fp32'}}
    result = loss_precision_comparison(selection, {'loss': 1.5}, [0])
    assert result['final_precision']['base_linear_weights'] == base_precision
    assert result['final_precision']['lora_weights'] == 'bf16'


def test_training_acceptance_requires_real_first_update_and_all_12_points():
    from odyn_sft.pytorch_sft.config import training_acceptance

    tags = [f'epoch-{epoch}-examples-{boundary:04d}'
            for epoch in (1, 2, 3) for boundary in (3, 5, 8, 10)]
    state = {'validated':tags[:2], 'first_update_has_learned_delta':True}
    partial = training_acceptance(state, 10, 3)
    assert not partial['passed']
    assert partial['completed_validation_points'] == 2
    assert partial['expected_validation_points'] == 12
    assert len(partial['missing_validation_tags']) == 10
    state['validated'] = tags
    assert training_acceptance(state, 10, 3)['passed']
    state['first_update_has_learned_delta'] = False
    assert not training_acceptance(state, 10, 3)['passed']


def test_one_step_warmup_has_no_lr_ramp():
    assert [lr_multiplier(step, 12, 1) for step in (0, 1)] == [1, 1]
    assert 0 < lr_multiplier(2, 12, 1) < 1


def test_padded_microbatch_preserves_independent_loss_and_gradients():
    torch = pytest.importorskip('torch')
    from odyn_sft.pytorch_sft.data import collate_training
    from odyn_sft.pytorch_sft.training.objective import completion_loss
    torch.manual_seed(73)
    model = tiny_model().eval()
    reference = copy.deepcopy(model)
    records = []
    for length, prompt in [(13, 8), (9, 5), (6, 3)]:
        ids = torch.randint(1, 64, (1, length))
        labels = ids.clone()
        labels[:, :prompt] = -100
        records.append({'input_ids': ids, 'labels': labels, 'target_tokens':length-prompt})
    batch = collate_training(records, 0)
    loss, count, _ = completion_loss(model, batch['input_ids'], batch['labels'],
                                   attention_mask=batch['attention_mask'])
    expected = sum(completion_loss(reference, row['input_ids'], row['labels'])[0]
                   for row in records)
    assert count == batch['target_tokens'] == 12
    assert batch['attention_mask'].sum().item() == 28
    assert torch.allclose(loss, expected, atol=1e-5)
    (loss/count).backward()
    (expected/count).backward()
    for actual, target in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual.grad, target.grad, atol=1e-6, rtol=1e-4)


def test_shortened_continuation_reconciles_lr_without_resetting_optimizer():
    import torch
    from odyn_sft.pytorch_sft.training.continuation import reconcile_shortened_schedule
    from odyn_sft.pytorch_sft.config import lr_multiplier

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    old = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, 96, 3))
    parameter.sum().backward()
    optimizer.step()
    old.step()
    saved_optimizer, saved_scheduler = optimizer.state_dict(), old.state_dict()
    new = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, 32, 1))
    optimizer.load_state_dict(saved_optimizer)
    new.load_state_dict(saved_scheduler)
    moment = optimizer.state[parameter]['exp_avg'].clone()
    reconcile_shortened_schedule(optimizer, new)
    assert new.last_epoch == 1
    assert optimizer.param_groups[0]['lr'] == pytest.approx(2e-4 * lr_multiplier(1, 32, 1))
    assert torch.equal(moment, optimizer.state[parameter]['exp_avg'])
    assert new.get_last_lr() == [optimizer.param_groups[0]['lr']]
