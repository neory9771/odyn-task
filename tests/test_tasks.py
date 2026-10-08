"""Task boundary and shared SFT2 preparation/generation tests without network/GPU."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from odyn_sft.common.storage import file_hash, read_json, write_json, write_jsonl
from odyn_sft.pytorch_sft.config import load_config
from odyn_sft.pytorch_sft.data import (
    HeldOutDataset,
    TokenDataset,
    prepare,
    validate_prepared,
)
from odyn_sft.pytorch_sft.evaluation.evaluator import evaluate, evaluate_loss_only
from odyn_sft.tasks import get_task

ROOT = Path(__file__).resolve().parents[1]


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "offline task-neutral fixture"
    padding_side = "right"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        prefix = [1, 3 + len(next(m["content"] for m in messages if m["role"] == "user")) % 40]
        if add_generation_prompt:
            return prefix
        text = messages[-1]["content"]
        return prefix + [3 + ord(char) % 40 for char in text] + [2, 9]

    def decode(self, ids, **kwargs):
        if list(ids) == [9]:
            return "\n"
        return "The evidence supports one part of the claim; causation remains unidentified."

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "tokenizer.json").write_text("{}")


@pytest.fixture
def synthesis_suite(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    for split in ("train", "validation", "test"):
        lineage = {
            "record_id": f"{split}-1",
            "family_id": f"{split}-family",
            "component_keys": [f"{split}-atom"],
            "split": split,
        }
        row = {
            "instruction": "Explain the supplied evidence.",
            "input": f"Claim and evidence for {split}.",
            "output": "Evidence is inconclusive.",
            "metadata": lineage,
        }
        filename = f"{split}_synthesis" + ("_gold" if split == "test" else "") + ".jsonl"
        write_jsonl(suite / filename, [row])
        write_jsonl(suite / f"{split}_lineage.jsonl", [lineage])
        if split == "test":
            write_jsonl(
                suite / "test_synthesis_inputs.jsonl",
                [{k: row[k] for k in ("instruction", "input")}],
            )
    write_json(
        suite / "manifest.json",
        {
            "version": "synthesis-sft-1",
            "counts": {"train": 1, "validation": 1, "test": 1},
            "release_status": "draft",
            "files": {p.name: file_hash(p) for p in suite.iterdir()},
        },
    )
    return suite


def settings(tmp_path, suite, **kwargs):
    return replace(
        load_config(ROOT / "configs/mistral-sft2-bf16.json"),
        source_suite=str(suite),
        prepared=str(tmp_path / "prepared"),
        output=str(tmp_path / "train"),
        thermal_log=str(tmp_path / "gpu.jsonl"),
        max_length=128,
        generation_max_new_tokens=5,
        **kwargs,
    )


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.set_num_threads(1)
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
    )


def test_sft2_uses_same_preparation_masks_loss_and_held_out_loader(
    tmp_path, synthesis_suite, monkeypatch
):
    tokenizer = ToyTokenizer()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    config = settings(tmp_path, synthesis_suite)
    manifest = prepare(config)
    assert manifest["task"] == "sft2"
    assert manifest["splits"].keys() == {"train", "validation"}
    assert not (Path(config.prepared) / "test.tokens.bin").exists()
    assert validate_prepared(config) == manifest
    train = TokenDataset(Path(config.prepared), "train")[0]
    assert (train["labels"][0, : train["prompt_length"]] == -100).all()
    assert train["input_ids"][0, -1] == tokenizer.eos_token_id
    held_out = HeldOutDataset(config, tokenizer)[0]
    assert held_out["prompt_length"] == 2
    metrics = evaluate_loss_only(
        tiny_model(),
        TokenDataset(Path(config.prepared), "validation"),
        config,
        torch.device("cpu"),
        Path(config.output),
        "quarter",
        lambda: False,
        "frozen",
    )
    assert metrics["loss"] > 0 and metrics["completion_tokens"] > 0
    assert "syntax_valid_rate" not in metrics
    with pytest.raises(ValueError, match="Prepared task changed"):
        validate_prepared(replace(config, task="sft1"))


def test_sft2_generated_prose_is_not_executed_or_claimed_semantically_correct(
    tmp_path, synthesis_suite, monkeypatch
):
    tokenizer = ToyTokenizer()
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    config = settings(tmp_path, synthesis_suite, validation_generation=True)
    prepare(config)
    model = tiny_model()

    def generate(**kwargs):
        assert kwargs["input_ids"].shape[1] == 2  # Only prompt tokens, never reference prose.
        return torch.cat([kwargs["input_ids"], torch.tensor([[12, tokenizer.eos_token_id]])], dim=1)

    monkeypatch.setattr(model, "generate", generate)

    def forbidden(*args, **kwargs):
        pytest.fail("SFT2 must never execute response text as Python")

    monkeypatch.setattr("odyn_sft.survey_evaluation.execution.execute", forbidden)
    metrics = evaluate(
        model,
        tokenizer,
        TokenDataset(Path(config.prepared), "validation"),
        config,
        torch.device("cpu"),
        Path(config.output),
        "response",
        lambda: False,
        "frozen",
    )
    assert metrics["generated_output_rate"] == 1
    assert metrics["semantic_evaluation_status"] == "not_run_requires_calibrated_judge"
    assert "syntax_valid_rate" not in metrics and "generated_success_rate" not in metrics
    record = read_json(Path(config.output) / "validation/response/case-0000.json")
    assert record["response"] and record["semantic_judgment"] is None
    assert "program" not in record and "execution_valid" not in record
    # Cached generation is task-specific and does not generate a second time.
    monkeypatch.setattr(model, "generate", forbidden)
    assert (
        evaluate(
            model,
            tokenizer,
            TokenDataset(Path(config.prepared), "validation"),
            config,
            torch.device("cpu"),
            Path(config.output),
            "response",
            lambda: False,
            "frozen",
        )["examples"]
        == 1
    )


def test_sft2_suite_rejects_cross_split_identity_leakage(tmp_path, synthesis_suite):
    path = synthesis_suite / "validation_lineage.jsonl"
    lineage = json.loads(path.read_text())
    lineage["family_id"] = "train-family"
    write_jsonl(path, [lineage])
    rows_path = synthesis_suite / "validation_synthesis.jsonl"
    row = json.loads(rows_path.read_text())
    row["metadata"] = lineage
    write_jsonl(rows_path, [row])
    manifest = read_json(synthesis_suite / "manifest.json")
    for changed in (path, rows_path):
        manifest["files"][changed.name] = file_hash(changed)
    write_json(synthesis_suite / "manifest.json", manifest)
    with pytest.raises(ValueError, match="identity overlap"):
        get_task("sft2").validate_suite(synthesis_suite)


def test_training_core_imports_no_survey_execution_or_sft1_contracts():
    for path in (ROOT / "src/odyn_sft/pytorch_sft").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                from importlib.util import resolve_name

                package = "odyn_sft.pytorch_sft" + (
                    "." + ".".join(path.parent.relative_to(ROOT / "src/odyn_sft/pytorch_sft").parts)
                    if path.parent != ROOT / "src/odyn_sft/pytorch_sft"
                    else ""
                )
                module = (
                    resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else (node.module or "")
                )
                assert not module.startswith(
                    (
                        "odyn_sft.survey_evaluation",
                        "odyn_sft.survey_api",
                        "odyn_sft.tasks.sft1",
                    )
                )
    assert get_task("sft1").view == "analysis"
    assert get_task("sft2").view == "synthesis"
    with pytest.raises(ValueError, match="task must"):
        replace(load_config(ROOT / "configs/mistral-bf16.json"), task="unknown")


def test_sft2_failure_types_do_not_reuse_code_failures(synthesis_suite):
    scorer = get_task("sft2").scorer(synthesis_suite, "validation")
    case = scorer.case(0)
    for text, truncated in [("", False), ("unfinished", True)]:
        record = {**scorer.score(text, case), "truncated": truncated}
        scorer.accumulate(record, case)
    metrics = scorer.metrics(2)
    assert metrics["generated_outcomes"]["failure_types"] == {"truncated": 1, "empty_output": 1}
    assert metrics["semantic_evaluation_status"] == "not_run_requires_calibrated_judge"
