"""Local SFT2 assembly preserves gates, lineage and held-out draft provenance."""

from pathlib import Path

import pytest

from odyn_sft.common.storage import file_hash, read_jsonl, write_json, write_jsonl
from odyn_sft.dataset_generation.synthesis_suite import assemble


def inputs(tmp_path):
    targets, source = tmp_path / "targets", tmp_path / "source"
    targets.mkdir()
    source.mkdir()
    gates = []
    for split in ["train", "validation"]:
        item = {
            "record_id": split,
            "family_id": split + "-family",
            "study_id": split + "-study",
            "component_keys": [split + "-atom"],
            "split": split,
            "target_source": "writer_gated",
        }
        row = {
            "instruction": "Explain evidence",
            "input": "Claim and executed evidence",
            "output": "Support and limitations",
            "metadata": item,
        }
        write_jsonl(targets / f"{split}_synthesis.jsonl", [row])
        gates.append(
            {"record_id": split, "split": split, "accepted": True, "answer": row["output"]}
        )
    write_jsonl(targets / "targets.jsonl", gates)
    write_json(targets / "summary.json", {"accepted": 2})
    item = {
        "record_id": "test",
        "family_id": "test-family",
        "study_id": "client-study",
        "component_keys": ["test-atom"],
        "split": "test",
    }
    write_jsonl(
        source / "test_synthesis_gold.jsonl",
        [
            {
                "instruction": "Explain evidence",
                "input": "Held-out evidence",
                "output": "Draft",
                "metadata": item,
            }
        ],
    )
    write_jsonl(source / "test_lineage.jsonl", [item])
    write_json(
        source / "manifest.json", {"files": {p.name: file_hash(p) for p in source.iterdir()}}
    )
    return targets, source


def test_assembler_preserves_existing_targets_and_private_test_output(tmp_path):
    targets, source = inputs(tmp_path)
    output = tmp_path / "suite"
    result = assemble(targets, source, output, test_limit=1)
    assert result["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["llm_calls"] == 0 and not result["client_acceptance_certified"]
    assert read_jsonl(output / "train_synthesis.jsonl") == read_jsonl(
        targets / "train_synthesis.jsonl"
    )
    assert read_jsonl(output / "test_synthesis_inputs.jsonl") == [
        {"instruction": "Explain evidence", "input": "Held-out evidence"}
    ]
    assert result["target_policy"]["test"] == "source_draft_reference_not_writer_gated"
    with pytest.raises(ValueError, match="immutable"):
        assemble(targets, source, output, test_limit=1)


@pytest.mark.parametrize("problem", ["gate", "heldout_hash", "split_leakage"])
def test_assembler_rejects_invalid_targets_or_lineage_without_partial_package(tmp_path, problem):
    targets, source = inputs(tmp_path)
    if problem == "gate":
        rows = read_jsonl(targets / "targets.jsonl")
        rows[0]["accepted"] = False
        write_jsonl(targets / "targets.jsonl", rows)
    elif problem == "heldout_hash":
        (source / "test_synthesis_gold.jsonl").write_text("{}\n")
    else:
        rows = read_jsonl(targets / "validation_synthesis.jsonl")
        rows[0]["metadata"]["family_id"] = "train-family"
        write_jsonl(targets / "validation_synthesis.jsonl", rows)
    with pytest.raises(ValueError):
        assemble(targets, source, tmp_path / "suite", test_limit=1)
    assert not (tmp_path / "suite").exists()
    assert not list(tmp_path.glob("suite.staging-*"))


def test_exact_token_selection_keeps_whole_rows_and_preserves_test(tmp_path, monkeypatch):
    targets, source = inputs(tmp_path)
    rows = read_jsonl(targets / "train_synthesis.jsonl")
    long = {
        **rows[0],
        "input": "Long evidence " * 100,
        "metadata": {
            **rows[0]["metadata"],
            "record_id": "long",
            "family_id": "long-family",
            "study_id": "long-study",
            "component_keys": ["long-atom"],
        },
    }
    write_jsonl(targets / "train_synthesis.jsonl", rows + [long])
    gates = read_jsonl(targets / "targets.jsonl")
    gates.append(
        {"record_id": "long", "split": "train", "accepted": True, "answer": long["output"]}
    )
    write_jsonl(targets / "targets.jsonl", gates)

    class Tokenizer:
        eos_token_id = 9

        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            user = next(m["content"] for m in messages if m["role"] == "user")
            prefix = [1] * len(user)
            return (
                prefix
                if add_generation_prompt
                else prefix + [3] * len(messages[-1]["content"]) + [9]
            )

        def decode(self, tokens, **kwargs):
            return ""

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: Tokenizer())
    config = Path(__file__).parents[1] / "configs/qwen-sft2-local.json"
    result = assemble(
        targets,
        source,
        tmp_path / "suite",
        test_limit=1,
        token_config=config,
        max_example_tokens=200,
    )
    assert result["counts"] == {"train": 1, "validation": 1, "test": 1}
    assert result["token_filter"]["splits"]["train"]["excluded_record_ids"] == ["long"]
    assert result["token_filter"]["test_filtered"] is False
    assert read_jsonl(tmp_path / "suite/train_synthesis.jsonl") == rows
