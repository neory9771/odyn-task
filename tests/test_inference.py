"""Inference preserves the chat contract, excludes gold and validates resumability."""

import json
from pathlib import Path

import pytest

from odyn_sft.common.storage import read_json, read_jsonl, write_json, write_jsonl
from odyn_sft.inference.runner import TemperatureStop, messages, run
from odyn_sft.pytorch_sft.config import load_config


def setup(tmp_path):
    profile = Path(__file__).parents[1] / "configs/qwen-sft2-local.json"
    # Installed-wheel tests receive the same external config schema.
    config = tmp_path / "config.json"
    value = json.loads(profile.read_text())
    value["workspace"] = "."
    write_json(config, value)
    rows = [
        {
            "instruction": "Explain",
            "input": "Evidence",
            "output": "SECRET GOLD",
            "metadata": {"record_id": str(i)},
        }
        for i in range(3)
    ]
    input_path = tmp_path / "inputs.jsonl"
    write_jsonl(input_path, rows)
    return config, input_path, tmp_path / "results.jsonl"


class FakeEngine:
    def __init__(self, *args):
        pass

    def __call__(self, row, budget):
        return {
            "completion": "Perspective",
            "status": "successful",
            "generated_tokens": 10,
            "seconds": 2,
            "finish_reason": "eos",
        }


def test_chat_contract_does_not_expose_gold(tmp_path):
    config, source, _ = setup(tmp_path)
    row = read_jsonl(source)[0]
    chat = messages(row, load_config(config))
    assert chat[0]["role"] == "system"
    assert chat[1]["content"] == "Explain\n\nEvidence"
    assert "SECRET GOLD" not in json.dumps(chat)


def test_limit_durable_outputs_and_safe_resume(tmp_path):
    config, source, output = setup(tmp_path)
    result = run(config, source, output, limit=2, engine_factory=FakeEngine)
    assert result["completed"] == 2
    assert result["generation_tokens_per_second"] == 5
    assert result["semantic_evaluation"] == "not_run"
    assert [r["record_id"] for r in read_jsonl(output)] == ["0", "1"]
    with pytest.raises(ValueError, match="Output exists"):
        run(config, source, output, limit=2, engine_factory=FakeEngine)
    # Completed resumes never load a model.
    assert (
        run(
            config,
            source,
            output,
            limit=2,
            resume=True,
            engine_factory=lambda *a: pytest.fail("unexpected reload"),
        )["completed"]
        == 2
    )
    with pytest.raises(ValueError, match="Cannot resume"):
        run(config, source, output, limit=3, resume=True, engine_factory=FakeEngine)


def test_temperature_stop_preserves_completed_rows_for_resume(tmp_path):
    config, source, output = setup(tmp_path)

    class Stops(FakeEngine):
        def __call__(self, row, budget):
            if row["metadata"]["record_id"] == "1":
                raise TemperatureStop("temperature")
            return super().__call__(row, budget)

    result = run(config, source, output, engine_factory=Stops)
    assert result["state"] == "interrupted"
    assert len(read_jsonl(output)) == 1
    result = run(config, source, output, resume=True, engine_factory=FakeEngine)
    assert result["state"] == "completed"
    assert len(read_jsonl(output)) == 3


def test_wrong_adapter_and_overwriting_input_are_rejected(tmp_path):
    config, source, output = setup(tmp_path)
    with pytest.raises(ValueError, match="differ"):
        run(config, source, source, engine_factory=FakeEngine)
    adapter = tmp_path / "adapter"
    write_json(adapter / "adapter_config.json", {"base_model_name_or_path": "wrong"})
    with pytest.raises(ValueError, match="does not match"):
        run(config, source, output, adapter=adapter, engine_factory=FakeEngine)


def test_load_failure_writes_summary(tmp_path):
    config, source, output = setup(tmp_path)

    def fail(*args):
        raise RuntimeError("no GPU")

    result = run(config, source, output, engine_factory=fail)
    assert result["state"] == "failed"
    assert result["error"] == "no GPU"
    assert output.with_suffix(".jsonl.summary.json").is_file()


def test_paired_comparison_checks_input_and_decoding_settings(tmp_path):
    from odyn_sft.inference.compare import compare

    config, source, base = setup(tmp_path)
    adapter = tmp_path / "adapter"
    write_json(
        adapter / "adapter_config.json", {"base_model_name_or_path": "Qwen/Qwen3-4B"}
    )
    sft = tmp_path / "sft.jsonl"
    run(config, source, base, limit=1, engine_factory=FakeEngine)
    run(config, source, sft, limit=1, adapter=adapter, engine_factory=FakeEngine)
    report = compare(base, sft, tmp_path / "comparison.json")
    assert report["examples"] == 1
    assert report["pairs"][0]["base"]["completion"] == "Perspective"
    assert report["semantic_evaluation"] == "not_run"
    manifest = sft.with_suffix(".jsonl.manifest.json")
    value = json.loads(manifest.read_text())
    value["max_new_tokens"] += 1
    write_json(manifest, value)
    with pytest.raises(ValueError, match="settings differ"):
        compare(base, sft, tmp_path / "invalid.json")


def test_resume_drops_a_partially_written_last_row(tmp_path):
    config, source, output = setup(tmp_path)
    run(config, source, output, limit=2, engine_factory=FakeEngine)
    with output.open("a") as handle:
        handle.write('{"source_index": 2, "comple')
    result = run(config, source, output, limit=2, resume=True, engine_factory=FakeEngine)
    assert result["completed"] == 2
    assert output.read_text().endswith("}\n")


def test_comparison_labels_the_adapter_by_task(tmp_path):
    from odyn_sft.inference.compare import compare

    config, source, base = setup(tmp_path)
    adapter = tmp_path / "adapter"
    write_json(adapter / "adapter_config.json", {"base_model_name_or_path": "Qwen/Qwen3-4B"})
    sft = tmp_path / "sft.jsonl"
    run(config, source, base, limit=1, engine_factory=FakeEngine)
    run(config, source, sft, limit=1, adapter=adapter, engine_factory=FakeEngine)
    assert read_jsonl(sft)[0]["model_variant"] == "sft2"
    with pytest.raises(ValueError, match="new comparison"):
        compare(base, sft, source)


class FakeScorer:
    def __init__(self, package, split):
        self.lineage = read_jsonl(package / f"{split}_lineage.jsonl")

    def case(self, index):
        return {"record_id": self.lineage[index]["record_id"], "claim": "Claim " + str(index)}

    def score(self, completion, case):
        ran = completion == "program"
        return {"program": completion, "execution_valid": ran, "evidence": {"x": 1} if ran else None,
                "error": None if ran else {"type": "SyntaxError", "message": "bad"}}

    def outcome(self, scored, case):
        return "successful" if scored["execution_valid"] else "syntax_error"


class FakeChain:
    def __init__(self, configs, adapters, **thermal):
        self.adapters, self.calls = adapters, []

    def generate(self, stage, row):
        self.calls.append(stage)
        if row.get("input") == "too long":
            raise ValueError("Prompt length leaves no room")
        return {"completion": "program" if stage == "sft1" else "Perspective", "status": "successful"}


def e2e_setup(tmp_path, monkeypatch, inputs=("claim", "claim", "too long")):
    from odyn_sft.inference import e2e

    monkeypatch.setattr(e2e, "synthesis_input", lambda claim, program, evidence, metadata: claim + "|" + program)
    config, _, _ = setup(tmp_path)
    package = tmp_path / "package"
    write_jsonl(package / "test_analysis_inputs.jsonl", [{"instruction": "Write", "input": i} for i in inputs])
    write_jsonl(package / "test_lineage.jsonl", [{"record_id": f"r{i}"} for i in range(len(inputs))])
    write_json(package / "metadata.json", {})
    adapters = {}
    for stage in ("sft1", "sft2"):
        adapters[stage] = tmp_path / stage
        write_json(adapters[stage] / "adapter_config.json", {"base_model_name_or_path": "Qwen/Qwen3-4B"})
    return e2e, config, package, adapters


def test_e2e_runs_sft1_then_sft2_and_records_failures(tmp_path, monkeypatch):
    e2e, config, package, adapters = e2e_setup(tmp_path, monkeypatch)
    engines = []

    def factory(*args, **kwargs):
        engines.append(FakeChain(*args, **kwargs))
        return engines[-1]

    result = e2e.run(config, config, package, tmp_path / "out", sft1_adapter=adapters["sft1"],
                     sft2_adapter=adapters["sft2"], engine_factory=factory, scorer_factory=FakeScorer)
    assert result["state"] == "completed" and result["generated"] == 3
    assert engines[0].calls == ["sft1", "sft2", "sft1", "sft2", "sft1"]
    records = read_jsonl(tmp_path / "out/e2e.jsonl")
    assert records[0]["synthesis_input"] == "Claim 0|program"
    assert records[0]["sft2"]["completion"] == "Perspective"
    assert records[2]["sft1"]["outcome"] == "runtime_error" and records[2]["sft2"] is None
    # Completed claims are skipped on resume; changed settings are refused.
    assert e2e.run(config, config, package, tmp_path / "out", sft1_adapter=adapters["sft1"],
                   sft2_adapter=adapters["sft2"], engine_factory=lambda *a, **k: pytest.fail("reload"),
                   scorer_factory=FakeScorer)["generated"] == 3
    with pytest.raises(ValueError, match="Cannot resume"):
        e2e.run(config, config, package, tmp_path / "out", engine_factory=factory, scorer_factory=FakeScorer)


def test_e2e_each_step_is_base_or_sft_and_inputs_align(tmp_path, monkeypatch):
    e2e, config, package, adapters = e2e_setup(tmp_path, monkeypatch, inputs=("claim",))
    seen = []

    def factory(configs, adapters, **thermal):
        seen.append(set(adapters))
        return FakeChain(configs, adapters)

    e2e.run(config, config, package, tmp_path / "base", engine_factory=factory, scorer_factory=FakeScorer)
    e2e.run(config, config, package, tmp_path / "mixed", sft1_adapter=adapters["sft1"],
            engine_factory=factory, scorer_factory=FakeScorer)
    assert seen == [set(), {"sft1"}]
    assert read_json(tmp_path / "base/e2e.manifest.json")["variant"] == "sft1=base,sft2=base"
    assert read_json(tmp_path / "mixed/e2e.manifest.json")["variant"] == "sft1=sft,sft2=base"
    write_jsonl(package / "test_lineage.jsonl", [{"record_id": "a"}, {"record_id": "b"}])
    with pytest.raises(ValueError, match="aligned"):
        e2e.run(config, config, package, tmp_path / "y", engine_factory=FakeChain, scorer_factory=FakeScorer)


def test_e2e_cli_requires_a_validation_report_to_judge():
    from odyn_sft.inference.e2e import main

    with pytest.raises(SystemExit):
        main(["--sft1-config", "c", "--sft2-config", "c", "--package", "p", "--out", "o"])
    with pytest.raises(SystemExit):
        main(["--sft1-config", "c", "--sft2-config", "c", "--package", "p", "--out", "o", "--skip-judge",
              "--limit", "0"])


class FakeServer:
    """Minimal OpenAI client double for a vLLM server."""

    def __init__(self, served, finish="stop"):
        from types import SimpleNamespace as NS

        self.requests, self.finish, self.NS = [], finish, NS
        self.models = NS(list=lambda: NS(data=[NS(id=name, root=root) for name, root in served.items()]))
        self.completions = NS(create=self.create)

    def create(self, **request):
        self.requests.append(request)
        NS = self.NS
        return NS(choices=[NS(text="Perspective", finish_reason=self.finish)], usage=NS(completion_tokens=7))


class FakeTokenizer:
    def apply_chat_template(self, chat, **kwargs):
        return {"input_ids": list(range(len(json.dumps(chat)) // 10))}


def vllm_engine(tmp_path, served, adapters, finish="stop"):
    from odyn_sft.inference.vllm_client import VllmEngine

    config, _, _ = setup(tmp_path)
    configs = {"sft1": load_config(config), "sft2": load_config(config)}
    server = FakeServer(served, finish)
    engine = VllmEngine(configs, adapters, url="http://x/v1", disable_temperature_checks=True,
                        client=server, tokenizer=FakeTokenizer())
    return engine, server


def test_vllm_engine_routes_steps_to_lora_or_base_greedily(tmp_path):
    adapter = tmp_path / "a1"
    engine, server = vllm_engine(tmp_path, {"Qwen/Qwen3-4B": None, "sft1": str(adapter)}, {"sft1": adapter})
    row = {"instruction": "Write", "input": "Claim"}
    first = engine.generate("sft1", row)
    engine.generate("sft2", row)
    assert [r["model"] for r in server.requests] == ["sft1", "Qwen/Qwen3-4B"]
    assert all(r["temperature"] == 0 and isinstance(r["prompt"][0], int) for r in server.requests)
    assert first["status"] == "successful" and first["generated_tokens"] == 7
    assert first["prompt_tokens"] == len(server.requests[0]["prompt"])
    engine, _ = vllm_engine(tmp_path, {"Qwen/Qwen3-4B": None}, {}, finish="length")
    assert engine.generate("sft2", row)["status"] == "truncated"


def test_vllm_engine_rejects_a_server_with_other_adapters(tmp_path):
    with pytest.raises(ValueError, match="LoRA module 'sft1'"):
        vllm_engine(tmp_path, {"Qwen/Qwen3-4B": None, "sft1": "/elsewhere"}, {"sft1": tmp_path / "a1"})
    with pytest.raises(ValueError, match="does not serve"):
        vllm_engine(tmp_path, {"other": None}, {})


def test_e2e_concurrent_claims_are_all_written_once(tmp_path, monkeypatch):
    e2e, config, package, adapters = e2e_setup(tmp_path, monkeypatch, inputs=("claim",) * 12)

    class Concurrent(FakeChain):
        concurrent = True

    result = e2e.run(config, config, package, tmp_path / "out", engine_factory=Concurrent,
                     scorer_factory=FakeScorer, concurrency=4)
    records = read_jsonl(tmp_path / "out/e2e.jsonl")
    assert result["generated"] == 12
    assert sorted(r["source_index"] for r in records) == list(range(12))
    assert all(r["record_id"] == f"r{r['source_index']}" for r in records)
    with pytest.raises(ValueError, match="one claim at a time"):
        e2e.run(config, config, package, tmp_path / "hf", engine_factory=FakeChain,
                scorer_factory=FakeScorer, concurrency=2)
