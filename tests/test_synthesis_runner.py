"""Judge gating, blinded payloads, validated quotes and API request replay protection."""

import json
from types import SimpleNamespace

import pytest
from test_synthesis_contract import CLAIM, GOOD, evidence

from odyn_sft.common.storage import file_hash, read_json, write_json, write_jsonl
from odyn_sft.providers.openai import ResponsesProvider
from odyn_sft.survey_evaluation.synthesis_judge import (
    ExpectationResult,
    SynthesisJudgement,
    validate,
)
from odyn_sft.survey_evaluation.synthesis_runner import run, validate_gate


def gate():
    return {
        "judge": {"model": "gpt-6-luna", "version": "synthesis-judge-1.0.4"},
        "gate_passed": True,
        "detection": {
            k: {"n": 40, "rate": 0.975}
            for k in (
                "flip_direction",
                "drop_contradicting",
                "inconclusive_as_support",
                "remove_gap",
                "append_verdict",
                "add_causal",
                "suppressed_value",
                "unavailable_comparison",
            )
        },
        "unmodified": {"item_failure_rate": 0.01},
        "self_agreement": {"rate": 0.99},
    }


def test_failed_gate_is_rejected():
    report = gate()
    report["self_agreement"]["rate"] = 0.8
    with pytest.raises(ValueError, match="validation gate failed"):
        validate_gate(report)


def test_quote_validation_rejects_invented_evidence():
    item = {"id": "E8"}
    judgement = SynthesisJudgement(
        results=[
            ExpectationResult(
                expectation_id="E8",
                passed=True,
                quote="never written",
                explanation="scope",
            )
        ]
    )
    assert not validate(judgement, [item], "Observed sample")["valid"]


def test_quotes_match_rendered_markdown_without_accepting_paraphrase():
    from odyn_sft.survey_evaluation.synthesis_judge import quote_found

    assert quote_found("The unweighted sample", "The **unweighted** sample")
    assert not quote_found("The weighted sample", "The **unweighted** sample")


@pytest.mark.parametrize("invalid_first", [False, True])
def test_blinded_judging_separates_semantics_from_truncation_and_resumes(
    tmp_path, invalid_first
):
    source, base, sft = [
        tmp_path / name for name in ("input.jsonl", "base.jsonl", "sft.jsonl")
    ]
    write_jsonl(
        source,
        [
            {
                "instruction": "interpret",
                "input": json.dumps(
                    {"api_response": evidence(), "original_claim": CLAIM}
                ),
                "metadata": {"record_id": "one"},
                "output": "SECRET TARGET",
            }
        ],
    )
    for p, status in ((base, "successful"), (sft, "truncated")):
        write_jsonl(
            p,
            [
                {
                    "source_index": 0,
                    "record_id": "one",
                    "completion": GOOD,
                    "status": status,
                }
            ],
        )
        write_json(
            p.with_suffix(".jsonl.manifest.json"),
            {
                "input_sha256": file_hash(source),
                "config": {},
                "limit": None,
                "max_new_tokens": 1536,
                "do_sample": False,
                "precision": {},
                "examples": 1,
                "adapter": None if p == base else "adapter",
            },
        )
    gate_path = tmp_path / "gate.json"
    write_json(gate_path, gate())
    calls = []

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, instruction, payload, schema, request_id):
            value = json.loads(payload)
            assert {"facts", "expectations", "answer"} <= set(value)
            assert set(value) <= {
                "facts",
                "expectations",
                "answer",
                "validation_feedback",
            }
            assert "SECRET TARGET" not in payload
            calls.append(request_id)
            return SynthesisJudgement(
                results=[
                    ExpectationResult(
                        expectation_id=i["id"],
                        passed=True,
                        quote=(
                            "invented quote"
                            if invalid_first
                            and not request_id.endswith(":validation-repair-1")
                            else "observed, unweighted survey results"
                        ),
                        explanation="matched",
                    )
                    for i in value["expectations"]
                ]
            )

    out = tmp_path / "judge"
    result = run(
        source, base, sft, out, gate_path, provider_factory=Provider, workers=2
    )
    assert len(calls) == (4 if invalid_first else 2)
    assert result["variants"]["base"]["combined_passed"] == 1
    assert result["variants"]["sft2"]["semantic_only_passed"] == 1
    assert result["variants"]["sft2"]["combined_passed"] == 0
    run(source, base, sft, out, gate_path, provider_factory=Provider)
    assert len(calls) == (4 if invalid_first else 2)


def test_provider_reuses_completed_calls_and_blocks_ambiguous_replay(
    tmp_path, monkeypatch
):
    count = []

    def parse(**kwargs):
        count.append(1)
        return SimpleNamespace(
            status="completed",
            output_parsed=SynthesisJudgement(results=[]),
            id="response",
            usage=None,
        )

    provider = ResponsesProvider("model", tmp_path / "ledger.json")
    monkeypatch.setattr(
        provider,
        "_client",
        lambda: SimpleNamespace(responses=SimpleNamespace(parse=parse)),
    )
    provider.complete("instruction", "payload", SynthesisJudgement, "id")
    provider.complete("instruction", "payload", SynthesisJudgement, "id")
    assert len(count) == 1
    with pytest.raises(ValueError, match="changed"):
        provider.complete("instruction", "changed", SynthesisJudgement, "id")
    value = read_json(provider.ledger)
    value["id"]["status"] = "started"
    write_json(provider.ledger, value)
    with pytest.raises(ValueError, match="reconcile"):
        provider.complete("instruction", "payload", SynthesisJudgement, "id")
    assert len(count) == 1
