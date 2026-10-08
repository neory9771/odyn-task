"""Judge paired saved responses against executed evidence; candidate identities stay hidden."""

from __future__ import annotations

import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..common.storage import (
    digest,
    file_hash,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ..providers.openai import ResponsesProvider
from . import synthesis_checks as checks
from . import synthesis_facts as facts_module
from . import synthesis_judge as judge_module


def validate_gate(report: dict[str, Any]) -> None:
    """Require control validation for the judge/version before submitting candidates."""
    judge = report.get("judge", {})
    if (
        judge.get("model") != judge_module.MODEL
        or judge.get("version") != judge_module.VERSION
    ):
        raise ValueError("Judge model/version differs from the validated gate")
    if report.get("prompt_sha256") is not None and report["prompt_sha256"] != digest(
        judge_module.INSTRUCTION
    ):
        raise ValueError("Judge prompt differs from the validated gate")
    required = {
        "flip_direction",
        "drop_contradicting",
        "inconclusive_as_support",
        "remove_gap",
        "append_verdict",
        "add_causal",
        "suppressed_value",
        "unavailable_comparison",
    }
    rates = report.get("detection", {})
    if not required <= set(rates) or any(
        rates[k]["n"] < 1 or rates[k]["rate"] < 0.9 for k in required
    ):
        raise ValueError("Judge planted-error detection gate failed")
    if (
        not report.get("gate_passed")
        or report["unmodified"]["item_failure_rate"] > 0.05
        or report["self_agreement"]["rate"] < 0.95
    ):
        raise ValueError("Judge validation gate failed")


def make_job(variant: str, index: int, record_id: str, synthesis_input: str, answer: str,
             generation_status: str) -> dict[str, Any]:
    """Facts and expectations come from the evidence inside the answer's own input."""
    payload = json.loads(synthesis_input)
    evidence = payload["api_response"]
    facts = facts_module.build_facts(evidence, payload["original_claim"])
    items = facts_module.build_expectations(facts)
    # The opaque request ID and variant remain local; the judge sees only facts/items/answer.
    request_id = digest({"variant": variant, "record_id": record_id, "answer": answer,
                         "facts": facts, "expectations": items})
    return {"request_id": request_id, "variant": variant, "source_index": index, "record_id": record_id,
            "facts": facts, "items": items, "answer": answer, "evidence": evidence,
            "generation_status": generation_status}


def build_jobs(source: Path, base: Path, sft: Path) -> list[dict[str, Any]]:
    """Require paired inference manifests and exact row identity before any paid call."""
    rows = read_jsonl(source)
    variants = [("base", base), ("sft2", sft)]
    manifests = [
        read_json(path.with_suffix(path.suffix + ".manifest.json"))
        for _, path in variants
    ]
    if manifests[0]["adapter"] is not None or manifests[1]["adapter"] is None:
        raise ValueError("Expected a base run and an adapter run")
    for field in (
        "input_sha256",
        "config",
        "limit",
        "max_new_tokens",
        "do_sample",
        "precision",
    ):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError("Paired inference settings differ: " + field)
    if manifests[0]["input_sha256"] != file_hash(source):
        raise ValueError("Source input differs from the inference source")
    jobs = []
    for (variant, path), manifest in zip(variants, manifests, strict=True):
        outputs = read_jsonl(path)
        if len(outputs) != manifest["examples"]:
            raise ValueError("Inference must finish all selected rows before judging")
        for index, candidate in enumerate(outputs):
            row = rows[index]
            if (
                candidate["source_index"] != index
                or candidate["record_id"] != row["metadata"]["record_id"]
            ):
                raise ValueError("Inference/source row identity mismatch")
            jobs.append(make_job(variant, index, candidate["record_id"], row["input"],
                                 candidate["completion"], candidate["status"]))
    random.Random(42).shuffle(jobs)
    return jobs


def run(
    source: Path,
    base: Path,
    sft: Path,
    output: Path,
    validation_report: Path,
    *,
    credentials_file: Path | None = None,
    workers: int = 4,
    dry_run: bool = False,
    request_cache: Path | None = None,
    provider_factory: Callable[..., Any] = ResponsesProvider,
) -> dict[str, Any]:
    """Judge paired base/SFT2 answers saved by odyn-infer on the same input file."""
    gate = read_json(validation_report)
    validate_gate(gate)
    jobs = build_jobs(source, base, sft)
    if dry_run:
        return {
            "dry_run": True,
            "examples": len(jobs),
            "calls": len(jobs),
            "judge": judge_module.MODEL,
            "gate_passed": True,
        }
    return judge_jobs(
        jobs,
        output,
        gate,
        [source, base, sft, validation_report],
        credentials_file=credentials_file,
        workers=workers,
        request_cache=request_cache,
        provider_factory=provider_factory,
    )


def judge_jobs(
    jobs: list[dict[str, Any]],
    output: Path,
    gate: dict[str, Any],
    files: list[Path],
    *,
    credentials_file: Path | None = None,
    workers: int = 4,
    request_cache: Path | None = None,
    provider_factory: Callable[..., Any] = ResponsesProvider,
) -> dict[str, Any]:
    """Grade prepared jobs (facts, expectations, answer); resumable per case.

    A job's variant names the answer source (e.g. base, sft2, e2e). The judge never sees it.
    """
    ledger_directory = request_cache or output / "ledgers"
    manifest = {
        "version": "paired-synthesis-evaluation-1",
        "judge_model": judge_module.MODEL,
        "judge_version": judge_module.VERSION,
        "request_cache": str(ledger_directory.resolve()),
        "maximum_completed_judgment_repairs": 1,
        "reasoning_effort": judge_module.REASONING_EFFORT,
        "judge_prompt_sha256": digest(judge_module.INSTRUCTION),
        "facts_version": facts_module.VERSION,
        "checks_version": checks.VERSION,
        "files": {
            str(p.resolve()): file_hash(p)
            for p in files
        },
        "implementation": {
            p.name: file_hash(p)
            for p in (
                Path(__file__),
                Path(facts_module.__file__),
                Path(checks.__file__),
                Path(judge_module.__file__),
            )
        },
        "jobs": [j["request_id"] for j in jobs],
        "candidate_identity_hidden_from_judge": True,
        "candidate_reference_answer_used": False,
        "calibration_scope": gate.get(
            "scope", "Automated control validation; no expert calibration"
        ),
    }
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.exists() or digest(read_json(manifest_path)) != digest(
            manifest
        ):
            raise ValueError("Existing judge run differs; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    write_json(output / "validation_gate.json", gate)

    def evaluate(job):
        request_id = job["request_id"]
        path = output / "cases" / (request_id + ".json")
        if path.exists():
            return read_json(path)
        deterministic = checks.check_answer(
            job["answer"], job["evidence"], job["facts"]
        )
        provider = provider_factory(
            judge_module.MODEL,
            ledger_directory / (request_id + ".json"),
            max_output_tokens=judge_module.MAX_OUTPUT_TOKENS,
            reasoning_effort=judge_module.REASONING_EFFORT,
            credentials_file=credentials_file,
        )
        attempts = []
        try:
            judged = judge_module.judge(
                provider, job["facts"], job["items"], job["answer"], request_id
            )
            attempts.append(judged)
            if not judged["valid"]:
                payload = json.loads(
                    judge_module.payload(job["facts"], job["items"], job["answer"])
                )
                payload["validation_feedback"] = {
                    "errors": judged["errors"],
                    "instruction": "Re-evaluate the same expectations. Copy quotations exactly from answer; do not invent quotations. No expected pass/fail decisions are provided.",
                }
                repaired = provider.complete(
                    judge_module.INSTRUCTION,
                    json.dumps(payload),
                    judge_module.SynthesisJudgement,
                    request_id + ":validation-repair-1",
                )
                judged = judge_module.validate(repaired, job["items"], job["answer"])
                attempts.append(judged)
            scored = judge_module.score(judged, job["items"], deterministic)
            critical = {i["id"] for i in job["items"] if i["critical"]}
            semantic_only = (
                not critical.intersection(scored["failed_items"])
                if judged["valid"]
                else None
            )
            error_type = None
        except Exception as exc:
            judged, scored, semantic_only = (
                None,
                {"judge_valid": False, "case_passed": None},
                None,
            )
            error_type = type(exc).__name__
        record = {
            "request_id": request_id,
            "variant": job["variant"],
            "source_index": job["source_index"],
            "record_id": job["record_id"],
            "generation_status": job["generation_status"],
            "facts": job["facts"],
            "expectations": job["items"],
            "checks": deterministic,
            "judged": judged,
            "judge_attempts": attempts,
            "score": scored,
            "semantic_only_pass": semantic_only,
            "combined_pass": (
                scored["case_passed"] and job["generation_status"] == "successful"
            )
            if scored["case_passed"] is not None
            else None,
            "error_type": error_type,
        }
        write_json(path, record)
        print(
            json.dumps(
                {
                    "variant": job["variant"],
                    "source_index": job["source_index"],
                    "judge_valid": scored["judge_valid"],
                    "semantic_pass": semantic_only,
                    "combined_pass": record["combined_pass"],
                }
            ),
            flush=True,
        )
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(evaluate, jobs))
    results.sort(key=lambda r: (r["variant"], r["source_index"]))
    write_jsonl(output / "results.jsonl", results)
    summary: dict[str, Any] = {
        "judge": manifest["judge_model"],
        "judge_version": manifest["judge_version"],
        "gate_passed": True,
        "expert_calibrated_on_current_pilot": False,
        "variants": {},
    }
    for variant in sorted({j["variant"] for j in jobs}):
        selected = [r for r in results if r["variant"] == variant]
        failures, criterion_counts = Counter(), Counter()
        for r in selected:
            if r["judged"] and r["judged"]["valid"]:
                for item in r["expectations"]:
                    criterion_counts[item["type"]] += 1
                    if not r["judged"]["results"][item["id"]]["passed"]:
                        failures[item["type"]] += 1
        summary["variants"][variant] = {
            "examples": len(selected),
            "valid_judgments": sum(r["score"]["judge_valid"] for r in selected),
            "invalid_or_error_judgments": sum(
                not r["score"]["judge_valid"] for r in selected
            ),
            "semantic_only_passed": sum(
                r["semantic_only_pass"] is True for r in selected
            ),
            "grounding_and_critical_expectations_passed": sum(
                r["score"]["case_passed"] is True for r in selected
            ),
            "combined_passed": sum(r["combined_pass"] is True for r in selected),
            "criterion_failures": dict(failures),
            "criterion_counts": dict(criterion_counts),
            "deterministic_check_failures": {
                name: sum(not r["checks"]["checks"][name]["passed"] for r in selected)
                for name in selected[0]["checks"]["checks"]
            },
        }
    summary["metric_definitions"] = {
        "semantic_only_passed": "Valid judge with every critical expectation passed",
        "grounding_and_critical_expectations_passed": "Semantic pass plus deterministic citation/numeric/suppression checks",
        "combined_passed": "Grounding/critical pass plus nonempty, untruncated generation",
        "invalid_or_error_judgments": "Unscored; do not treat as model successes or semantic failures",
    }
    write_json(output / "summary.json", summary)
    return summary
