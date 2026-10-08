"""End to end on held-out claims: SFT1 writes code, the API runs it, SFT2 interprets the result.

For each test claim of a paired package:
  1. analyse:   the SFT1 adapter generates a Survey API program from the claim, and the
                program is executed and scored against the gold contract (deterministic);
  2. interpret: if it ran, the SFT2 adapter writes a perspective from the claim, the program
                and its own evidence (never the gold evidence);
  3. judge:     the judge grades that perspective against facts built from the same evidence.

A case passes end to end only if step 1 is `successful` (right analysis) and step 3
passes (faithful interpretation). Both steps share one loaded base model. Each step
uses its adapter when one is given, otherwise the base model, so base/base, sft/sft
and mixed chains run through the same code.

  odyn-e2e --sft1-config C1 --sft1-adapter A1 --sft2-config C2 --sft2-adapter A2 \\
           --package PKG --out DIR --validation-report REPORT [--credentials-file F]
  odyn-e2e --sft1-config C1 --sft2-config C2 --package PKG --out DIR ...   # base/base

Judge requests are cached by content under --judge-cache (default .cache/judge), so
reruns and repeated answers are not paid for twice.

`--engine vllm --vllm-url URL` sends the same prompt tokens to a vLLM server (see
vllm_client.py) and runs `--concurrency` claims at once so the server can batch them.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from collections import Counter
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

from ..common.storage import digest, file_hash, read_json, read_jsonl, write_json
from ..prompts.survey import SYNTHESIS_INSTRUCTION, synthesis_input
from ..pytorch_sft.config import Config, load_config
from .runner import Engine, TemperatureStop, adapter_hashes, cast_lora_bf16, check_adapter, read_results

SFT1, SFT2 = "sft1", "sft2"
PACKAGE_FILES = ("test_analysis_inputs.jsonl", "test_lineage.jsonl", "metadata.json", "private_gold.jsonl")


class ChainEngine(Engine):
    """One base model; a step with an adapter switches to it, a step without one runs the base."""

    def __init__(self, configs: dict[str, Config], adapters: dict[str, Path], *,
                 max_temperature: float, disable_temperature_checks: bool):
        for key in ("model", "revision", "quantization", "fix_mistral_regex"):
            if getattr(configs[SFT1], key) != getattr(configs[SFT2], key):
                raise ValueError("SFT1 and SFT2 must share the base model settings: " + key)
        super().__init__(configs[SFT1], None, max_temperature, disable_temperature_checks)
        self.configs, self.adapters = configs, adapters
        if adapters:
            from peft import PeftModel

            (first, path), *rest = adapters.items()
            self.model = PeftModel.from_pretrained(self.model, path, adapter_name=first, is_trainable=False)
            for stage, path in rest:
                self.model.load_adapter(str(path), adapter_name=stage, is_trainable=False)
            cast_lora_bf16(self.model)
            self.model.requires_grad_(False)
            self.model.eval()

    def generate(self, stage: str, row: dict[str, Any]) -> dict[str, Any]:
        """Each step uses its own adapter (or the base), system prompt and generation budget."""
        config = self.configs[stage]
        if stage in self.adapters:
            self.model.set_adapter(stage)
            return self(row, config.generation_max_new_tokens, config)
        if self.adapters:
            with self.model.disable_adapter():
                return self(row, config.generation_max_new_tokens, config)
        return self(row, config.generation_max_new_tokens, config)


def variant_name(adapters: dict[str, Any]) -> str:
    """For example `sft1=sft,sft2=base`."""
    return ",".join(f"{stage}={'sft' if stage in adapters else 'base'}" for stage in (SFT1, SFT2))


def generate(engine: Any, stage: str, row: dict[str, Any]) -> dict[str, Any]:
    """A failed generation (e.g. prompt too long) becomes a recorded row, not a crashed run."""
    try:
        return engine.generate(stage, row)
    except TemperatureStop:
        raise
    except (ValueError, RuntimeError) as exc:
        return {"completion": "", "status": "runtime_error", "error": str(exc)}


def analyse(engine: Any, scorer: Any, row: dict[str, Any], case: dict[str, Any],
            lock: Any = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """SFT1: claim -> program -> executed and scored against the gold contract."""
    step = generate(engine, SFT1, row)
    with lock or nullcontext():  # The scorer holds shared study data and counters.
        scored = scorer.score(step["completion"], case)
        scored["truncated"] = step["status"] == "truncated"
        if step["status"] == "runtime_error":
            outcome, error = "runtime_error", {"type": "RuntimeError", "message": step["error"]}
        else:
            outcome, error = scorer.outcome(scored, case), scored["error"]
    return {**step, "program": scored["program"], "outcome": outcome, "error": error}, scored


def interpret(engine: Any, case: dict[str, Any], scored: dict[str, Any],
              metadata: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """SFT2: claim + the program SFT1 wrote + the evidence it produced -> perspective."""
    prompt = synthesis_input(case["claim"], scored["program"], scored["evidence"], metadata)
    return prompt, generate(engine, SFT2, {"instruction": SYNTHESIS_INSTRUCTION, "input": prompt})


def run(sft1_config: Path, sft2_config: Path, package: Path, output: Path, *,
        sft1_adapter: Path | None = None, sft2_adapter: Path | None = None, limit: int | None = None,
        max_temperature: float = 90, disable_temperature_checks: bool = False,
        engine: str = "hf", vllm_url: str | None = None, concurrency: int = 1,
        engine_factory: Callable[..., Any] | None = None,
        scorer_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Analyse then interpret each test claim; one JSON line per claim, flushed as it finishes.

    A step without an adapter runs the base model. Resume skips finished claims, and is
    refused if the engine, configs, adapters or package changed. With concurrency > 1,
    claims finish (and are written) out of order; each line carries its source_index.
    """
    if engine not in ("hf", "vllm"):
        raise ValueError("engine must be hf or vllm")
    if engine_factory is None and engine == "vllm":
        if not vllm_url:
            raise ValueError("--engine vllm requires --vllm-url")
        from .vllm_client import VllmEngine

        def engine_factory(*args: Any, **kwargs: Any) -> Any:
            return VllmEngine(*args, url=vllm_url, **kwargs)
    engine_factory = engine_factory or ChainEngine
    configs = {SFT1: load_config(sft1_config), SFT2: load_config(sft2_config)}
    adapters = {stage: path.resolve() for stage, path in ((SFT1, sft1_adapter), (SFT2, sft2_adapter))
                if path is not None}
    for stage, adapter in adapters.items():
        check_adapter(adapter, configs[stage])
    rows = read_jsonl(package / "test_analysis_inputs.jsonl")
    lineage = read_jsonl(package / "test_lineage.jsonl")
    if not rows or len(rows) != len(lineage):
        raise ValueError("Test inputs and lineage must be nonempty and aligned")
    rows, lineage = rows[:limit], lineage[:limit]
    manifest = {
        "version": "e2e-1",
        "engine": engine,
        "variant": variant_name(adapters),
        "package": {name: file_hash(package / name) for name in PACKAGE_FILES if (package / name).exists()},
        "configs": {stage: config.as_dict() for stage, config in configs.items()},
        "adapters": {stage: adapter_hashes(adapter) for stage, adapter in adapters.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    path, manifest_path = output / "e2e.jsonl", output / "e2e.manifest.json"
    if path.exists() and (not manifest_path.exists() or digest(read_json(manifest_path)) != digest(manifest)):
        raise ValueError("Cannot resume: configs/adapters/package changed; use a new --out")
    write_json(manifest_path, manifest)
    done = {r["record_id"] for r in read_results(path)}
    if scorer_factory is None:
        from ..tasks.sft1 import AnalysisScorer as scorer_factory
    scorer = scorer_factory(package, "test")
    metadata = read_json(package / "metadata.json")
    pending = []
    for index, item in enumerate(lineage):
        if item["record_id"] in done:
            continue
        case = scorer.case(index)
        if case["record_id"] != item["record_id"]:
            raise ValueError("Scorer/lineage record mismatch at index " + str(index))
        pending.append((index, rows[index], item, case))
    state, error = "completed", None
    score_lock, write_lock = threading.Lock(), threading.Lock()
    with path.open("a", encoding="utf-8") as handle:

        def one(index: int, row: dict[str, Any], item: dict[str, Any], case: dict[str, Any]) -> None:
            sft1, scored = analyse(chain, scorer, row, case, score_lock)
            prompt, sft2 = interpret(chain, case, scored, metadata) if scored["execution_valid"] else (None, None)
            record = {"source_index": index, "record_id": item["record_id"],
                      "claim_variant": item.get("claim_variant", "template"), "claim": case["claim"],
                      "sft1": sft1, "synthesis_input": prompt, "sft2": sft2}
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            with write_lock:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"record_id": item["record_id"], "sft1": sft1["outcome"],
                                  "sft2": sft2 and sft2["status"]}), flush=True)

        try:
            if pending:
                chain = engine_factory(configs, adapters, max_temperature=max_temperature,
                                       disable_temperature_checks=disable_temperature_checks)
                if concurrency > 1 and not getattr(chain, "concurrent", False):
                    raise ValueError("This engine generates one claim at a time; use --concurrency 1")
                pool = ThreadPoolExecutor(max_workers=concurrency)
                try:
                    futures = [pool.submit(one, *task) for task in pending]
                    finished, _ = wait(futures, return_when=FIRST_EXCEPTION)
                    failed = [f.exception() for f in finished if f.exception() is not None]
                    if failed:
                        raise failed[0]
                finally:
                    # Claims already in flight finish and are written; queued ones are dropped.
                    pool.shutdown(wait=True, cancel_futures=True)
        except (TemperatureStop, KeyboardInterrupt) as exc:
            state, error = "interrupted", str(exc) or "KeyboardInterrupt"
    generated = len(read_results(path))
    return {"state": state, "error": error, "generated": generated, "expected": len(rows), "output": str(path)}


def judge(output: Path, validation_report: Path, *, credentials_file: Path | None = None,
          workers: int = 4, cache: Path = Path(".cache/judge")) -> dict[str, Any]:
    """Grade every SFT2 answer against its own evidence, then combine with the SFT1 outcome."""
    from ..survey_evaluation.synthesis_runner import judge_jobs, make_job, validate_gate

    gate = read_json(validation_report)
    validate_gate(gate)
    records = read_results(output / "e2e.jsonl")
    jobs = [make_job("e2e", r["source_index"], r["record_id"], r["synthesis_input"], r["sft2"]["completion"],
                     r["sft2"]["status"]) for r in records if r["sft2"] is not None]
    summary = judge_jobs(jobs, output / "judge", gate, [output / "e2e.jsonl", validation_report],
                         credentials_file=credentials_file, workers=workers, request_cache=cache.resolve())
    judged = {r["record_id"]: r for r in read_results(output / "judge" / "results.jsonl")}

    def passed(r: dict[str, Any]) -> bool:
        verdict = judged.get(r["record_id"])
        return r["sft1"]["outcome"] == "successful" and bool(verdict and verdict["combined_pass"])

    def rates(selected: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(selected)
        return {"cases": n,
                "sft1_outcomes": dict(Counter(r["sft1"]["outcome"] for r in selected)),
                "sft1_successful": sum(r["sft1"]["outcome"] == "successful" for r in selected),
                "sft2_judged": sum(r["record_id"] in judged for r in selected),
                "sft2_judge_passed": sum(bool(judged.get(r["record_id"], {}).get("combined_pass")) for r in selected),
                "e2e_passed": sum(map(passed, selected)),
                "e2e_pass_rate": sum(map(passed, selected)) / n if n else None}

    result = {"definition": "E2E pass = SFT1 outcome successful (executes and meets the gold contract) "
                            "AND the SFT2 answer passes the judge plus deterministic checks on that evidence",
              "variant": read_json(output / "e2e.manifest.json")["variant"],
              "all": rates(records),
              "by_claim_variant": {v: rates([r for r in records if r["claim_variant"] == v])
                                   for v in sorted({r["claim_variant"] for r in records})},
              "judge_summary": summary}
    write_json(output / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="odyn-e2e", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft1-config", type=Path, required=True)
    parser.add_argument("--sft1-adapter", type=Path, help="Omit to run SFT1 on the base model")
    parser.add_argument("--sft2-config", type=Path, required=True)
    parser.add_argument("--sft2-adapter", type=Path, help="Omit to run SFT2 on the base model")
    parser.add_argument("--package", type=Path, required=True, help="Paired package with the held-out test split")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, help="Passing judge validation report")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--judge-cache", type=Path, default=Path(".cache/judge"),
                        help="Content-addressed judge request cache, shared across runs")
    parser.add_argument("--limit", type=int, help="First N test claims; default all")
    parser.add_argument("--workers", type=int, default=4, help="Parallel judge calls")
    parser.add_argument("--max-temperature", type=float, default=90.0, help="GPU stop temperature in C")
    parser.add_argument("--disable-temperature-checks", action="store_true")
    parser.add_argument("--skip-judge", action="store_true", help="Generate and execute only")
    parser.add_argument("--engine", choices=("hf", "vllm"), default="hf",
                        help="hf: in-process Transformers model; vllm: OpenAI-compatible vLLM server")
    parser.add_argument("--vllm-url", help="vLLM server base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--concurrency", type=int, default=1, help="Claims in flight at once (vllm only)")
    args = parser.parse_args(argv)
    if not args.skip_judge and not args.validation_report:
        parser.error("--validation-report is required unless --skip-judge")
    if args.engine == "vllm" and not args.vllm_url:
        parser.error("--engine vllm requires --vllm-url")
    if args.engine == "hf" and args.concurrency != 1:
        parser.error("--concurrency applies to --engine vllm only")
    for name in ("limit", "workers", "max_temperature", "concurrency"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    generated = run(args.sft1_config, args.sft2_config, args.package, args.out, sft1_adapter=args.sft1_adapter,
                    sft2_adapter=args.sft2_adapter, limit=args.limit, max_temperature=args.max_temperature,
                    disable_temperature_checks=args.disable_temperature_checks, engine=args.engine,
                    vllm_url=args.vllm_url, concurrency=args.concurrency)
    if args.skip_judge or generated["state"] != "completed":
        print(json.dumps(generated, indent=2))
        return 0 if generated["state"] == "completed" else 1
    result = judge(args.out, args.validation_report, credentials_file=args.credentials_file, workers=args.workers,
                   cache=args.judge_cache)
    print(json.dumps({k: result[k] for k in ("variant", "all", "by_claim_variant")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
