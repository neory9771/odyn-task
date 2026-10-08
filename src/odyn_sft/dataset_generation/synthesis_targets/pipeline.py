"""Write gated SFT2 targets for the train/validation synthesis rows of a paired package.

Resumable: records already in targets.jsonl are skipped, and request ledgers never
replay a failed or ambiguous call. Output (consumed by `synthesis assemble`):
  targets.jsonl              every attempt and the accept/reject decision per record
  {split}_synthesis.jsonl    accepted rows: the package input, the written target as output
  summary.json               acceptance, rejection reasons and token usage
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...common.storage import read_json, read_jsonl, write_json, write_jsonl
from ...providers.openai import ResponsesProvider
from ...survey_evaluation import synthesis_judge as sj
from ...survey_evaluation.synthesis_facts import build_expectations, build_facts
from ..types import DataObject
from . import writer

SPLITS = ("train", "validation")


def load_records(package: Path) -> list[DataObject]:
    """Synthesis rows with their facts and expectations; split identity comes from lineage."""
    records = []
    for split in SPLITS:
        rows = read_jsonl(package / f"{split}_synthesis.jsonl")
        lineage = read_jsonl(package / f"{split}_lineage.jsonl")
        for row, item in zip(rows, lineage, strict=True):
            payload = json.loads(row["input"])
            facts = build_facts(payload["api_response"], payload["original_claim"])
            records.append({"record_id": item["record_id"], "split": split, "row": row,
                            "metadata": {**item, "split": split}, "facts": facts,
                            "items": build_expectations(facts), "evidence": payload["api_response"]})
    return records


def write_targets(package: Path, output: Path, *, workers: int = 8, credentials: Path | None = None) -> DataObject:
    records = load_records(package)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"package": str(package.resolve()),
                                        "package_manifest": read_json(package / "manifest.json").get("files", {}),
                                        "writer": {"model": writer.MODEL, "version": writer.VERSION,
                                                   "effort": writer.REASONING_EFFORT},
                                        "judge": {"model": sj.MODEL, "version": sj.VERSION}})
    path = output / "targets.jsonl"
    done = {r["record_id"] for r in read_jsonl(path)} if path.exists() else set()
    lock = threading.Lock()
    shards: dict[int, list[DataObject]] = defaultdict(list)
    for r in records:
        if r["record_id"] not in done:
            shards[int(hashlib.sha256(r["record_id"].encode()).hexdigest(), 16) % workers].append(r)

    def work(shard: int) -> None:
        base = output / "ledgers" / f"w{shard}"
        write = ResponsesProvider(writer.MODEL, base / "writer" / "ledger.json", max_output_tokens=writer.MAX_OUTPUT_TOKENS,
                                  reasoning_effort=writer.REASONING_EFFORT, timeout=writer.TIMEOUT_SECONDS,
                                  credentials_file=credentials)
        judge = ResponsesProvider(sj.MODEL, base / "judge" / "ledger.json", max_output_tokens=sj.MAX_OUTPUT_TOKENS,
                                  reasoning_effort=sj.REASONING_EFFORT, credentials_file=credentials)
        for r in shards[shard]:
            result = writer.generate_target(write, judge, r["record_id"], r["facts"], r["items"], r["evidence"])
            with lock, path.open("a") as handle:
                handle.write(json.dumps({"record_id": r["record_id"], "split": r["split"], **result}) + "\n")
            print(json.dumps({"record_id": r["record_id"], "accepted": result["accepted"],
                              "attempts": len(result["attempts"])}), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, sorted(shards)))
    return finish(output, records)


def finish(output: Path, records: list[DataObject]) -> DataObject:
    by_id = {r["record_id"]: r for r in records}
    results = [g for g in read_jsonl(output / "targets.jsonl") if g["record_id"] in by_id]
    accepted = [g for g in results if g["accepted"]]
    for split in SPLITS:
        rows = []
        for g in accepted:
            if g["split"] == split:
                source = by_id[g["record_id"]]
                rows.append({**source["row"], "output": g["answer"],
                             "metadata": {**source["metadata"], "target_source": "writer_gated",
                                          "target_writer": {"model": writer.MODEL, "version": writer.VERSION},
                                          "gate": {"judge_model": sj.MODEL, "judge_version": sj.VERSION}}})
        write_jsonl(output / f"{split}_synthesis.jsonl", rows)
    reasons: Counter[str] = Counter()
    for g in results:
        if g["accepted"]:
            continue
        last = g["attempts"][-1]
        if last.get("error"):
            reasons["error"] += 1
        elif not last["checks"]["passed"]:
            reasons.update(k for k, v in last["checks"]["checks"].items() if not v["passed"])
        elif last["judged"] and last["judged"]["valid"]:
            reasons.update(i for i, r in last["judged"]["results"].items() if not r["passed"])
        else:
            reasons["judge_invalid"] += 1
    usage: dict[str, Counter[str]] = defaultdict(Counter)
    for ledger in output.glob("ledgers/*/*/ledger.json"):
        for entry in read_json(ledger).values():
            u = entry.get("usage") or {}
            role = ledger.parent.name
            usage[role]["calls"] += 1
            usage[role]["input_tokens"] += u.get("input_tokens", 0)
            usage[role]["output_tokens"] += u.get("output_tokens", 0)
    summary = {"writer": {"model": writer.MODEL, "version": writer.VERSION},
               "judge": {"model": sj.MODEL, "version": sj.VERSION},
               "records": len(results), "accepted": len(accepted),
               "by_split": {s: {"n": sum(g["split"] == s for g in results),
                                "accepted": sum(g["split"] == s for g in accepted)} for s in SPLITS},
               "attempts_to_accept": dict(Counter(len(g["attempts"]) for g in accepted)),
               "rejection_reasons_last_attempt": dict(reasons),
               "usage": {k: dict(v) for k, v in usage.items()}}
    write_json(output / "summary.json", summary)
    return summary
