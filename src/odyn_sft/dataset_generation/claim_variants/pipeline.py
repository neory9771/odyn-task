"""Gate the verifier, write one paraphrase per claim, and apply them to a paired package.

Steps (each resumable; results are appended to JSONL as they finish):
  validate_verifier  planted meaning changes and untouched claims; the gate must pass
  generate           one writer call plus one verifier call per claim
  apply              a new package: every template row, plus one paraphrase row per accepted claim

A paraphrase row shares its template's family, split, program, evidence and gold. Only
the claim text, and therefore the model inputs built from it, differ. The same
paraphrase feeds both the analysis (SFT1) input and the synthesis (SFT2) input.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...common.storage import file_hash, read_json, read_jsonl, write_json, write_jsonl
from ...prompts.survey import (
    ANALYSIS_INSTRUCTION,
    SYNTHESIS_INSTRUCTION,
    analysis_input,
    synthesis_input,
)
from ...providers.openai import ResponsesProvider
from ...survey_evaluation.package_integrity import SPLITS, validate_splits_package
from ...survey_metadata.catalogue import numeric_program, numeric_references
from ..types import DataObject
from . import paraphrase as cp
from .planted import PERTURBATIONS, perturb
from .signature import VERSION as SIGNATURE_VERSION
from .signature import SignatureError, signature

GATE = {"rejection": 0.95, "acceptance": 0.95, "self_agreement": 0.95}
SUFFIX = ":p1"


def load_records(package: Path) -> tuple[list[DataObject], list[DataObject]]:
    """Template claims with their meaning signature; claims whose program cannot be parsed are skipped."""
    variables = read_json(package / "metadata.json")["variables"]
    records, skipped = [], []
    for gold in read_jsonl(package / "private_gold.jsonl"):
        if gold.get("claim_variant", "template") != "template":
            continue
        try:
            sig = signature(gold["program"], variables, gold["claim"])
        except (SignatureError, KeyError, ValueError, AttributeError) as exc:
            skipped.append({"record_id": gold["record_id"], "reason": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        records.append({"record_id": gold["record_id"], "split": gold["split"], "use_case": gold["scenario"],
                        "claim": gold["claim"], "signature": sig})
    return records, skipped


def _providers(base: Path, credentials: Path | None) -> tuple[ResponsesProvider, ResponsesProvider]:
    def make(role: str) -> ResponsesProvider:
        return ResponsesProvider(cp.MODEL, base / role / "ledger.json", max_output_tokens=cp.MAX_OUTPUT_TOKENS,
                                 reasoning_effort=cp.REASONING_EFFORT, credentials_file=credentials)
    return make("writer"), make("verifier")


def _shards(jobs: list[DataObject], key: str, workers: int) -> dict[int, list[DataObject]]:
    shards: dict[int, list[DataObject]] = defaultdict(list)
    for job in jobs:
        shards[int(hashlib.sha256(job[key].encode()).hexdigest(), 16) % workers].append(job)
    return shards


def _append(path: Path, row: DataObject, lock: threading.Lock) -> None:
    with lock, path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _records_hash(records: list[DataObject]) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def _accepted(row: DataObject) -> bool:
    verified = row.get("verified")
    return bool(verified and verified["valid"] and all(x["passed"] for x in verified["results"].values()))


def validate_verifier(records: list[DataObject], variables: list[DataObject], work: Path, *, originals: int, per_type: int, workers: int,
                      seed: int, credentials: Path | None = None) -> DataObject:
    """The verifier must accept untouched claims, reject planted changes and agree with itself."""
    pool = [r for r in records if r["split"] != "test"]  # Held-out claims are never used for tuning.
    jobs = []
    for r in random.Random(seed).sample(pool, min(originals, len(pool))):
        for run in (1, 2):
            jobs.append({"id": f"{r['record_id']}:original:run{run}", "kind": "original", "run": run,
                         "record": r, "claim": r["claim"]})
    for kind in PERTURBATIONS:
        order = pool[:]
        random.Random(f"{seed}:{kind}").shuffle(order)
        taken = 0
        for r in order:
            if taken == per_type:
                break
            changed = perturb(kind, r["claim"], r["signature"], variables, random.Random(f"{seed}:{kind}:{r['record_id']}"))
            if changed:
                jobs.append({"id": f"{r['record_id']}:{kind}:run1", "kind": kind, "run": 1, "record": r, "claim": changed})
                taken += 1
    path = work / "verifier_validation.jsonl"
    done = {row["id"] for row in read_jsonl(path)} if path.exists() else set()
    lock = threading.Lock()
    shards = _shards([j for j in jobs if j["id"] not in done], "id", workers)

    def worker(shard: int) -> None:
        _, verifier = _providers(work / "ledgers" / "verifier-validation" / f"w{shard}", credentials)
        for job in shards[shard]:
            row = {k: job[k] for k in ("id", "kind", "run", "claim")} | {"record_id": job["record"]["record_id"]}
            try:
                row["verified"] = cp.verify(verifier, job["record"]["signature"], job["claim"], job["id"])
            except Exception as exc:  # The ledger keeps the failed call; it is never silently replayed.
                row["verified"], row["error"] = None, f"{type(exc).__name__}: {exc}"[:300]
            _append(path, row, lock)

    work.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool_executor:
        list(pool_executor.map(worker, sorted(shards)))
    ids = {j["id"] for j in jobs}
    rows = list({r["id"]: r for r in read_jsonl(path) if r["id"] in ids}.values())
    untouched = [r for r in rows if r["kind"] == "original" and r["run"] == 1]
    by_kind = {k: [r for r in rows if r["kind"] == k] for k in PERTURBATIONS}
    runs: dict[str, dict[int, DataObject]] = defaultdict(dict)
    for r in rows:
        if r["kind"] == "original" and r.get("verified") and r["verified"]["valid"]:
            runs[r["record_id"]][r["run"]] = r
    agree = total = 0
    for pair in runs.values():
        if 1 in pair and 2 in pair:
            for item, result in pair[1]["verified"]["results"].items():
                total += 1
                agree += result["passed"] == pair[2]["verified"]["results"][item]["passed"]
    summary = {"verifier": {"model": cp.MODEL, "version": cp.VERIFIER_VERSION},
               "signature_version": SIGNATURE_VERSION, "records_sha256": _records_hash(records),
               "calls": len(rows), "errors": sum(bool(r.get("error")) for r in rows),
               "originals": {"n": len(untouched), "accepted": sum(map(_accepted, untouched))},
               "rejection": {k: {"n": len(v), "rejected": sum(not _accepted(r) for r in v)} for k, v in by_kind.items()},
               "self_agreement": {"items": total, "rate": agree / total if total else None},
               "false_rejections": [{"claim": r["claim"][:200],
                                     "failed": [i for i, x in r["verified"]["results"].items() if not x["passed"]]}
                                    for r in untouched if r.get("verified") and not _accepted(r)][:10],
               "missed": [{"kind": r["kind"], "claim": r["claim"][:200]}
                          for v in by_kind.values() for r in v if _accepted(r)][:10]}
    acceptance = summary["originals"]["accepted"] / max(1, summary["originals"]["n"])
    # A change type with no applicable claim in this package cannot be tested; it does not fail the gate.
    summary["gate"] = {"acceptance": acceptance >= GATE["acceptance"],
                       "self_agreement": (summary["self_agreement"]["rate"] or 0) >= GATE["self_agreement"],
                       "rejection": {k: v["rejected"] / v["n"] >= GATE["rejection"]
                                     for k, v in summary["rejection"].items() if v["n"]}}
    summary["untested_change_types"] = [k for k, v in summary["rejection"].items() if not v["n"]]
    summary["gate_passed"] = (len(rows) == len(jobs) and not summary["errors"] and summary["gate"]["acceptance"]
                              and summary["gate"]["self_agreement"] and all(summary["gate"]["rejection"].values()))
    write_json(work / "verifier_validation_summary.json", summary)
    return summary


def require_gate(records: list[DataObject], work: Path) -> None:
    """A passing gate must cover these exact records and the current verifier."""
    path = work / "verifier_validation_summary.json"
    if not path.exists():
        raise ValueError("Run validate-verifier first")
    gate = read_json(path)
    if (not gate.get("gate_passed")
            or gate.get("verifier") != {"model": cp.MODEL, "version": cp.VERIFIER_VERSION}
            or gate.get("signature_version") != SIGNATURE_VERSION
            or gate.get("records_sha256") != _records_hash(records)):
        raise ValueError("Verifier gate failed or is stale; rerun validate-verifier for this package")


def generate(records: list[DataObject], work: Path, *, workers: int, max_attempts: int = 1,
             credentials: Path | None = None) -> DataObject:
    """One paraphrase per claim; train/validation use T cards, test uses test-only S cards."""
    require_gate(records, work)
    path = work / "variants.jsonl"
    done = {row["record_id"] for row in read_jsonl(path)} if path.exists() else set()
    lock = threading.Lock()
    shards = _shards([r for r in records if r["record_id"] not in done], "record_id", workers)

    def worker(shard: int) -> None:
        writer, verifier = _providers(work / "ledgers" / "generate" / f"w{shard}", credentials)
        for r in shards[shard]:
            result = cp.paraphrase(writer, verifier, r["record_id"], r["claim"], r["signature"],
                                   cp.card_for(r["record_id"], r["split"]), max_attempts=max_attempts)
            _append(path, {"record_id": r["record_id"], "split": r["split"], "use_case": r["use_case"],
                           "original": r["claim"], **result}, lock)
            print(json.dumps({"record_id": r["record_id"], "accepted": result["accepted"]}), flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(worker, sorted(shards)))
    ids = {r["record_id"] for r in records}
    rows = [v for v in read_jsonl(path) if v["record_id"] in ids]
    by_split: dict[str, list[DataObject]] = defaultdict(list)
    for v in rows:
        by_split[v["split"]].append(v)
    summary = {"records": len(rows), "accepted": sum(v["accepted"] for v in rows),
               "by_split": {s: {"n": len(v), "accepted": sum(x["accepted"] for x in v)} for s, v in by_split.items()},
               "by_card": dict(Counter(v["card"] for v in rows if v["accepted"]))}
    write_json(work / "generate_summary.json", summary)
    return summary


def apply(source: Path, work: Path, output: Path) -> DataObject:
    """Write a new package with template rows plus one paraphrase row after each accepted claim."""
    if output.exists():
        raise ValueError("Use a new output directory; packages are immutable")
    records, _ = load_records(source)
    require_gate(records, work)
    accepted = {v["record_id"]: v for v in read_jsonl(work / "variants.jsonl") if v["accepted"]}
    metadata = read_json(source / "metadata.json")
    references = numeric_references(metadata["variables"])
    gold = read_jsonl(source / "private_gold.jsonl")
    shutil.copytree(source, output)
    expanded: list[DataObject] = []
    for case in gold:
        expanded.append({**case, "claim_variant": "template"})
        variant = accepted.get(case["record_id"])
        if variant:
            if variant["original"] != case["claim"]:
                raise ValueError("Claim changed since paraphrasing: " + case["record_id"])
            expanded.append({**case, "record_id": case["record_id"] + SUFFIX, "claim": variant["claim"],
                             "claim_variant": "paraphrase", "variant_of": case["record_id"], "claim_card": variant["card"]})
    write_jsonl(output / "private_gold.jsonl", expanded)
    lineage_keys = ("record_id", "family_id", "component_keys", "variant", "claim_variant", "variant_of")
    for split in SPLITS:
        rows = [c for c in expanded if c["split"] == split]
        suffix = "_gold" if split == "test" else ""
        analysis = [{"instruction": ANALYSIS_INSTRUCTION, "input": analysis_input(c["claim"], metadata),
                     "output": numeric_program(c["program"], references)} for c in rows]
        synthesis = [{"instruction": SYNTHESIS_INSTRUCTION,
                      "input": synthesis_input(c["claim"], c["program"], c["evidence"], metadata),
                      "output": c["reference_perspective"]} for c in rows]
        write_jsonl(output / f"{split}_analysis{suffix}.jsonl", analysis)
        write_jsonl(output / f"{split}_synthesis{suffix}.jsonl", synthesis)
        write_jsonl(output / f"{split}_lineage.jsonl", [{k: c[k] for k in lineage_keys if k in c} for c in rows])
        if split == "test":
            write_jsonl(output / "test_analysis_inputs.jsonl", [{k: r[k] for k in ("instruction", "input")} for r in analysis])
            write_jsonl(output / "test_synthesis_oracle_inputs.jsonl",
                        [{k: r[k] for k in ("instruction", "input")} for r in synthesis])
    manifest = read_json(source / "manifest.json")
    manifest["files"] = {name: file_hash(output / name) for name in manifest["files"]}
    manifest["claim_variants"] = {
        "policy": "every template row kept; one meaning-verified paraphrase row per accepted claim; "
                  "train/validation use T style cards, test uses test-only S cards",
        "writer": {"model": cp.MODEL, "version": cp.VERSION},
        "verifier": {"model": cp.MODEL, "version": cp.VERIFIER_VERSION},
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "variants_sha256": file_hash(work / "variants.jsonl"),
        "verifier_gate_sha256": file_hash(work / "verifier_validation_summary.json"),
        "paraphrases": dict(Counter(c["split"] for c in expanded if c["claim_variant"] == "paraphrase")),
    }
    write_json(output / "manifest.json", manifest)
    return {"paraphrases": manifest["claim_variants"]["paraphrases"], "validation": validate_splits_package(output)}
