"""Validate a judge on frozen positive/negative controls and positive repeat calls."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..common.storage import digest, file_hash, read_json, write_json
from ..providers.openai import ResponsesProvider
from . import synthesis_judge as judge


def run(
    bank: Path,
    output: Path,
    credentials_file: Path | None = None,
    workers: int = 4,
    request_cache: Path | None = None,
) -> dict:
    controls = read_json(bank)["controls"]
    jobs = [
        (row, repeat)
        for row in controls
        for repeat in ([1, 2] if row["expected_pass"] else [1])
    ]
    manifest = {
        "bank_sha256": file_hash(bank),
        "judge_model": judge.MODEL,
        "judge_version": judge.VERSION,
        "prompt_sha256": digest(judge.INSTRUCTION),
    }
    if output.exists() and read_json(output / "manifest.json") != manifest:
        raise ValueError("Existing calibration differs")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)

    def call(job):
        row, repeat = job
        key = f"{row['id']}-repeat-{repeat}"
        path = output / "cases" / (key + ".json")
        if path.exists():
            return read_json(path)
        provider = ResponsesProvider(
            judge.MODEL,
            (request_cache or output / "ledgers") / (key + ".json"),
            max_output_tokens=judge.MAX_OUTPUT_TOKENS,
            reasoning_effort=judge.REASONING_EFFORT,
            credentials_file=credentials_file,
        )
        value = provider.complete(
            judge.INSTRUCTION,
            json.dumps(
                {
                    "facts": row["facts"],
                    "expectations": [
                        {"id": row["item"]["id"], "statement": row["item"]["statement"]}
                    ],
                    "answer": row["answer"],
                }
            ),
            judge.SynthesisJudgement,
            key,
        )
        checked = judge.validate(value, [row["item"]], row["answer"])
        observed = (
            checked["results"].get(row["item"]["id"], {}).get("passed")
            if checked["valid"]
            else None
        )
        result = {
            "id": row["id"],
            "repeat": repeat,
            "error_type": row["error_type"],
            "expected_pass": row["expected_pass"],
            "observed_pass": observed,
            "judged": checked,
        }
        write_json(path, result)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(call, jobs))
    negative = defaultdict(list)
    positive = []
    for row in results:
        if not row["expected_pass"]:
            negative[row["error_type"]].append(row)
        else:
            positive.append(row)
    detection = {
        name: {
            "n": len(rows),
            "detected": sum(r["observed_pass"] is False for r in rows),
            "rate": sum(r["observed_pass"] is False for r in rows) / len(rows),
        }
        for name, rows in negative.items()
    }
    first = {r["id"]: r for r in positive if r["repeat"] == 1}
    repeated = [r for r in positive if r["repeat"] == 2]
    failure = sum(r["observed_pass"] is not True for r in positive) / len(positive)
    agreement = sum(
        r["observed_pass"] is not None
        and r["observed_pass"] == first[r["id"]]["observed_pass"]
        for r in repeated
    ) / len(repeated)
    invalid = sum(not r["judged"]["valid"] for r in results)
    report = {
        "judge": {
            "model": judge.MODEL,
            "version": judge.VERSION,
            "reasoning_effort": judge.REASONING_EFFORT,
        },
        "prompt_sha256": digest(judge.INSTRUCTION),
        "calls": len(jobs),
        "invalid": invalid,
        "detection": detection,
        "unmodified": {"item_failure_rate": failure, "answers": len(positive)},
        "self_agreement": {"rate": agreement, "items": len(repeated)},
        "gate_passed": not invalid
        and all(r["rate"] >= 0.9 for r in detection.values())
        and failure <= 0.05
        and agreement >= 0.95,
        "scope": "Small controlled rule bank; not expert calibration or a full survey benchmark",
        "bank_sha256": file_hash(bank),
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-cache", type=Path)
    args = parser.parse_args()
    report = run(
        args.bank, args.out, args.credentials_file, args.workers, args.request_cache
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["gate_passed"] else 1)
