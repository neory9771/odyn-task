"""Compare paired saved inference results without invoking a judge or loading models."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ..common.storage import read_json, read_jsonl, write_json


def compare(base: Path, sft: Path, output: Path) -> dict[str, Any]:
    """Pair a base run with an adapter run; the adapter side is labelled sft1 or sft2."""
    if output.exists():
        raise ValueError("Use a new comparison output")
    manifests = [
        read_json(p.with_suffix(p.suffix + ".manifest.json")) for p in (base, sft)
    ]
    for field in (
        "input_sha256",
        "config",
        "limit",
        "max_new_tokens",
        "do_sample",
        "precision",
    ):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError("Comparison settings differ: " + field)
    if manifests[0]["adapter"] is not None or manifests[1]["adapter"] is None:
        raise ValueError("Expected a base run and an adapter run")
    label = manifests[1]["config"].get("task", "adapter")
    records = [read_jsonl(p) for p in (base, sft)]
    if len(records[0]) != manifests[0]["examples"] or len(records[1]) != len(
        records[0]
    ):
        raise ValueError("Both inference runs must be complete and aligned")
    pairs = []
    for left, right in zip(*records, strict=True):
        if (left["source_index"], left["record_id"]) != (
            right["source_index"],
            right["record_id"],
        ):
            raise ValueError("Inference record alignment mismatch")
        pairs.append(
            {
                "source_index": left["source_index"],
                "record_id": left["record_id"],
                "base": left,
                label: right,
            }
        )
    report = {
        "examples": len(pairs),
        "base": str(base.resolve()),
        label: str(sft.resolve()),
        "outcomes": {
            name: dict(Counter(r["status"] for r in rows))
            for name, rows in zip(("base", label), records, strict=True)
        },
        "semantic_evaluation": "not_run",
        "pairs": pairs,
    }
    write_json(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.base, args.sft, args.out)
    print(f"Saved {result['examples']} paired responses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
