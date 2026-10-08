"""Select existing research-response records without generation or provider calls.

Run with final/sft/src on PYTHONPATH. Train quotas preserve the source mix of
claim use cases and existing complexity bands. Validation keeps every research
record; held-out test files are copied byte-for-byte without selecting on them.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Any

from odyn_sft.common.storage import file_hash, read_json, read_jsonl, write_json, write_jsonl
from odyn_sft.tasks import get_task


def select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Largest-remainder proportional quotas; seeded hash order within strata."""
    if not 0 < count <= len(rows):
        raise ValueError(f"Requested {count} train records from {len(rows)} eligible")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item = row['metadata']
        strata[(item['use_case'], item.get('difficulty', {}).get('P', 'unknown'))].append(row)
    quotas = {key: count * len(items) // len(rows) for key, items in strata.items()}
    remainder_order = sorted(strata, key=lambda key: (-(count * len(strata[key]) % len(rows)), key))
    for key in remainder_order[:count - sum(quotas.values())]:
        quotas[key] += 1
    selected: set[str] = set()
    for key, items in strata.items():
        ranked = sorted(items, key=lambda row: hashlib.sha256(
            f"{seed}:{row['metadata']['record_id']}".encode()).hexdigest())
        selected.update(row['metadata']['record_id'] for row in ranked[:quotas[key]])
    return [row for row in rows if row['metadata']['record_id'] in selected]


def build(source: Path, output: Path, count: int, seed: int) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing packages are immutable')
    task = get_task('sft2')
    task.validate_suite(source)
    parent = read_json(source / 'manifest.json')
    token_filter = parent.get('token_filter', {})
    if token_filter.get('cap') != 32768 or token_filter.get('truncation') is not False:
        raise ValueError('Expected the existing complete-example 32768-token filtered suite')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.staging-', dir=output.parent))
    try:
        report: dict[str, Any] = {'seed': seed, 'train_limit': count, 'splits': {}}
        for split in ('train', 'validation'):
            rows = read_jsonl(task.record_path(source, split))
            lineage = read_jsonl(source / f'{split}_lineage.jsonl')
            for row, item in zip(rows, lineage, strict=True):
                if 'metadata' in row and row['metadata'] != item:
                    raise ValueError('Record/lineage metadata mismatch')
                row['metadata'] = item
            if any(row['metadata'].get('task_type') not in ('research_claim', 'api_tutorial') for row in rows):
                raise ValueError('Unknown task type; do not silently discard records')
            eligible = [row for row in rows if row['metadata']['task_type'] == 'research_claim']
            retained = select(eligible, count, seed) if split == 'train' else eligible
            write_jsonl(stage / f'{split}_synthesis.jsonl', retained)
            write_jsonl(stage / f'{split}_lineage.jsonl', [row['metadata'] for row in retained])
            report['splits'][split] = {
                'source_count': len(rows), 'excluded_api_tutorials': len(rows) - len(eligible),
                'eligible_research': len(eligible), 'retained': len(retained),
                'use_cases': dict(Counter(row['metadata']['use_case'] for row in retained)),
                'complexity_P': dict(Counter(row['metadata']['difficulty']['P'] for row in retained)),
                'study_count': len({row['metadata']['study_id'] for row in retained}),
            }
        for name in ('test_synthesis_gold.jsonl', 'test_lineage.jsonl', 'test_synthesis_inputs.jsonl'):
            shutil.copyfile(source / name, stage / name)
            assert file_hash(stage / name) == file_hash(source / name)
        write_json(stage / 'selection_report.json', report)
        manifest = {
            **{key: value for key, value in parent.items() if key not in ('files', 'source_files', 'counts', 'selection')},
            'counts': {'train': count, 'validation': report['splits']['validation']['retained'], 'test': parent['counts']['test']},
            'selection': 'Research-only train/validation; proportional train quotas by use_case and existing P complexity; test unchanged',
            'source_files': {'manifest.json': file_hash(source / 'manifest.json'), **parent['files']},
            'source_suite': str(source), 'llm_calls': 0,
            'files': {path.name: file_hash(path) for path in sorted(stage.iterdir())},
        }
        write_json(stage / 'manifest.json', manifest)
        validation = task.validate_suite(stage)
        stage.rename(output)
        return {'output': str(output), 'validation': validation, 'selection': report}
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == '__main__':
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-suite', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args.source_suite, args.out, args.train_count, args.seed), indent=2))
