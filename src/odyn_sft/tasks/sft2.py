"""SFT2 adapter: evidence-conditioned research perspectives, never Python code.

Generation diagnostics intentionally do not judge whether a perspective is
correct. Grounding and semantic acceptance require the separate calibrated judge.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..common.storage import file_hash, read_json, read_jsonl
from .base import ReportLayout


class SynthesisTask:
    name = 'sft2'
    view = 'synthesis'
    scope = 'Step 2 response generation; semantic grounding and perspective quality require a separate calibrated judge'
    diagnostic_scope = 'Response-generation diagnostic; no semantic judge was run and checkpoint selection is unchanged.'
    comparison_metrics = ('loss', 'nonempty_response_rate', 'generated_output_rate')

    def record_path(self, suite: Path, split: str, *, inputs: bool = False) -> Path:
        if split != 'test':
            return suite / f'{split}_synthesis.jsonl'
        if not inputs:
            return suite / 'test_synthesis_gold.jsonl'
        standard = suite / 'test_synthesis_inputs.jsonl'
        return standard if standard.exists() else suite / 'test_synthesis_oracle_inputs.jsonl'

    def validate_suite(self, suite: Path) -> dict[str, Any]:
        manifest = read_json(suite / 'manifest.json')
        if manifest.get('version') != 'synthesis-sft-1':
            # Existing frozen paired analysis/synthesis packages have their own audit.
            from ..survey_evaluation.package_integrity import validate_splits_package
            result = validate_splits_package(suite)
            for split in ('train', 'validation', 'test'):
                if not self.record_path(suite, split).exists():
                    raise ValueError('Missing synthesis records for ' + split)
            return result
        hashes = manifest['files']
        for name, expected in hashes.items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Invalid suite artifact path')
            if file_hash(suite / name) != expected:
                raise ValueError('Dataset file changed: ' + name)
        owners: dict[tuple[str, str], str] = {}
        for split in ('train', 'validation', 'test'):
            path = self.record_path(suite, split)
            lineage_path = suite / f'{split}_lineage.jsonl'
            for required in (path, lineage_path):
                if str(required.relative_to(suite)) not in hashes:
                    raise ValueError('Required synthesis artifact is not hashed: ' + required.name)
            rows, lineage = read_jsonl(path), read_jsonl(lineage_path)
            if not rows or len(rows) != manifest['counts'][split] or len(rows) != len(lineage):
                raise ValueError('Missing/misaligned synthesis split: ' + split)
            for row, item in zip(rows, lineage, strict=True):
                for field in ('instruction', 'input', 'output'):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError('Synthesis records require nonempty text: ' + field)
                if item.get('split', split) != split:
                    raise ValueError('Synthesis lineage split mismatch')
                if 'metadata' in row and row['metadata'] != item:
                    raise ValueError('Synthesis record/lineage mismatch')
                for kind, keys in (('record', [item['record_id']]), ('family', [item['family_id']]),
                                   ('study', [item['study_id']] if 'study_id' in item else []),
                                   ('atomic', item.get('component_keys', []))):
                    for key in keys:
                        previous = owners.get((kind, key))
                        if previous is not None and (previous != split or kind == 'record'):
                            raise ValueError('Synthesis identity overlap: ' + kind)
                        owners[(kind, key)] = split
            if split == 'test':
                inputs_path = self.record_path(suite, split, inputs=True)
                if str(inputs_path.relative_to(suite)) not in hashes:
                    raise ValueError('Held-out synthesis inputs must be hashed')
                inputs = read_jsonl(inputs_path)
                if len(inputs) != len(rows) or any(
                    prompt != {key: reference[key] for key in ('instruction', 'input')}
                    for prompt, reference in zip(inputs, rows, strict=True)
                ):
                    raise ValueError('Held-out synthesis prompt/reference mismatch')
        return {'passed': True, 'counts': manifest['counts'], 'task': self.name,
                'test_used_for_selection': False, 'expert_review_complete': manifest.get('release_status') == 'human_approved'}

    def source_files(self, suite: Path, split: str) -> dict[str, str]:
        paths = [suite / 'manifest.json', self.record_path(suite, split), suite / f'{split}_lineage.jsonl']
        if split == 'test':
            paths.append(self.record_path(suite, split, inputs=True))
        return {str(path.relative_to(suite)): file_hash(path) for path in paths}

    def implementation_hashes(self) -> dict[str, str]:
        return {f'tasks/{p.name}': file_hash(p) for p in Path(__file__).parent.glob('*.py')}

    def scorer(self, suite: Path, split: str) -> SynthesisScorer:
        return SynthesisScorer(suite, split)

    def report_layout(self, use_adapter: bool) -> ReportLayout:
        return ReportLayout('generated_validation_sft2_adapter' if use_adapter else 'generated_validation_sft2_base',
                            'selected-response' if use_adapter else 'base-response', 'report.json')


class SynthesisScorer:
    """Mechanical output checks only; no execution and no semantic acceptance claim."""
    def __init__(self, suite: Path, split: str):
        self.lineage = read_jsonl(suite / f'{split}_lineage.jsonl')
        self.counts: Counter[str] = Counter()

    def case(self, source_index: int) -> dict[str, Any]:
        return self.lineage[source_index]

    def score(self, completion: str, case: dict[str, Any]) -> dict[str, Any]:
        return {'response': completion, 'nonempty_response': bool(completion.strip()),
                'semantic_judgment': None, 'error': None}

    def outcome(self, record: dict[str, Any], case: dict[str, Any]) -> str:
        if record['truncated']:
            return 'truncated'
        return 'generated' if record['nonempty_response'] else 'empty_output'

    def accumulate(self, record: dict[str, Any], case: dict[str, Any]) -> None:
        self.counts[self.outcome(record, case)] += 1
        self.counts['nonempty'] += int(record['nonempty_response'])

    def metrics(self, examples: int) -> dict[str, Any]:
        return {'nonempty_response_rate': self.counts['nonempty'] / examples,
                'generated_output_rate': self.counts['generated'] / examples,
                'generated_outcomes': {'generated': self.counts['generated'],
                    'failed': examples - self.counts['generated'],
                    'failure_types': {key: self.counts[key] for key in ('truncated', 'empty_output') if self.counts[key]}},
                'generated_outcome_definition': 'Generated means nonempty text ending within the generation budget; it does not mean the perspective is correct or grounded.',
                'semantic_evaluation_status': 'not_run_requires_calibrated_judge',
                'scope': SynthesisTask.scope}
