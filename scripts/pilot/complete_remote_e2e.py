"""Finish the isolated Qwen pilot, then grade both stages and their real pipeline.

Run on the remote GPU only. Training uses its frozen snapshot; evaluation uses
the separately saved final-code snapshot. Completed artifacts make restart safe.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/dev/shm/odyn-e2e')
TRAIN_CODE = ROOT / 'sft'
EVAL_CODE = ROOT / 'eval-sft'
CREDENTIALS = ROOT / 'private/credentials.json'
GATE = ROOT / 'inputs/judge-validation-report.json'


def command(name: str, args: list[str], *, code: Path = EVAL_CODE) -> None:
    """Keep a durable log and stop the pipeline if a subprocess fails."""
    from odyn_sft.common.storage import write_json
    write_json(ROOT / 'runs/pipeline-status.json', {'stage': name, 'started_at': time.time()})
    env = {**os.environ, 'PYTHONPATH': str(code / 'src'), 'HF_HOME': '/workspace/.hf_home'}
    with (ROOT / 'runs' / (name + '.log')).open('a', buffering=1) as log:
        subprocess.run([sys.executable, '-u', *args], cwd=code, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main() -> None:
    from odyn_sft.common.storage import read_json, read_jsonl, write_json
    from odyn_sft.survey_evaluation.analysis_judge import run as judge_analysis
    from odyn_sft.survey_evaluation.synthesis_runner import judge_jobs, make_job, validate_gate
    from odyn_sft.survey_metadata.catalogue import numeric_evidence, numeric_references

    # Do not interrupt the SFT1 worker already running when this continuation starts.
    first = ROOT / 'runs/qwen-sft1/train/training_summary.json'
    write_json(ROOT / 'runs/pipeline-status.json', {'stage': 'waiting_for_sft1'})
    deadline = time.monotonic() + 7200
    while not first.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('SFT1 did not finish within the continuation wait limit')
        time.sleep(15)
    assert read_json(first)['completed_max_epochs']
    assert read_json(first)['acceptance_checks']['passed']

    second = ROOT / 'runs/qwen-sft2/train/training_summary.json'
    if not second.exists():
        command('sft2-training', ['-m', 'odyn_sft.run_management.cli', 'train', '--config',
                                str(TRAIN_CODE / 'configs/qwen-e2e-sft2.json'),
                                '--disable-temperature-checks'], code=TRAIN_CODE)
    assert read_json(second)['completed_max_epochs']
    assert read_json(second)['acceptance_checks']['passed']

    # These gates require all epochs and validation boundaries before touching test.
    for stage in ('sft1', 'sft2'):
        report = ROOT / f'runs/qwen-{stage}/train/final_evaluation/report.json'
        if not report.exists():
            command(stage + '-final-test', ['-m', 'odyn_sft.run_management.cli', 'test', '--config',
                                          str(TRAIN_CODE / f'configs/qwen-e2e-{stage}.json'),
                                          '--disable-temperature-checks'])
        assert read_json(report)['status'] == 'complete'

    package = ROOT / 'work/package-variants'
    assert read_json(ROOT / 'runs/analysis-judge-controls/report.json')['passed']
    cases1 = ROOT / 'runs/qwen-sft1/train/final_evaluation/test/selected-final'
    command('sft1-judge', ['-m', 'odyn_sft.survey_evaluation.analysis_judge', '--package',
                         str(package), '--cases', str(cases1), '--out', str(ROOT / 'runs/sft1-judge'),
                         '--credentials-file', str(CREDENTIALS), '--workers', '8'])

    gate = read_json(GATE)
    validate_gate(gate)
    source = ROOT / 'work/sft2-suite/test_synthesis_inputs.jsonl'
    inputs = read_jsonl(source)
    lineage = read_jsonl(ROOT / 'work/sft2-suite/test_lineage.jsonl')
    cases2 = ROOT / 'runs/qwen-sft2/train/final_evaluation/test/selected-final'
    candidates = {read_json(p)['record_id']: read_json(p) for p in cases2.glob('case-*.json')}
    assert set(candidates) == {r['record_id'] for r in lineage}
    jobs = []
    for index, (row, identity) in enumerate(zip(inputs, lineage, strict=True)):
        candidate = candidates[identity['record_id']]
        status = 'successful' if candidate['outcome'] == 'generated' else candidate['outcome']
        jobs.append(make_job('sft2', index, identity['record_id'], row['input'],
                             candidate['response'], status))
    write_json(ROOT / 'runs/pipeline-status.json', {'stage': 'sft2-judge'})
    if not (ROOT / 'runs/sft2-judge/summary.json').exists():
        judge_jobs(jobs, ROOT / 'runs/sft2-judge', gate, [source, GATE, cases2 / 'metrics.json'],
                   credentials_file=CREDENTIALS, workers=8)
    command('sft2-grounding-audit', [str(ROOT / 'scripts/rescore_grounding.py'),
             '--raw', str(ROOT / 'runs/sft2-judge'), '--responses', str(cases2),
             '--inputs', str(source), '--out', str(ROOT / 'runs/sft2-judge-audited')])

    command('e2e', ['-m', 'odyn_sft.inference.e2e', '--sft1-config',
                    str(TRAIN_CODE / 'configs/qwen-e2e-sft1.json'), '--sft1-adapter',
                    str(ROOT / 'runs/qwen-sft1/train/best_adapter'), '--sft2-config',
                    str(TRAIN_CODE / 'configs/qwen-e2e-sft2.json'), '--sft2-adapter',
                    str(ROOT / 'runs/qwen-sft2/train/best_adapter'), '--package', str(package),
                    '--out', str(ROOT / 'runs/e2e'), '--validation-report', str(GATE),
                    '--credentials-file', str(CREDENTIALS), '--workers', '8',
                    '--disable-temperature-checks'])
    # Also review the actual SFT1 programs produced by the two-adapter engine.
    # Casting/adapter switching can change generation relative to separate tests.
    e2e_cases = ROOT / 'runs/e2e/analysis-cases'
    records = read_jsonl(ROOT / 'runs/e2e/e2e.jsonl')
    references = numeric_references(read_json(package / 'metadata.json')['variables'])
    for record in records:
        evidence = (json.loads(record['synthesis_input'])['api_response']
                    if record['synthesis_input'] else None)
        if evidence is not None:
            encoded = json.loads(json.dumps(evidence))
            # SFT2 receives compact numeric IDs. The analysis judge's reference
            # evidence uses canonical IDs; decode structured IDs only, preserving
            # every number, evidence citation, label and decision. Verify roundtrip.
            for row in [r for name in ('results', 'estimates', 'comparisons') for r in evidence.get(name, [])]:
                if 'variable' in row:
                    row['variable'] = references['variables'].get(str(row['variable']), row['variable'])
                for field in ('variables', 'controls'):
                    if isinstance(row.get(field), list):
                        row[field] = [references['variables'].get(str(v), v) for v in row[field]]
                if row.get('kind') == 'ranking':
                    row['block_id'] = references['blocks'].get(str(row['block_id']), row['block_id'])
                    for options in row['rankings'].values():
                        for option in options:
                            option['variable'] = references['variables'].get(str(option['variable']), option['variable'])
            for trace in evidence.get('analysis_trace', {}).values():
                trace['variable'] = references['variables'].get(str(trace['variable']), trace['variable'])
            assert numeric_evidence(evidence, references) == encoded
        write_json(e2e_cases / f"case-{record['source_index']:04d}.json", {
            'record_id': record['record_id'], 'program': record['sft1']['program'],
            'outcome': record['sft1']['outcome'], 'error': record['sft1']['error'],
            'evidence': evidence})
    write_json(ROOT / 'runs/pipeline-status.json', {'stage': 'e2e-analysis-judge'})
    judge_analysis(package, e2e_cases, ROOT / 'runs/e2e/analysis-judge',
                   credentials=CREDENTIALS, workers=8)
    analysis = {r['record_id']: r for r in read_jsonl(ROOT / 'runs/e2e/analysis-judge/results.jsonl')}
    command('e2e-grounding-audit', [str(ROOT / 'scripts/rescore_grounding.py'),
             '--raw', str(ROOT / 'runs/e2e/judge'), '--responses', str(ROOT / 'runs/e2e/e2e.jsonl'),
             '--out', str(ROOT / 'runs/e2e/judge-audited')])
    synthesis = {r['record_id']: r for r in read_jsonl(ROOT / 'runs/e2e/judge-audited/results.jsonl')}
    strict_passes = sum(analysis[r['record_id']]['combined_pass']
                        and bool(synthesis.get(r['record_id'], {}).get('combined_pass')) for r in records)
    write_json(ROOT / 'runs/pipeline-status.json', {'stage': 'complete', 'finished_at': time.time(),
               'sft1': read_json(ROOT / 'runs/sft1-judge/summary.json'),
               'sft2': read_json(ROOT / 'runs/sft2-judge-audited/summary.json'),
               'e2e': read_json(ROOT / 'runs/e2e/summary.json'),
               'e2e_both_judges': {'passed': strict_passes, 'rows': len(records),
                   'pass_rate': strict_passes / len(records),
                   'definition': 'Gold execution contract AND SFT1 code-meaning judge AND SFT2 grounded-perspective judge'}})


if __name__ == '__main__':
    main()
