"""Recompute deterministic checks on saved judgments, without new LLM calls.

Keep the original results unchanged. This audits a checker correction, not a
model/prompt change: expectations, judge decisions and completions are frozen.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from odyn_sft.common.storage import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from odyn_sft.survey_evaluation.synthesis_checks import VERSION, check_answer
from odyn_sft.survey_evaluation.synthesis_facts import build_facts
from odyn_sft.survey_evaluation.synthesis_judge import score


def run(raw: Path, responses: Path, output: Path, inputs: Path | None = None):
    payloads = {}
    if responses.is_dir():
        assert inputs is not None
        assert read_json(raw / 'manifest.json')['files'][str(inputs.resolve())] == file_hash(inputs)
        source_inputs = read_jsonl(inputs)
        files = sorted(responses.glob('case-*.json'))
        answers = {read_json(p)['record_id']: read_json(p)['response'] for p in files}
    else:
        files = [responses]
        records = [r for r in read_jsonl(responses) if r['sft2']]
        answers = {r['record_id']: r['sft2']['completion'] for r in records}
        payloads = {r['record_id']: json.loads(r['synthesis_input']) for r in records}
    source = read_jsonl(raw / 'results.jsonl')
    rows = []
    changed = []
    for original in source:
        row = dict(original)
        payload = (json.loads(source_inputs[row['source_index']]['input'])
                   if responses.is_dir() else payloads[row['record_id']])
        evidence = payload['api_response']
        assert digest(build_facts(evidence, payload['original_claim'])) == digest(row['facts'])
        row['checks'] = check_answer(answers[row['record_id']], evidence, row['facts'])
        for review in row['checks']['checks']['D3_no_suppressed_values']['requires_E7_semantic_review']:
            assert any(item['type'] == 'E7_suppression' and item['critical']
                       and item['id'].endswith(':' + str(review['group'])) for item in row['expectations'])
        if row['judged'] is not None:
            row['score'] = score(row['judged'], row['expectations'], row['checks'])
            passed = row['score']['case_passed']
            row['combined_pass'] = (passed and row['generation_status'] == 'successful') if passed is not None else None
        row['original_combined_pass'] = original['combined_pass']
        if row['combined_pass'] != original['combined_pass']:
            changed.append(row['record_id'])
        rows.append(row)
    summary = read_json(raw / 'summary.json')
    for variant, data in summary['variants'].items():
        selected = [r for r in rows if r['variant'] == variant]
        data['grounding_and_critical_expectations_passed'] = sum(r['score']['case_passed'] is True for r in selected)
        data['combined_passed'] = sum(r['combined_pass'] is True for r in selected)
        data['deterministic_check_failures'] = dict(Counter(
            name for r in selected for name, check in r['checks']['checks'].items() if not check['passed']))
    summary['checker_version'] = VERSION
    summary['llm_api_calls'] = 0
    summary['changed_record_ids'] = changed
    manifest = {'raw_results_sha256': file_hash(raw / 'results.jsonl'),
                'raw_manifest_sha256': file_hash(raw / 'manifest.json'),
                'responses': {str(p): file_hash(p) for p in files},
                'inputs_sha256': file_hash(inputs) if inputs else None,
                'checks_version': VERSION, 'implementation': file_hash(Path(__file__)),
                'claim_model_prompt_or_judge_decisions_changed': False, 'llm_api_calls': 0}
    write_json(output / 'manifest.json', manifest)
    write_jsonl(output / 'results.jsonl', rows)
    write_json(output / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, help='Frozen synthesis inputs, required for separate-stage case files')
    args = parser.parse_args()
    run(args.raw, args.responses, args.out, args.inputs)
