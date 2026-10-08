"""Small live control check: gold code accepted, flipped directions rejected.

This is an automated pilot diagnostic, not expert calibration. Results are saved
before model outputs are submitted to the same reference-conditioned rubric.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from odyn_sft.common.storage import digest, read_jsonl, write_json
from odyn_sft.providers.openai import ResponsesProvider
from odyn_sft.survey_evaluation.analysis_judge import ITEMS, MODEL, VERSION, review

ROOT = Path('/dev/shm/odyn-e2e')


def main():
    package = ROOT / 'work/package-variants'
    lineage = read_jsonl(package / 'test_lineage.jsonl')
    gold = {r['record_id']: r for r in read_jsonl(package / 'private_gold.jsonl')}
    selected = []
    for identity, reference in zip(lineage, read_jsonl(package / 'test_analysis_gold.jsonl'), strict=True):
        if identity.get('claim_variant') != 'template' or "expect='>'" not in reference['output']:
            continue
        selected.append((identity, reference))
        if len(selected) == 4:
            break
    assert len(selected) == 4
    jobs = []
    for identity, reference in selected:
        contract = gold[identity['record_id']]
        for variant in ('gold', 'flip_direction'):
            candidate = reference['output']
            if variant == 'flip_direction':
                candidate = candidate.replace("expect='>'", "expect='<'", 1)
            payload = {'claim': contract['claim'], 'reference_program': reference['output'],
                       'reference_evidence': contract['evidence'], 'expectations': ITEMS,
                       'candidate_program': candidate, 'candidate_evidence': None,
                       'execution_error': None}
            jobs.append((variant, payload))

    def work(job):
        variant, payload = job
        key = digest({'version': VERSION, 'payload': payload})
        provider = ResponsesProvider(MODEL, ROOT / 'runs/analysis-judge-controls/ledgers' / (key + '.json'),
                                     max_output_tokens=4000, reasoning_effort='low',
                                     credentials_file=ROOT / 'private/credentials.json')
        result = review(provider, payload, key)
        return {'control': variant, 'expected_pass': variant == 'gold', **result}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(work, jobs))
    passed = all(r['semantic_pass'] is r['expected_pass'] for r in results)
    write_json(ROOT / 'runs/analysis-judge-controls/report.json', {
        'version': VERSION, 'model': MODEL, 'passed': passed, 'cases': len(results),
        'scope': 'Four gold references and four flipped-direction controls; not expert calibration',
        'results': results})
    print('analysis judge controls:', passed)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
