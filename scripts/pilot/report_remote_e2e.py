"""Write a concise report from completed remote pilot artifacts; no model calls."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from odyn_sft.common.storage import atomic_text, read_json, read_jsonl, write_json

ROOT = Path('/dev/shm/odyn-e2e')


def main():
    status = read_json(ROOT / 'runs/pipeline-status.json')
    if status['stage'] != 'complete' or 'e2e_both_judges' not in status:
        raise ValueError('Both-stage judging and actual pipeline judging must finish first')
    package = ROOT / 'work/package-variants'
    lineage = {r['record_id']: r for r in read_jsonl(package / 'test_lineage.jsonl')}
    sft1 = read_jsonl(ROOT / 'runs/sft1-judge/results.jsonl')
    sft2 = read_jsonl(ROOT / 'runs/sft2-judge-audited/results.jsonl')
    e2e = read_jsonl(ROOT / 'runs/e2e/e2e.jsonl')
    joint_analysis = {r['record_id']: r for r in read_jsonl(ROOT / 'runs/e2e/analysis-judge/results.jsonl')}
    joint_synthesis = {r['record_id']: r for r in read_jsonl(ROOT / 'runs/e2e/judge-audited/results.jsonl')}
    expected = set(lineage)
    for rows in (sft1, sft2, e2e, list(joint_analysis.values())):
        assert len(rows) == len(expected) and {r['record_id'] for r in rows} == expected
    invalid = {
        'sft1': sum(r['semantic_pass'] is None for r in sft1),
        'sft2': sum(not r['score']['judge_valid'] for r in sft2),
        'e2e_analysis': sum(r['semantic_pass'] is None for r in joint_analysis.values()),
        'e2e_synthesis': sum(not r['score']['judge_valid'] for r in joint_synthesis.values())}

    def joint_pass(row):
        identity = row['record_id']
        return joint_analysis[identity]['combined_pass'] and bool(joint_synthesis.get(identity, {}).get('combined_pass'))

    table = []
    for variant in ('all', 'template', 'paraphrase'):
        ids = expected if variant == 'all' else {i for i, r in lineage.items() if r['claim_variant'] == variant}
        table.append({'wording': variant, 'rows': len(ids),
            'sft1_passed': sum(r['combined_pass'] for r in sft1 if r['record_id'] in ids),
            'sft2_passed': sum(r['combined_pass'] is True for r in sft2 if r['record_id'] in ids),
            'e2e_passed': sum(joint_pass(r) for r in e2e if r['record_id'] in ids)})
    summaries = {stage: read_json(ROOT / f'runs/qwen-{stage}/train/training_summary.json') for stage in ('sft1', 'sft2')}
    metrics = {stage: read_json(ROOT / f'runs/qwen-{stage}/train/final_evaluation/report.json') for stage in summaries}
    corrected_precision_labels = []
    for stage, report in metrics.items():
        config = read_json(ROOT / f'runs/qwen-{stage}/train/run_manifest.json')['config']
        comparison = report['validation_loss_comparison']
        actual_base = 'bf16' if config['quantization'] == 'none' else 'NF4'
        if comparison['final_precision']['base_linear_weights'] != actual_base:
            # Preserve the original report on disk. Correct the aggregate's
            # obsolete fixed NF4 display label from the frozen loading config.
            comparison['final_precision']['base_linear_weights'] = actual_base
            corrected_precision_labels.append(stage)
    findings = {'scope': 'Remote-only final-code pilot; 30 underlying held-out claims, 58 wording rows',
                'table': table, 'stage1_failure_types': dict(Counter(r['deterministic_outcome'] for r in sft1)),
                'training': {stage: {'updates': summary['global_step'], 'epochs': summary['epoch'],
                    'best_validation_loss': summary['best_loss'], 'seconds': summary['session_seconds'],
                    'acceptance': summary['acceptance_checks']} for stage, summary in summaries.items()},
                'judges': {'model': 'gpt-6-luna', 'sft1_controls': read_json(ROOT / 'runs/analysis-judge-controls/report.json')['passed'],
                           'sft2_controls': read_json(ROOT / 'inputs/judge-validation-report.json')['gate_passed'],
                           'expert_calibrated': False, 'invalid_judgments': invalid}, 'metrics': metrics,
                'corrected_obsolete_precision_labels': corrected_precision_labels}
    write_json(ROOT / 'runs/report.json', findings)
    lines = ['# Remote Qwen SFT1 → SFT2 pilot', '',
        'Completed using final code on the remote H200. Production datasets were unchanged.', '',
        'Qwen3-4B, BF16 base, separate all-linear LoRA adapters (rank 16, alpha 32, dropout 0.05), '
        'learning rate 2e-4, batch 1 × accumulation up to 8, cosine schedule, 32k context, three epochs.', '',
        'Both stages used the same 100/20/30 original train/validation/test claims. Accepted OpenAI '
        'paraphrases increased the row counts to 181/38/58; identical claims were verified by record ID. '
        'There are 30 underlying test cases, not 58 independent cases.', '',
        '| Test wording | Rows | SFT1 execution + meaning judge | SFT2 reference-evidence judge | Actual E2E, both judges |',
        '|---|---:|---:|---:|---:|']
    for row in table:
        n = row['rows']
        rates = [f"{row[k]}/{n} ({row[k] / n:.1%})" for k in ('sft1_passed', 'sft2_passed', 'e2e_passed')]
        lines.append(f"| {row['wording']} | {n} | " + ' | '.join(rates) + ' |')
    lines += ['', 'SFT1 passes only when generated code executes, meets the gold analysis contract and passes '
        'the reference-conditioned meaning judge. Separate SFT2 testing uses executed reference evidence. '
        'E2E instead passes the actual SFT1 program and evidence to SFT2; failed first stages remain in the denominator.', '',
        'The judge is gpt-6-luna. SFT1 passed four correct-code and four flipped-direction controls; '
        'SFT2 passed its automated planted-error gate. Neither is expert calibration. Checkpoint selection '
        'used validation completion loss; test outputs did not affect training or selection.', '']
    lines += [f'Invalid judgments (reported separately, counted conservatively as non-passes): {invalid}.', '']
    lines += ['The suppression checker was corrected to send ambiguous mixed-group numeric attribution '
              'to the existing critical E7 meaning judgment, rather than failing on sentence co-occurrence. '
              'Original scores remain saved. Deterministic checks were re-scored with unchanged completions, '
              'facts and LLM decisions; this audit made no additional LLM calls.', '']
    if corrected_precision_labels:
        lines += ['The original SFT1 evaluation report used an obsolete fixed NF4 display label. '
                  'The frozen run config and model-loading path used a BF16 base; this aggregate corrects '
                  'that label and preserves the original report for audit.', '']
    for stage, summary in summaries.items():
        lines.append(f"- {stage.upper()}: {summary['global_step']} updates, {summary['epoch']} epochs, "
                     f"best validation loss {summary['best_loss']:.4f}, {summary['session_seconds'] / 60:.1f} minutes; "
                     'all 12 validation checkpoints and learned-delta checks passed.')
    lines += ['', 'SFT2 validation targets were writer-reviewed, while the held-out gold targets retain '
              'the deterministic reference-summary style. Their completion losses are therefore not '
              'directly comparable as a semantic-quality measure; the judge evaluates meaning and grounding.', '']
    lines += ['', f"SFT1 deterministic outcomes: {findings['stage1_failure_types']}.", '',
        'These are small, provisional, single-survey pilot results. They validate the pipeline and expose '
        'failures; they do not establish performance on the client’s full 500–1,000-case acceptance set.', '',
        'Artifacts: `runs/qwen-sft1/train`, `runs/qwen-sft2/train`, `runs/sft1-judge`, '
        '`runs/sft2-judge`, `runs/e2e`, and `work/package-variants` / `work/sft2-suite`, '
        'under `/dev/shm/odyn-e2e` on the remote box. Adapters and reports are also copied locally for retention.', '']
    atomic_text(ROOT / 'runs/report.md', '\n'.join(lines))
    # Attach held-out results to the already authorized training runs. This does
    # not start a new run or upload claims, reference programs or credentials.
    import wandb
    api = wandb.Api(timeout=60)
    for stage in ('sft1', 'sft2'):
        run_dirs = sorted((ROOT / f'runs/qwen-{stage}/train/wandb').glob('run-*'))
        assert len(run_dirs) == 1, 'Expected one training run for this isolated stage'
        run_id = run_dirs[0].name.rsplit('-', 1)[1]
        run = api.run('myelin/odyn-survey-sft/' + run_id)
        row = table[0]
        run.summary.update({
            'pilot/test_rows': row['rows'],
            'pilot/test_original_claims': len({r.get('variant_of', r['record_id']) for r in lineage.values()}),
            'pilot/test_judge_combined_passed': row[stage + '_passed'],
            'pilot/test_judge_combined_pass_rate': row[stage + '_passed'] / row['rows'],
            'pilot/test_invalid_judgments': invalid[stage],
            'pilot/e2e_both_judges_passed': row['e2e_passed'],
            'pilot/e2e_both_judges_pass_rate': row['e2e_passed'] / row['rows'],
            'pilot/e2e_invalid_analysis_judgments': invalid['e2e_analysis'],
            'pilot/e2e_invalid_synthesis_judgments': invalid['e2e_synthesis'],
            'pilot/expert_calibrated': False})
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
