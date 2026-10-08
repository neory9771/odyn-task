"""The code judge must cover its rubric and cite the actual candidate code."""
from odyn_sft.survey_evaluation.analysis_judge import ITEMS, review
from odyn_sft.survey_evaluation.synthesis_judge import SynthesisJudgement


def judgment(quote='not_measured', passed=True):
    return SynthesisJudgement.model_validate({'results': [
        {'expectation_id': item['id'], 'passed': passed, 'quote': quote,
         'explanation': 'The requested analysis is represented.'}
        for item in ITEMS
    ]})


class Provider:
    def __init__(self, results):
        self.results = iter(results)
        self.requests = []

    def complete(self, instruction, payload, schema, request_id):
        self.requests.append(request_id)
        return next(self.results)


def test_valid_quotes_and_all_criteria_required():
    provider = Provider([judgment()])
    result = review(provider, {'candidate_program': "not_measured('cause', 'not identified')"}, 'case')
    assert result['semantic_pass'] is True
    assert provider.requests == ['case']


def test_a_failed_criterion_rejects_the_program():
    provider = Provider([judgment(passed=False)])
    assert review(provider, {'candidate_program': 'not_measured()'}, 'case')['semantic_pass'] is False


def test_invented_quotes_get_one_format_repair_only():
    provider = Provider([judgment('invented'), judgment('invented')])
    result = review(provider, {'candidate_program': 'not_measured()'}, 'case')
    assert result['semantic_pass'] is None
    assert provider.requests == ['case', 'case:quote-repair']


def test_false_verdict_cannot_cite_invented_code():
    provider = Provider([judgment('invented', False), judgment('not_measured', False)])
    result = review(provider, {'candidate_program': 'not_measured()'}, 'case')
    assert result['semantic_pass'] is False
    assert len(provider.requests) == 2


def test_completed_benchmark_keeps_invalid_judgments_without_aborting(monkeypatch, tmp_path):
    from odyn_sft.survey_evaluation import analysis_judge
    monkeypatch.setattr(analysis_judge, 'run', lambda *a, **kw: {'cases': 1, 'invalid_or_error': 1})
    assert analysis_judge.main(['--package', str(tmp_path), '--cases', str(tmp_path),
                                '--out', str(tmp_path)]) == 0
