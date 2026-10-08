"""SFT1 adapter: survey API Python programs and deterministic evidence scoring."""
from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..common.storage import file_hash, read_json, read_jsonl
from .base import ReportLayout


def code_text(completion: str) -> str:
    """Unwrap one whole-response Markdown code fence, without repairing code."""
    match = re.fullmatch(
        r"\s*```(?:python|py)?\s*\n(.*?)\n```\s*", completion, re.DOTALL
    )
    return (match.group(1) if match else completion).strip()


def generated_outcome(record: dict[str, Any], case: dict[str, Any]) -> str:
    """Assign one mutually exclusive generated-code outcome for reporting."""
    if record.get("truncated"):
        return "truncated"
    if not record.get("syntax_valid"):
        error_type = (record.get("error") or {}).get("type")
        return "syntax_error" if error_type in {"SyntaxError", "IndentationError"} else "empty_or_invalid_output"
    if not record.get("execution_valid"):
        return "execution_error"
    contract = case.get("analysis_contract", {})
    required = contract.get("required_analyses", []) or contract.get("required_records", [])
    if required and record.get("measured_semantics_pass") is not True:
        return "semantic_failure" if record.get("measured_semantics_pass") is False else "semantic_check_unavailable"
    return "successful"



class AnalysisTask:
    name = 'sft1'
    view = 'analysis'
    scope = 'Step 1 code; measured semantics exclude gap/proxy meaning requiring expert/judge review'
    diagnostic_scope = 'Generated-code diagnostic; does not change checkpoint selection.'
    comparison_metrics = ('loss', 'syntax_valid_rate', 'execution_valid_rate',
                          'measured_semantic_pass_rate', 'generated_success_rate')

    def validate_suite(self, suite: Path) -> dict[str, Any]:
        from ..survey_evaluation.package_integrity import validate_splits_package
        from .sft1_suite import validate_small_training_suite, validate_training_suite
        version = read_json(suite / 'manifest.json').get('version')
        if version == 'synthetic-client-sft-1':
            return validate_training_suite(suite)
        if version == 'synthetic-smallest100-sft-1':
            return validate_small_training_suite(suite)
        return validate_splits_package(suite)

    def record_path(self, suite: Path, split: str, *, inputs: bool = False) -> Path:
        suffix = ('_inputs' if inputs else '_gold') if split == 'test' else ''
        return suite / f'{split}_analysis{suffix}.jsonl'

    def source_files(self, suite: Path, split: str) -> dict[str, str]:
        names = ['manifest.json', f'{split}_lineage.jsonl']
        if split == 'validation':
            names += ['validation_analysis.jsonl']
            if (suite / 'validation_private_gold.jsonl').exists():
                names += ['validation_private_gold.jsonl']
            else:
                names += ['private_gold.jsonl', 'respondents.parquet', 'metadata.json']
        else:
            names += ['test_analysis_inputs.jsonl', 'test_analysis_gold.jsonl',
                      'private_gold.jsonl', 'respondents.parquet', 'metadata.json']
        return {name: file_hash(suite / name) for name in names}

    def implementation_hashes(self) -> dict[str, str]:
        from ..survey_evaluation.provenance import implementation_hashes
        hashes = implementation_hashes()
        hashes.update({f'tasks/{p.name}': file_hash(p) for p in Path(__file__).parent.glob('*.py')})
        return hashes

    def scorer(self, suite: Path, split: str) -> AnalysisScorer:
        return AnalysisScorer(suite, split)

    def report_layout(self, use_adapter: bool) -> ReportLayout:
        return (ReportLayout('generated_validation_semantic_v2', 'selected-semantic-v2', 'report_semantic.json')
                if use_adapter else ReportLayout('generated_validation_base_v1', 'base-v1', 'report.json'))


class AnalysisScorer:
    """Keep survey loading, execution and SFT1 failure taxonomy out of the engine."""
    def __init__(self, suite: Path, split: str):
        from ..survey_evaluation.study_data import load_inputs
        self.per_study = split == 'validation' and (suite / 'validation_private_gold.jsonl').exists()
        self.lineage = read_jsonl(suite / f'{split}_lineage.jsonl')
        self.studies: dict[str, tuple[Any, Any]] = {}
        if self.per_study:
            gold = read_jsonl(suite / 'validation_private_gold.jsonl')
            for case in gold:
                directory = case['metadata']['study_dir']
                if directory not in self.studies:
                    self.studies[directory] = load_inputs(suite / directory / 'respondents.parquet',
                                                        suite / directory / 'metadata.json')
            self.data, self.metadata = None, None
        else:
            self.metadata = read_json(suite / 'metadata.json')
            self.data, _ = load_inputs(suite / 'respondents.parquet', suite / 'metadata.json')
            gold = [case for case in read_jsonl(suite / 'private_gold.jsonl') if case['split'] == split]
        self.by_record = {case['record_id']: case for case in gold}
        self.counts: Counter[str] = Counter()
        self.outcomes: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.details: Counter[str] = Counter()

    def case(self, source_index: int) -> dict[str, Any]:
        return self.by_record[self.lineage[source_index]['record_id']]

    def score(self, completion: str, case: dict[str, Any]) -> dict[str, Any]:
        from ..survey_evaluation.execution import check_analysis_semantics, execute
        program = code_text(completion)
        data, metadata = (self.studies[case['metadata']['study_dir']] if self.per_study
                          else (self.data, self.metadata))
        syntax, executed, semantics, evidence, error = False, False, None, None, None
        try:
            ast.parse(program)
            syntax = bool(program)
            if syntax:
                evidence = execute(program, data, metadata)
                executed = True
                contract = case['analysis_contract']
                if contract.get('required_analyses') or contract.get('required_records'):
                    semantics = check_analysis_semantics(contract, evidence, data, metadata)['passed']
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        return dict(program=program, syntax_valid=syntax, execution_valid=executed,
                    measured_semantics_pass=semantics, evidence=evidence, error=error)

    def outcome(self, record: dict[str, Any], case: dict[str, Any]) -> str:
        return generated_outcome(record, case)

    def accumulate(self, record: dict[str, Any], case: dict[str, Any]) -> None:
        self.outcomes[self.outcome(record, case)] += 1
        self.counts['syntax'] += int(record['syntax_valid'])
        self.counts['execution'] += int(record['execution_valid'])
        required = case['analysis_contract']
        if required.get('required_analyses') or required.get('required_records'):
            self.counts['measured_cases'] += 1
            self.counts['measured_semantics'] += int(record['measured_semantics_pass'] is True)
        error = record.get('error') or {}
        if error.get('type'):
            self.errors[error['type']] += 1
            detail = str(error.get('message', '')).split(':', 1)[0].strip() or error['type']
            self.details[f"{error['type']}/{detail}"] += 1

    def metrics(self, examples: int) -> dict[str, Any]:
        measured = self.counts['measured_cases']
        return {
            'syntax_valid_rate': self.counts['syntax'] / examples,
            'execution_valid_rate': self.counts['execution'] / examples,
            'generated_success_rate': self.outcomes['successful'] / examples,
            'generated_outcomes': {'successful': self.outcomes['successful'],
                'failed': examples - self.outcomes['successful'],
                'failure_types': {k:v for k,v in sorted(self.outcomes.items()) if k != 'successful'},
                'exception_types': dict(sorted(self.errors.items())),
                'failure_subtypes': dict(sorted(self.details.items()))},
            'generated_outcome_definition': 'Successful means the program parses, executes, and passes the deterministic API contract when required analyses or records are specified.',
            'measured_semantic_pass_rate': self.counts['measured_semantics'] / measured if measured else None,
            'measured_semantic_denominator': measured,
            'scope': AnalysisTask.scope,
        }
