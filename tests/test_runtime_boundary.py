"""Runtime scoring works even when every dataset generator import is forbidden."""

import os
import subprocess
import sys

from odyn_sft.common.storage import write_json, write_jsonl
from odyn_sft.dataset_generation.use_cases import fixture


def test_scoring_and_model_evaluation_imports_work_without_generators(tmp_path):
    data, metadata = fixture()
    data.to_parquet(tmp_path / "respondents.parquet")
    write_json(tmp_path / "metadata.json", metadata)
    write_jsonl(tmp_path / "validation_lineage.jsonl", [{"record_id": "case-1"}])
    contract = {
        "required_analyses": [
            {
                "groups": {"A": ["A"], "B": ["B"]},
                "group_variable": "segment",
                "variable": "choice",
                "operation": "value=Pass A",
                "component_id": "c1",
                "threshold": None,
                "expect": ">",
            }
        ]
    }
    write_jsonl(
        tmp_path / "private_gold.jsonl",
        [{"record_id": "case-1", "split": "validation", "analysis_contract": contract}],
    )
    script = r"""
import importlib.abc
import sys
from pathlib import Path
class BlockGeneration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'odyn_sft.dataset_generation' or fullname.startswith('odyn_sft.dataset_generation.'):
            raise AssertionError('Runtime imported dataset generation: ' + fullname)
sys.meta_path.insert(0, BlockGeneration())
from odyn_sft.pytorch_sft.evaluation.validation import evaluate_base_validation
from odyn_sft.survey_evaluation.package_integrity import validate_splits_package
from odyn_sft.tasks import get_task
from odyn_sft.survey_evaluation.provenance import implementation_hashes
hashes = implementation_hashes()
assert hashes and not any(key.startswith('dataset_generation/') for key in hashes)
scorer = get_task('sft1').scorer(Path(sys.argv[1]), 'validation')
case = scorer.case(0)
program = "g = group('segment', {'A': ['A'], 'B': ['B']})\nwith subclaim('c1', 'A chooses Pass A more often than B'):\n    e = proportion('choice', 'Pass A', by=g)\n    compare(e, 'A', 'B', expect='>')\n"
result = scorer.score(program, case)
assert result['execution_valid'], result['error']
assert result['measured_semantics_pass'] is True, result
scorer.accumulate(result, case)
assert scorer.metrics(1)['generated_success_rate'] == 1
assert not any(name.startswith('odyn_sft.dataset_generation') for name in sys.modules)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
