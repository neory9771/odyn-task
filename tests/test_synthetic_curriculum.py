"""Executable gold, contract mutation, isolation and filter regressions."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from odyn_sft.survey_api import CALLS, execute_program
from odyn_sft.survey_evaluation.meaning import semantics
from odyn_sft.common.storage import read_json
from odyn_sft.survey_api.tracing import TracedSurveyAPI
from odyn_sft.dataset_generation.synthetic.complexity import q_level,p_level
from odyn_sft.dataset_generation.synthetic.curriculum import build,validate,filter_subset,iter_rows,check_realization
from odyn_sft.dataset_generation.synthetic.curriculum_cases import lesson,run_case,study_blueprints,program_for
from odyn_sft.dataset_generation.synthetic.curriculum_studies import minimal_study,subset_study,scaled_spec
from odyn_sft.dataset_generation.synthetic.survey import generate_study


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def fail(*args,**kwargs):
        raise AssertionError('No provider calls allowed')
    monkeypatch.setattr(socket.socket,'connect',fail)


@pytest.fixture(autouse=True)
def offline_token_counter(monkeypatch):
    """Test package plumbing without downloading tokenizer vocabulary.

    Real o200k counts are exercised by the CLI smoke run; these regressions
    check budget rejection, alignment and isolation, not tokenizer accuracy.
    """
    import re
    import tiktoken

    class TestEncoder:
        def encode(self, text):
            return re.findall(r"\w+|[^\w\s]", text)

    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: TestEncoder())


@pytest.mark.parametrize('operation',sorted(CALLS))
def test_minimal_lesson_executes(operation):
    data,metadata,ids = minimal_study('lesson_'+operation,100)
    blueprint = lesson(operation,ids,100)
    data,metadata = subset_study(data,metadata,blueprint['variables'])
    case = run_case(blueprint,data,metadata,tutorial=True)
    assert case['reference_semantics_passed']
    assert 1<=len(metadata['variables'])<=4
    assert case['evidence']['results']
    if operation=='proportion':
        result = next(r for r in case['evidence']['results'] if r['kind']=='estimate')['estimates'][0]
        assert result['n']==230
        assert result['numerator']==data[ids['selected']].eq(True).sum()


@pytest.mark.parametrize('domain',('brand_tracking','transit_service','streaming_services','mobile_services'))
def test_scaled_study_rich_gold_and_controlled_claims(domain,tmp_path):
    spec,roles = scaled_spec('rich_'+domain,77,domain,256,20000)
    data,metadata,_ = generate_study(spec,tmp_path/'study',tmp_path/'hidden')
    assert len(metadata['variables'])==256
    assert sum(c['answer_type']=='rank' for c in metadata['variables'])==6
    assert all(c['id'].startswith('q') for c in metadata['variables'])
    blueprints = study_blueprints(metadata,roles,32,77)
    for b in blueprints:
        check_realization(b)
        case = run_case(b,data,metadata)
        assert case['reference_semantics_passed']
    b = deepcopy(blueprints[0]);b['claim']+=' and another proposition'
    with pytest.raises(ValueError,match='realization'):
        check_realization(b)


def test_wrong_population_and_missing_gap_are_rejected(tmp_path):
    spec,roles = scaled_spec('mutations',89,'brand_tracking',20,20000)
    data,metadata,_ = generate_study(spec,tmp_path/'study',tmp_path/'hidden')
    blueprints = study_blueprints(metadata,roles,32,89)
    blueprint = next(b for b in blueprints if b['pattern']=='population_and')
    case = run_case(blueprint,data,metadata)
    code = case['program'].replace(' & ',' | ')
    api = TracedSurveyAPI(data,metadata)
    evidence = execute_program(code,api);evidence['analysis_trace']=api.trace
    assert not semantics(case['analysis_contract'],evidence,data,metadata)['passed']
    blueprint = next(b for b in blueprints if b['pattern']=='proxy_and_gap')
    case = run_case(blueprint,data,metadata)
    wrong = deepcopy(case['evidence'])
    wrong['results'] = [r for r in wrong['results'] if not(r['kind']=='note' and r['label']=='proxy_variable')]
    assert not semantics(case['analysis_contract'],wrong,data,metadata)['passed']


def test_tutorial_eligible_base_is_checked():
    data,metadata,ids = minimal_study('restricted',88)
    blueprint = lesson('eligible',ids,88)
    data,metadata = subset_study(data,metadata,blueprint['variables'])
    case = run_case(blueprint,data,metadata,tutorial=True)
    api = TracedSurveyAPI(data,metadata)
    evidence = execute_program(f"proportion(var({ids['scale']!r}), 'Agree')",api)
    evidence['analysis_trace']=api.trace
    assert not semantics(case['analysis_contract'],evidence,data,metadata)['passed']


def test_complexity_rule_boundaries():
    q = dict(label_exposed=True,catalogue_variables=640,mapping_required=True)
    assert q_level(q)=='low'
    assert q_level({**q,'label_exposed':False})=='high'
    assert q_level({**q,'label_exposed':False,'catalogue_variables':128})=='medium'
    assert q_level({**q,'label_exposed':False,'catalogue_variables':64})=='low'
    assert q_level({**q,'mapping_required':False,'label_exposed':False})=='low'
    p = dict(n_subclaims=1,n_estimands=1,n_comparisons=1,n_filters=0,n_recodes=0,n_gap_calls=0,variable_defined_groups=0)
    assert p_level(p)=='low'
    assert p_level({**p,'n_filters':1})=='medium'
    assert p_level({**p,'n_estimands':2,'n_filters':1})=='high'
    assert p_level({**p,'n_subclaims':3})=='high'


@pytest.fixture
def package(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'counts':{'train':30,'validation':30,'test':30},'tutorial_fraction':.1,
                                 'widths':[20],'client_calibration_size':0,'client_pool_limit':0}))
    output = tmp_path/'curriculum'
    build(config,output)
    return output


def test_small_build_counts_isolation_and_filter(package,tmp_path):
    report = validate(package)
    assert report['cases']==90 and report['paired_records']==180
    assert not report['expert_gold_certified']
    report = filter_subset(package,tmp_path/'subset',limit=2,task_type='api_tutorial',max_variables=4)
    assert report['cases']==2
    subset = tmp_path/'subset'
    analysis = list(iter_rows(subset/'train_analysis.jsonl'))
    synthesis = list(iter_rows(subset/'train_synthesis.jsonl'))
    assert [r['metadata']['record_id'] for r in analysis]==[r['metadata']['record_id'] for r in synthesis]
    for row in analysis:
        assert 'metadata' not in json.loads(row['input'])
        assert (subset/row['metadata']['study_dir']/'respondents.parquet').exists()
    manifest = read_json(package/'manifest.json')
    assert manifest['llm_calls']==0 and not manifest['client_final_test_selected']


def test_tampered_dataset_is_rejected(package):
    with (package/'train_lineage.jsonl').open('a') as handle:
        handle.write('{}\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        validate(package)


def test_budget_failure_does_not_publish(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'counts':{'train':1,'validation':1,'test':1},'tutorial_fraction':1,
                                 'max_context_tokens':1,'client_calibration_size':0,'client_pool_limit':0}))
    with pytest.raises(ValueError,match='token budget'):
        build(config,tmp_path/'run')
    assert not (tmp_path/'run').exists()
    assert (tmp_path/'run.partial').exists()
