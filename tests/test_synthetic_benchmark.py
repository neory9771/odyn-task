"""Offline tests for synthetic StudySpec, survey process, truth layer and benchmark splits."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from odyn_sft.survey_api import run_reference
from odyn_sft.common.storage import read_json, read_jsonl, write_json, file_hash
from odyn_sft.dataset_generation.synthetic import benchmark
from odyn_sft.dataset_generation.synthetic.population import causal_truth, group_shares, simulate
from odyn_sft.dataset_generation.synthetic.spec import validate_spec
from odyn_sft.dataset_generation.synthetic.survey import field_survey, generate_study
from odyn_sft.dataset_generation.synthetic.templates import template_spec


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Network forbidden')))


def small(domain='brand_tracking', seed=3, size=20_000):
    return template_spec(domain, 'T1', seed, size)


def test_templates_validate_and_are_seeded():
    assert small() == small()
    assert small(seed=3) != small(seed=4)
    assert validate_spec(small('transit_service'))


@pytest.mark.parametrize('mutate,message', [
    (lambda s: s['latent_variables'][2].setdefault('parents', {}).update({'lat_consider_a': .5}), 'cycle'),
    (lambda s: s['intended_stressors'].append('routing_ambiguity') if 'routing_ambiguity' not in s['intended_stressors']
     else s['intended_stressors'].append('nonexistent'), 'stressor'),
    (lambda s: s['questionnaire']['blocks'][0].update(routing={'variable': 'next_brand', 'values': ['Brand A'], 'documented': True}),
     'routing'),
    (lambda s: s['measurement_model']['unmeasured_constructs'].append('lat_aware_a'), 'unmeasured'),
    (lambda s: s['demographics'][0]['contrasts'].append({'A': ['18-24'], 'B': ['18-24']}), 'disjoint'),
])
def test_invalid_specs_rejected(mutate, message):
    spec = small()
    spec['measurement_model']['checkbox_encoding'] = 'true_false'
    for block in spec['questionnaire']['blocks']:
        if block.get('routing'):
            block['routing']['documented'] = True
    spec['intended_stressors'] = [s for s in spec['intended_stressors'] if s not in {'routing_ambiguity', 'denominator_ambiguity'}]
    validate_spec(spec)
    mutate(spec)
    with pytest.raises(ValueError):
        validate_spec(spec)


def test_population_truth_matches_independent_pandas():
    spec = small()
    population = simulate(spec)
    rows = group_shares(population, 'aware_brand_a', 'selection', 'age_band', {'18-34': ['18-24', '25-34'], '55+': ['55-64', '65 and above']})
    labels = np.asarray(spec['demographics'][0]['values'], dtype=object)[population.demographics['age_band']]
    hit = population.responses['aware_brand_a']
    young = np.isin(labels, ['18-24', '25-34'])
    assert rows[0]['population_n'] == young.sum()
    assert rows[0]['share'] == pytest.approx(hit[young].mean(), abs=1e-12)
    # Routed item truth uses only routing-eligible members.
    routed = group_shares(population, 'consider_brand_a', 'selection', 'gender', {'Female': ['Female'], 'Male': ['Male']})
    eligible = population.responses['category_user'] == 0
    assert sum(r['population_n'] for r in routed) == eligible.sum()


def test_causal_truth_distinguishes_effect_from_confounding():
    spec = small()
    # Remove every causal path from region; keep its association through lat_u_lifestyle.
    for latent in spec['latent_variables']:
        latent.get('effects', {}).pop('region', None)
    spec['demographics'][2]['depends_on'] = {'lat_u_lifestyle': [0, 1.5, 0, -1.5, 0]}
    for latent in spec['latent_variables']:
        if latent['id'] == 'lat_aware_a':
            latent['parents']['lat_u_lifestyle'] = 1.0
    spec['intended_stressors'] = []
    population = simulate(validate_spec(spec))
    truth = causal_truth(population, 'aware_brand_a', 'selection', 'region', {'South': ['South'], 'West': ['West']})
    assert truth['demographic_is_causal_ancestor'] is False
    assert abs(truth['average_causal_effect']) < 1e-12
    assert truth['confounding_latents'] == ['lat_u_lifestyle']
    shares = group_shares(population, 'aware_brand_a', 'selection', 'region', {'South': ['South'], 'West': ['West']})
    assert shares[0]['share'] - shares[1]['share'] > .1  # associated, not caused


def test_checkbox_blanks_shrink_api_base_exactly_as_measurement_truth_records(tmp_path):
    spec = small()
    spec['measurement_model'].update(checkbox_encoding='true_or_blank', item_nonresponse=0.0, error_rate=0.0)
    spec['intended_stressors'] = sorted(set(spec['intended_stressors']) - {'measurement_error', 'item_nonresponse'} | {'routing_ambiguity'})
    data, metadata, _ = generate_study(validate_spec(spec), tmp_path/'study', tmp_path/'hidden')
    truth = read_json(tmp_path/'hidden'/'measurement_truth.json')['blocks']['brand_awareness']
    evidence = run_reference("e = proportion(var('aware_brand_a'))\n", data, metadata)
    estimate = next(r for r in evidence['results'] if r['kind'] == 'estimate')['estimates'][0]
    assert estimate['n'] == truth['shown'] - truth['excluded_from_api_base']
    assert truth['excluded_from_api_base'] > 0


def test_visible_artefacts_hide_dgp(tmp_path):
    spec = small()
    generate_study(spec, tmp_path/'study', tmp_path/'hidden')
    visible = ' '.join(p.read_text() for p in (tmp_path/'study').glob('*.json'))
    for latent in spec['latent_variables']:
        assert latent['id'] not in visible and latent['label'] not in visible
    assert 'oracle_weight' not in visible and 'loading' not in visible
    frame = pd.read_parquet(tmp_path/'study'/'respondents.parquet')
    assert not any(c.startswith('lat_') for c in frame.columns)
    with pytest.raises(ValueError):
        generate_study(spec, tmp_path/'study', tmp_path/'hidden2')


def config(tmp_path, **changes):
    value = {'workspace': '.', 'seed': 11, 'population_size': 20_000,
             'family_partition_ratios': {'train': .6, 'validation': .2, 'test': .2}, 'train_variants': 2,
             'familiar_test_cases_per_study': 2,
             'studies': {'train': [{'study_id': 'A1', 'domain': 'brand_tracking', 'seed': 1},
                                   {'study_id': 'A2', 'domain': 'transit_service', 'seed': 2}],
                         'validation': [{'study_id': 'B1', 'domain': 'brand_tracking', 'seed': 3}],
                         'test': [{'study_id': 'C1', 'domain': 'brand_tracking', 'seed': 4}]},
             'splits': {s: {'cases_per_study': 4, 'coverage_minima': {}, 'interaction_minima': {}} for s in benchmark.SPLITS}}
    value.update(changes)
    write_json(tmp_path/'config.json', value)
    return tmp_path/'config.json'


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic')
    result = benchmark.build(config(root), root/'out')
    return root/'out', result


def test_build_isolates_studies_and_families(built):
    out, result = built
    assert result['records'] == {'train': 16, 'validation': 4, 'test': 4, 'test_familiar': 2}
    assert benchmark.validate_benchmark(out)['integrity_valid']
    plan = read_json(out/'plan.json')
    gold = read_jsonl(out/'hidden'/'private_gold.jsonl')
    by_slice = {}
    for case in gold:
        by_slice.setdefault(case['slice'], set()).add(case['study_id'])
        owners = {plan['family_owners'].get(k) for k in case['component_keys']} - {None}
        assert owners == {'train' if case['slice'] == 'test_familiar' else case['slice']}
    assert by_slice['train'] == {'A1', 'A2'} and by_slice['test'] == by_slice['test_familiar'] == {'C1'}
    train_atoms = {k for c in gold if c['slice'] == 'train' for k in c['component_keys']}
    strict_atoms = {k for c in gold if c['slice'] in ('validation', 'test') for k in c['component_keys']}
    assert not train_atoms & strict_atoms
    lineage = read_jsonl(out/'train_lineage.jsonl')
    assert {r['study_dir'] for r in lineage} == {'studies/A1', 'studies/A2'}


def test_gold_follows_evidence_and_truth_stays_hidden(built):
    out, _ = built
    for case in read_jsonl(out/'hidden'/'private_gold.jsonl'):
        assert case['latent_truth']['gold_policy'].startswith('Gold follows')
        for component, truth in zip([c for c in case['components'] if 'analytical_core' in c], case['latent_truth']['components']):
            assert truth['E'] == component['E']
        assert '2022' not in case['claim'] and 'United States' not in json.dumps(case)
    for name in ('train_analysis.jsonl', 'train_synthesis.jsonl', 'test_analysis_inputs.jsonl'):
        text = (out/name).read_text()
        assert 'latent_truth' not in text and 'average_causal_effect' not in text
    for row in read_jsonl(out/'test_analysis_inputs.jsonl'):
        assert set(row) == {'instruction', 'input'}


def test_validation_detects_tampering_and_leaks(built, tmp_path):
    import shutil
    out, _ = built
    copy_root = tmp_path/'copy'
    shutil.copytree(out, copy_root)
    path = copy_root/'train_synthesis.jsonl'
    original = path.read_text()
    path.write_text(original.replace('"output": "', '"output": "lat_media_exposure ', 1))
    assert path.read_text() != original
    manifest = read_json(copy_root/'manifest.json')
    manifest['files']['train_synthesis.jsonl'] = file_hash(path)
    write_json(copy_root/'manifest.json', manifest)
    with pytest.raises(ValueError, match='alignment|leaked'):
        benchmark.validate_benchmark(copy_root)
    gold = read_jsonl(out/'hidden'/'private_gold.jsonl')
    plan = read_json(out/'plan.json')
    owners = copy.deepcopy(plan['family_owners'])
    test_core = next(k for c in gold if c['slice'] == 'test' for k in c['component_keys'] if k in owners)
    owners[test_core] = 'train'
    with pytest.raises(ValueError, match='ownership'):
        benchmark.assert_isolation(gold, owners, plan['study_splits'])


def test_config_rejects_reused_study_and_existing_output(tmp_path):
    path = config(tmp_path)
    value = read_json(path)
    value['studies']['test'].append(value['studies']['train'][0])
    write_json(path, value)
    with pytest.raises(ValueError, match='more than once'):
        benchmark.load_config(path)
    (tmp_path/'exists').mkdir()
    with pytest.raises(ValueError, match='new output'):
        benchmark.build(config(tmp_path), tmp_path/'exists')


def test_infeasible_selection_publishes_nothing(tmp_path):
    path = config(tmp_path, splits={s: {'cases_per_study': 100000, 'coverage_minima': {}, 'interaction_minima': {}}
                                    for s in benchmark.SPLITS})
    with pytest.raises(ValueError, match='shortages'):
        benchmark.build(path, tmp_path/'out')
    assert not (tmp_path/'out').exists() and not (tmp_path/'.out.partial').exists()


def test_survey_process_is_deterministic():
    spec = small()
    population = simulate(spec)
    a, design_a, _ = field_survey(spec, population)
    b, design_b, _ = field_survey(spec, simulate(spec))
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(design_a, design_b)
