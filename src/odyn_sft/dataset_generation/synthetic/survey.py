"""Survey process and the candidate-visible/benchmark-only artefact boundary.

population -> issued sample -> unit response -> measurement error -> routing ->
item nonresponse -> recorded survey. Candidate artefacts are written only from the
recorded survey and disclosed design facts; everything else goes to ``hidden``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...survey_api import normalize_data
from ...survey_metadata.interpretation import DENOMINATOR_NOTE, RESPONSE_METADATA_NOTE
from ...common.storage import digest, write_json
from .population import Population, _sigmoid, group_shares, population_frame, recorded_values, routing_eligible, selected, simulate
from .spec import REFUSAL, Spec, causal_edges, items, latent_order, validate_spec

SALT_SURVEY = 3
METADATA_VERSION = 'synthetic-study-0.1.0'
VISIBLE_FILES = ('respondents.parquet', 'metadata.json', 'codebook.json', 'study_description.json')
HIDDEN_FILES = ('study_spec.json', 'population_truth.json', 'dgp.json', 'sampling_truth.json',
                'measurement_truth.json', 'population.parquet', 'respondent_design.parquet')


def _issue(spec: Spec, population: Population, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Issued population indices and their inclusion probabilities."""
    design, n = spec['sampling_design'], population.size
    if design['type'] == 'srs':
        issued = np.sort(rng.choice(n, design['issued_sample'], replace=False))
        return issued, np.full(len(issued), design['issued_sample']/n)
    variable = design['strata_variable']
    labels = recorded_values(spec, variable)
    chosen, probabilities = [], []
    for code, value in enumerate(labels):
        members = np.flatnonzero(population.demographics[variable] == code)
        take = min(len(members), max(1, round(design['issued_sample']*design['allocation'][value])))
        chosen.append(np.sort(rng.choice(members, take, replace=False)))
        probabilities.append(np.full(take, take/len(members)))
    return np.concatenate(chosen), np.concatenate(probabilities)


def _propensity(spec: Spec, population: Population, units: np.ndarray) -> np.ndarray:
    response = spec['response_process']
    logit = np.full(len(units), float(response['base_logit']))
    for demographic, coefficients in response['effects'].items():
        table = np.array([coefficients.get(v, 0.0) for v in recorded_values(spec, demographic)])
        logit += table[population.demographics[demographic][units]]
    for latent, coefficient in response['latent_effects'].items():
        logit += coefficient*population.latents[latent][units]
    return _sigmoid(logit)


def field_survey(spec: Spec, population: Population) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Recorded responses, hidden respondent design rows and measurement truth."""
    rng = np.random.default_rng([spec['seed'], SALT_SURVEY])
    issued, inclusion = _issue(spec, population, rng)
    propensity = _propensity(spec, population, issued)
    responded = rng.random(len(issued)) < propensity
    order = rng.permutation(int(responded.sum()))
    units = issued[responded][order]
    inclusion, propensity = inclusion[responded][order], propensity[responded][order]
    n = len(units)
    if n < 30:
        raise ValueError('Synthetic survey produced fewer than 30 respondents; enlarge issued_sample')
    measurement = spec['measurement_model']
    columns: dict[str, Any] = {'respondent_id': [f"{spec['study_id']}_r{i:05d}" for i in range(1, n+1)]}
    for demographic in spec['demographics']:
        columns[demographic['id']] = np.asarray(demographic['values'], dtype=object)[population.demographics[demographic['id']][units]]
    report: dict[str, Any] = {'error_rate': measurement['error_rate'], 'item_nonresponse': measurement['item_nonresponse'],
                              'refusal_rate': measurement['refusal_rate'], 'checkbox_encoding': measurement['checkbox_encoding'],
                              'items': {}, 'blocks': {}}
    for block in spec['questionnaire']['blocks']:
        routing = block.get('routing')
        shown = np.ones(n, dtype=bool) if routing is None else np.isin(
            np.asarray(columns[routing['variable']], dtype=object), routing['values'])
        kind = block['answer_type']
        skipped_block = rng.random(n) < measurement['item_nonresponse'] if kind == 'select_all' else np.zeros(n, dtype=bool)
        answered = shown & ~skipped_block
        any_selected = np.zeros(n, dtype=bool)
        for item in block['items']:
            truth = population.responses[item['id']][units]
            error = rng.random(n) < measurement['error_rate']
            if kind == 'select_all':
                recorded = np.where(error, ~truth, truth)
                any_selected |= recorded & answered
                if measurement['checkbox_encoding'] == 'true_false':
                    values = pd.array(np.where(answered, recorded, False), dtype='boolean')
                else:
                    values = pd.array(np.where(answered & recorded, True, False), dtype='boolean')
                    values[~(answered & recorded)] = pd.NA
                values[~answered] = pd.NA
                columns[item['id']] = values
                report['items'][item['id']] = {'flipped': int((error & answered).sum())}
                continue
            labels = recorded_values(spec, item['id'])
            codes = np.where(error, rng.integers(len(labels), size=n), truth)
            recorded = np.asarray(labels, dtype=object)[codes]
            skipped = rng.random(n) < measurement['item_nonresponse']
            refused = (rng.random(n) < measurement['refusal_rate']) & item['refusal']
            recorded = np.where(refused, REFUSAL, recorded).astype(object)
            recorded[~shown | skipped] = None
            columns[item['id']] = recorded
            report['items'][item['id']] = {'perturbed': int((error & shown & ~skipped & ~refused).sum()),
                                           'missing': int((~shown | skipped).sum()), 'refused': int((refused & shown & ~skipped).sum())}
        report['blocks'][block['id']] = {'shown': int(shown.sum()), 'not_shown': int((~shown).sum()),
                                         'block_skipped': int((shown & skipped_block).sum())}
        if kind == 'select_all':
            # The hidden denominator gap: shown respondents who selected nothing.
            silent = answered & ~any_selected
            report['blocks'][block['id']]['answered_none_selected'] = int(silent.sum())
            report['blocks'][block['id']]['excluded_from_api_base'] = int(silent.sum()) if measurement['checkbox_encoding'] == 'true_or_blank' else 0
    observed = pd.DataFrame(columns)
    design = pd.DataFrame({'respondent_id': columns['respondent_id'], 'population_index': units,
                           'inclusion_probability': inclusion, 'response_propensity': propensity,
                           'design_weight': 1/inclusion, 'oracle_weight': 1/(inclusion*propensity)})
    report['unit_response'] = {'issued': int(len(issued)), 'responded': n, 'response_rate': n/len(issued)}
    return observed, design, report


def _routing_text(spec: Spec, block: dict) -> str | None:
    routing = block.get('routing')
    if routing is None or not routing['documented']:
        return None
    labels = {d['id']: d['label'] for d in spec['demographics']} | {i['id']: i['label'] for _, i in items(spec)}
    answers = ' or '.join(repr(v) for v in routing['values'])
    return f"Asked only of respondents answering {answers} to {labels[routing['variable']]!r}."


def build_metadata(spec: Spec, observed: pd.DataFrame) -> dict[str, Any]:
    """Survey API metadata containing only recorded-survey and disclosed design facts."""
    measurement, design = spec['measurement_model'], spec['sampling_design']
    blank = measurement['checkbox_encoding'] == 'true_or_blank'
    cards = []
    for d in spec['demographics']:
        cards.append({'id': d['id'], 'source_column': d['id'], 'label': d['label'], 'label_origin': 'codebook',
                      'block_id': d['id'], 'section': 'demographics', 'answer_type': 'single', 'values': list(d['values']),
                      'ordered_values': list(d['values']) if d.get('ordered') else None,
                      'order_source': 'codebook' if d.get('ordered') else 'none',
                      'ordering_basis': 'codebook_declared' if d.get('ordered') else None, 'unranked_values': [],
                      'denominator_rule': 'answered_item_excluding_refusals', 'flags': ['unweighted'],
                      'answered_n': int(observed[d['id']].notna().sum()), 'review_status': 'synthetic_generated'})
    for block, item in items(spec):
        kind = block['answer_type']
        flags = ['unweighted']
        routing = block.get('routing')
        if kind == 'select_all' and (blank or (routing is not None and not routing['documented'])):
            flags.append('eligibility_unknown')
        if kind == 'ordered_scale':
            flags.append('acquiescence_risk')
        if kind == 'rank':
            flags.append('rank_direction_unknown')
        values = ([True] if blank else [True, False]) if kind == 'select_all' else list(item['values'])+([REFUSAL] if item['refusal'] else []) \
            if kind in {'single', 'rank'} else list(block['scale'])+([REFUSAL] if item['refusal'] else [])
        ordered = list(block['scale']) if kind == 'ordered_scale' else None
        cards.append({'id': item['id'], 'source_column': item['id'], 'label': item['label'], 'label_origin': 'codebook',
                      'block_id': block['label'], 'section': block['section'], 'answer_type': kind, 'values': values,
                      'ordered_values': ordered, 'order_source': 'codebook' if ordered else 'none',
                      'ordering_basis': 'codebook_declared' if ordered else None,
                      'unranked_values': [REFUSAL] if ordered and item['refusal'] else [],
                      'denominator_rule': 'observed_block_respondents' if kind == 'select_all' else 'answered_item_excluding_refusals',
                      'flags': flags, 'answered_n': int(observed[item['id']].notna().sum()), 'review_status': 'synthetic_generated'})
    population = spec['target_population']
    if design['type'] == 'stratified':
        strata = {d['id']: d['label'] for d in spec['demographics']}[design['strata_variable']]
        sampling = f"{spec['fieldwork']['mode'].capitalize()} sample stratified by {strata.lower()} with unequal allocation; no survey weights supplied."
    else:
        sampling = f"{spec['fieldwork']['mode'].capitalize()} sample; no survey weights supplied."
    sampling += ' Response propensities are not documented.'
    routing_notes = [f"{block['label']} {text}" for block in spec['questionnaire']['blocks'] if (text := _routing_text(spec, block))]
    limitations = ['unweighted', 'sample_only', 'no_causal_identification']
    if blank or any(b.get('routing') and not b['routing']['documented'] for b in spec['questionnaire']['blocks']):
        limitations.append('true_routing_unknown')
    missingness = ('Reasons for missing cells are not recorded. Checkbox blanks cannot distinguish not selected from not shown.'
                   if blank else 'Reasons for missing cells are not recorded. Checkbox options record True when selected and False when shown but not selected; blank means not shown or not answered.')
    study = {'label': spec['label'], 'period': spec['fieldwork']['period'], 'n': int(len(observed)), 'weights': None,
             'target_population': population['description'],
             'scope': f"single {spec['fieldwork']['mode']} sample of {population['description']}; cross-sectional self-report; synthetic study",
             'sampling_note': sampling, 'missingness_note': missingness,
             'response_metadata_note': RESPONSE_METADATA_NOTE, 'denominator_note': DENOMINATOR_NOTE,
             'limitations': limitations}
    if routing_notes:
        study['routing_note'] = ' '.join(routing_notes)
    metadata = {'version': METADATA_VERSION, 'study_id': spec['study_id'], 'study': study, 'variables': cards, 'excluded': [],
                'supported_generation': ['select_all_proportion', 'agreement_top_two', 'group_difference', 'majority_threshold', 'scope_notes'],
                'unsupported_generation': ['adjusted_compare', 'mean_score', 'trend', 'association', 'cross_study', 'weighted_estimates'],
                'review_status': 'synthetic_generated_pending_review'}
    metadata['hash'] = digest(metadata)
    return metadata


def codebook(spec: Spec, metadata: dict[str, Any]) -> dict[str, Any]:
    blocks = []
    for block in spec['questionnaire']['blocks']:
        blocks.append({'question': block['label'], 'section': block['section'], 'answer_type': block['answer_type'],
                       'routing': _routing_text(spec, block),
                       'variables': [{'id': c['id'], 'label': c['label'], 'values': c['values']}
                                     for c in metadata['variables'] if c['block_id'] == block['label']]})
    return {'study_id': spec['study_id'], 'demographics': [{'id': c['id'], 'label': c['label'], 'values': c['values']}
            for c in metadata['variables'] if c['section'] == 'demographics'], 'blocks': blocks}


def population_truth(spec: Spec, population: Population) -> dict[str, Any]:
    """Marginal truth; per-case contrasts are added to the private case gold."""
    marginals = {}
    for d in spec['demographics']:
        counts = np.bincount(population.demographics[d['id']], minlength=len(d['values']))
        marginals[d['id']] = {v: int(c) for v, c in zip(d['values'], counts)}
    shares = {}
    for block, item in items(spec):
        eligible = routing_eligible(population, block)
        if block['answer_type'] == 'select_all':
            shares[item['id']] = {'routed_population_n': int(eligible.sum()),
                                  'selection': float(selected(population, item['id'], 'selection')[eligible].mean())}
        else:
            labels = recorded_values(spec, item['id'])
            counts = np.bincount(population.responses[item['id']][eligible], minlength=len(labels))
            shares[item['id']] = {'routed_population_n': int(eligible.sum()),
                                  'distribution': {str(v): float(c/eligible.sum()) for v, c in zip(labels, counts)}}
    return {'population_size': population.size, 'demographic_counts': marginals, 'item_truth': shares,
            'definition': 'finite-population true responses among routing-eligible members, before sampling, nonresponse and measurement error'}


def generate_study(spec: Spec, public: Path, hidden: Path) -> tuple[pd.DataFrame, dict[str, Any], Population]:
    """Write visible and hidden study artefacts; return normalised data and metadata."""
    validate_spec(spec)
    public, hidden = Path(public), Path(hidden)
    if public.exists() or hidden.exists():
        raise ValueError('Use new study output directories')
    population = simulate(spec)
    observed, design, measurement = field_survey(spec, population)
    metadata = build_metadata(spec, observed)
    data = normalize_data(observed, metadata)
    public.mkdir(parents=True)
    hidden.mkdir(parents=True)
    observed.to_parquet(public/'respondents.parquet', index=False)
    write_json(public/'metadata.json', metadata)
    write_json(public/'codebook.json', codebook(spec, metadata))
    write_json(public/'study_description.json', {'study_id': spec['study_id'], **metadata['study']})
    write_json(hidden/'study_spec.json', spec)
    write_json(hidden/'dgp.json', {'latent_order': latent_order(spec), 'causal_edges': causal_edges(spec),
        'unmeasured_constructs': spec['measurement_model']['unmeasured_constructs'],
        'proxies': spec['measurement_model']['proxies'], 'coefficients': 'see study_spec.json'})
    write_json(hidden/'population_truth.json', population_truth(spec, population))
    respondents = Counter(observed[spec['sampling_design'].get('strata_variable', spec['demographics'][0]['id'])])
    write_json(hidden/'sampling_truth.json', {'design': spec['sampling_design'], 'response_process': spec['response_process'],
        **measurement.pop('unit_response'), 'respondents_by_stratum': dict(sorted(respondents.items())),
        'weights_file': 'respondent_design.parquet',
        'weights_note': 'design_weight=1/inclusion probability; oracle_weight also divides by the true response propensity. Neither is candidate-visible.'})
    write_json(hidden/'measurement_truth.json', measurement)
    population_frame(population).to_parquet(hidden/'population.parquet', index=False)
    design.to_parquet(hidden/'respondent_design.parquet', index=False)
    return data, metadata, population


def load_study(public: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    from ...common.storage import read_json
    metadata = read_json(Path(public)/'metadata.json')
    return normalize_data(pd.read_parquet(Path(public)/'respondents.parquet'), metadata), metadata


__all__ = ['generate_study', 'load_study', 'field_survey', 'build_metadata', 'group_shares', 'VISIBLE_FILES', 'HIDDEN_FILES']
