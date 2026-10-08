"""Portable, validated StudySpec: the single private source of synthetic-study truth.

A spec defines the population, latent constructs and their causal structure, the
questionnaire with its measurement model, the sampling/response process and the
intended stressors. The causal graph is derived from the declared coefficients
rather than stored twice, so it cannot disagree with the data-generating process.
Nothing in a spec is candidate-visible.
"""
from __future__ import annotations

from collections import defaultdict
import math
import re
from typing import Any

SPEC_VERSION = 'study-spec-0.1.0'
# The Survey API hardcodes this refusal response when building analysis bases.
REFUSAL = 'Prefer not to say'
LATENT_PREFIX = 'lat_'
ANSWER_TYPES = {'select_all', 'single', 'ordered_scale', 'rank'}
STRESSORS = {'small_subgroup', 'unweighted_sample', 'nonresponse_bias', 'routing_ambiguity',
             'denominator_ambiguity', 'measurement_error', 'item_nonresponse', 'unmeasured_construct',
             'proxy_measure', 'confounded_association'}
SCENARIO_KEYS = {'causal_group_variables', 'media_group_variables', 'media_outcome_blocks',
                 'purchase_outcome_blocks', 'purchase_values'}
_IDENTIFIER = re.compile(r'^[a-z][a-z0-9_]{0,63}$')
_STUDY_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')

Spec = dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('Invalid StudySpec: '+message)


def _number(value: Any, name: str, low: float = -math.inf, high: float = math.inf) -> None:
    _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high, name)


def _identifier(value: Any, name: str, latent: bool = False) -> None:
    _require(isinstance(value, str) and bool(_IDENTIFIER.match(value)), f'{name} must be a lowercase identifier')
    _require(value.startswith(LATENT_PREFIX) == latent,
             f'{name}: latent ids, and only latent ids, start with {LATENT_PREFIX!r}')


def _strings(values: Any, name: str, minimum: int = 1) -> None:
    _require(isinstance(values, list) and len(values) >= minimum and all(isinstance(v, str) and v for v in values)
             and len(set(values)) == len(values), f'{name} must be {minimum}+ unique nonempty strings')


def items(spec: Spec) -> list[tuple[dict, dict]]:
    """Questionnaire items in order, each with its block."""
    return [(block, item) for block in spec['questionnaire']['blocks'] for item in block['items']]


def item_values(block: dict, item: dict) -> list[Any]:
    """Recorded response categories, including the refusal where offered."""
    if block['answer_type'] == 'select_all':
        return [True]
    base = list(item['values']) if block['answer_type'] in {'single', 'rank'} else list(block['scale'])
    return base + ([REFUSAL] if item.get('refusal') else [])


def latent_order(spec: Spec) -> list[str]:
    """Topological latent order; raises on causal cycles."""
    latents = {l['id']: l for l in spec['latent_variables']}
    order, state = [], {}

    def visit(name: str) -> None:
        if state.get(name) == 'done':
            return
        _require(state.get(name) != 'active', 'latent causal structure contains a cycle at '+name)
        state[name] = 'active'
        for parent in latents[name].get('parents', {}):
            visit(parent)
        state[name] = 'done'
        order.append(name)
    for name in latents:
        visit(name)
    return order


def exogenous(latent: dict) -> bool:
    return not latent.get('effects') and not latent.get('parents')


def causal_edges(spec: Spec) -> list[dict[str, Any]]:
    """Directed DGP edges derived from nonzero coefficients."""
    result = []
    for demographic in spec['demographics']:
        for latent, coefficients in demographic.get('depends_on', {}).items():
            if any(c != 0 for c in coefficients):
                result.append({'from': latent, 'to': demographic['id'], 'kind': 'latent_to_demographic'})
    for latent in spec['latent_variables']:
        for demographic, coefficients in latent.get('effects', {}).items():
            if any(c != 0 for c in coefficients.values()):
                result.append({'from': demographic, 'to': latent['id'], 'kind': 'demographic_to_latent'})
        for parent, coefficient in latent.get('parents', {}).items():
            if coefficient != 0:
                result.append({'from': parent, 'to': latent['id'], 'kind': 'latent_to_latent'})
    for _, item in items(spec):
        result.append({'from': item['construct'], 'to': item['id'], 'kind': 'measurement'})
    return result


def descendants(spec: Spec, node: str) -> set[str]:
    children = defaultdict(set)
    for edge in causal_edges(spec):
        children[edge['from']].add(edge['to'])
    seen, frontier = set(), [node]
    while frontier:
        for child in children[frontier.pop()]:
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return seen


def realized_stressors(spec: Spec) -> set[str]:
    """Stressors that the declared process actually produces."""
    result = set()
    design, response = spec['sampling_design'], spec['response_process']
    measurement = spec['measurement_model']
    for demographic in spec['demographics']:
        probabilities = softmax(demographic['base_logits'])
        if min(probabilities) * design['issued_sample'] < 60:
            result.add('small_subgroup')
    if design['type'] == 'stratified' or response.get('effects') or response.get('latent_effects'):
        result.add('unweighted_sample')
    if response.get('effects') or response.get('latent_effects'):
        result.add('nonresponse_bias')
    blocks = spec['questionnaire']['blocks']
    has_select_all = any(b['answer_type'] == 'select_all' for b in blocks)
    if has_select_all and measurement['checkbox_encoding'] == 'true_or_blank':
        result.update({'routing_ambiguity', 'denominator_ambiguity'})
    if any(b.get('routing') and not b['routing']['documented'] for b in blocks):
        result.add('routing_ambiguity')
    if measurement['error_rate'] > 0:
        result.add('measurement_error')
    if measurement['item_nonresponse'] > 0 or measurement['refusal_rate'] > 0:
        result.add('item_nonresponse')
    if measurement.get('unmeasured_constructs'):
        result.add('unmeasured_construct')
    if measurement.get('proxies'):
        result.add('proxy_measure')
    if any(e['kind'] == 'latent_to_demographic' for e in causal_edges(spec)):
        result.add('confounded_association')
    return result


def softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    weights = [math.exp(v-top) for v in logits]
    return [w/sum(weights) for w in weights]


def validate_spec(spec: Spec) -> Spec:
    """Reject inconsistent specs before any population is generated."""
    required = {'spec_version', 'study_id', 'domain', 'label', 'seed', 'target_population', 'fieldwork',
                'population_size', 'demographics', 'latent_variables', 'questionnaire', 'measurement_model',
                'sampling_design', 'response_process', 'intended_stressors', 'scenario_bindings'}
    _require(isinstance(spec, dict) and set(spec) == required, 'keys must be exactly '+', '.join(sorted(required)))
    _require(spec['spec_version'] == SPEC_VERSION, 'unsupported spec_version')
    _require(isinstance(spec['study_id'], str) and bool(_STUDY_ID.match(spec['study_id'])), 'study_id')
    for name in ('domain', 'label'):
        _require(isinstance(spec[name], str) and bool(spec[name]), name)
    _require(type(spec['seed']) is int and spec['seed'] >= 0, 'seed')
    population = spec['target_population']
    _require(isinstance(population, dict) and set(population) == {'description', 'region', 'unit'}
             and all(isinstance(v, str) and v for v in population.values()), 'target_population')
    fieldwork = spec['fieldwork']
    _require(isinstance(fieldwork, dict) and set(fieldwork) == {'year', 'period', 'mode'}
             and type(fieldwork['year']) is int and 1990 <= fieldwork['year'] <= 2100
             and all(isinstance(fieldwork[k], str) and fieldwork[k] for k in ('period', 'mode')), 'fieldwork')
    _require(type(spec['population_size']) is int and 1000 <= spec['population_size'] <= 5_000_000, 'population_size')

    names: set[str] = set()
    latents = {}
    for latent in spec['latent_variables']:
        _identifier(latent.get('id'), 'latent id', latent=True)
        _require(latent['id'] not in names, 'duplicate id '+latent['id'])
        _require(set(latent) <= {'id', 'label', 'intercept', 'noise_sd', 'effects', 'parents'}, 'latent keys')
        _require(isinstance(latent.get('label'), str) and bool(latent['label']), 'latent label')
        _number(latent.get('intercept'), 'latent intercept')
        _number(latent.get('noise_sd'), 'latent noise_sd', 1e-6, 100)
        names.add(latent['id'])
        latents[latent['id']] = latent
    demographics = {}
    for demographic in spec['demographics']:
        _identifier(demographic.get('id'), 'demographic id')
        _require(demographic['id'] not in names, 'duplicate id '+demographic['id'])
        _require(set(demographic) <= {'id', 'label', 'values', 'base_logits', 'depends_on', 'contrasts', 'ordered'},
                 'demographic keys')
        _strings(demographic['values'], demographic['id']+' values', 2)
        _require(REFUSAL not in demographic['values'], 'demographics do not record refusals')
        _require(isinstance(demographic.get('label'), str) and bool(demographic['label']), 'demographic label')
        _require(isinstance(demographic['base_logits'], list) and len(demographic['base_logits']) == len(demographic['values']),
                 demographic['id']+' base_logits')
        for v in demographic['base_logits']:
            _number(v, 'base logit', -20, 20)
        for latent, coefficients in demographic.get('depends_on', {}).items():
            _require(latent in latents and exogenous(latents[latent]),
                     demographic['id']+' may depend only on exogenous latents')
            _require(isinstance(coefficients, list) and len(coefficients) == len(demographic['values']), 'depends_on length')
            for c in coefficients:
                _number(c, 'depends_on coefficient', -10, 10)
        _require(isinstance(demographic.get('contrasts'), list), demographic['id']+' contrasts')
        for contrast in demographic['contrasts']:
            _require(isinstance(contrast, dict) and len(contrast) == 2, 'each contrast compares two labelled groups')
            members = [v for values in contrast.values() for v in values]
            _require(all(isinstance(k, str) and k for k in contrast) and all(v in demographic['values'] for v in members)
                     and len(members) == len(set(members)) and all(contrast.values()),
                     demographic['id']+' contrast groups must be disjoint, nonempty recorded values')
        _require(type(demographic.get('ordered', False)) is bool, 'ordered flag')
        names.add(demographic['id'])
        demographics[demographic['id']] = demographic
    for latent in latents.values():
        for demographic, coefficients in latent.get('effects', {}).items():
            _require(demographic in demographics and isinstance(coefficients, dict)
                     and set(coefficients) <= set(demographics[demographic]['values']), latent['id']+' effects')
            for c in coefficients.values():
                _number(c, 'effect coefficient', -10, 10)
        for parent, coefficient in latent.get('parents', {}).items():
            _require(parent in latents and parent != latent['id'], latent['id']+' parents')
            _number(coefficient, 'parent coefficient', -10, 10)
    latent_order(spec)

    questionnaire = spec['questionnaire']
    _require(isinstance(questionnaire, dict) and set(questionnaire) == {'blocks'} and questionnaire['blocks'], 'questionnaire')
    block_ids, labels, single_values = set(), set(), {d: v['values'] for d, v in demographics.items()}
    constructs = set()
    for block in questionnaire['blocks']:
        _identifier(block.get('id'), 'block id')
        _require(block['id'] not in block_ids, 'duplicate block '+block['id'])
        _require(set(block) <= {'id', 'label', 'section', 'answer_type', 'routing', 'scale', 'items'}, 'block keys')
        _require(isinstance(block.get('label'), str) and ' ' in block['label'] and block['label'] not in labels,
                 'block label must be unique question wording')
        _require(isinstance(block.get('section'), str) and block['section'] and block['section'] != 'demographics',
                 'block section')
        _require(block.get('answer_type') in ANSWER_TYPES, 'answer_type')
        routing = block.get('routing')
        if routing is not None:
            _require(isinstance(routing, dict) and set(routing) == {'variable', 'values', 'documented'}
                     and routing['variable'] in single_values and type(routing['documented']) is bool
                     and isinstance(routing['values'], list) and routing['values']
                     and set(routing['values']) <= set(single_values[routing['variable']]),
                     block['id']+' routing must use a demographic or earlier single item')
        if block['answer_type'] == 'ordered_scale':
            _strings(block.get('scale'), block['id']+' scale', 3)
            _require(REFUSAL not in block['scale'], 'scale cannot include the refusal')
        else:
            _require('scale' not in block, 'only ordered scales declare a scale')
        _require(isinstance(block.get('items'), list) and block['items'], block['id']+' items')
        _require(block['answer_type'] == 'select_all' or len(block['items']) == 1 or block['answer_type'] == 'ordered_scale',
                 'single-choice blocks hold one item')
        for item in block['items']:
            _identifier(item.get('id'), 'item id')
            _require(item['id'] not in names, 'duplicate id '+item['id'])
            _require(isinstance(item.get('label'), str) and bool(item['label']), 'item label')
            _require(item.get('construct') in latents, item['id']+' construct must be a latent')
            constructs.add(item['construct'])
            kind = block['answer_type']
            if kind == 'select_all':
                _require(set(item) == {'id', 'label', 'construct', 'intercept', 'loading'}, item['id']+' keys')
                _number(item['intercept'], 'intercept', -20, 20)
                _number(item['loading'], 'loading', -20, 20)
            elif kind in {'single', 'rank'}:
                _require(set(item) == {'id', 'label', 'construct', 'values', 'value_intercepts', 'value_loadings', 'refusal'},
                         item['id']+' keys')
                _strings(item['values'], item['id']+' values', 2)
                _require(REFUSAL not in item['values'], 'declare refusals with refusal=true')
                for field in ('value_intercepts', 'value_loadings'):
                    _require(isinstance(item[field], list) and len(item[field]) == len(item['values']), item['id']+' '+field)
                    for v in item[field]:
                        _number(v, field, -20, 20)
                _require(type(item['refusal']) is bool, 'refusal flag')
                if kind == 'single':
                    single_values[item['id']] = item['values']
            else:
                _require(set(item) == {'id', 'label', 'construct', 'loading', 'cutpoints', 'refusal'}, item['id']+' keys')
                _number(item['loading'], 'loading', -20, 20)
                cuts = item['cutpoints']
                _require(isinstance(cuts, list) and len(cuts) == len(block['scale'])-1
                         and all(type(c) in (int, float) and math.isfinite(c) for c in cuts)
                         and all(a < b for a, b in zip(cuts, cuts[1:])), item['id']+' cutpoints must increase')
                _require(type(item['refusal']) is bool, 'refusal flag')
            names.add(item['id'])
        block_ids.add(block['id'])
        labels.add(block['label'])

    measurement = spec['measurement_model']
    _require(isinstance(measurement, dict) and set(measurement) == {'error_rate', 'item_nonresponse', 'refusal_rate',
             'checkbox_encoding', 'proxies', 'unmeasured_constructs'}, 'measurement_model keys')
    for name in ('error_rate', 'item_nonresponse', 'refusal_rate'):
        _number(measurement[name], name, 0, 0.49)
    _require(measurement['checkbox_encoding'] in {'true_or_blank', 'true_false'}, 'checkbox_encoding')
    item_ids = {item['id'] for _, item in items(spec)}
    for proxy in measurement['proxies']:
        _require(isinstance(proxy, dict) and set(proxy) == {'concept', 'item', 'rationale'} and proxy['item'] in item_ids
                 and all(isinstance(proxy[k], str) and proxy[k] for k in ('concept', 'rationale')), 'proxy')
    for latent in measurement['unmeasured_constructs']:
        _require(latent in latents and latent not in constructs, 'unmeasured constructs must have no survey item')

    design = spec['sampling_design']
    _require(isinstance(design, dict) and design.get('type') in {'srs', 'stratified'}, 'sampling_design type')
    _require(type(design.get('issued_sample')) is int and 50 <= design['issued_sample'] <= spec['population_size'],
             'issued_sample')
    if design['type'] == 'stratified':
        _require(set(design) == {'type', 'issued_sample', 'strata_variable', 'allocation'}, 'stratified design keys')
        _require(design['strata_variable'] in demographics, 'strata_variable')
        allocation = design['allocation']
        _require(isinstance(allocation, dict) and set(allocation) == set(demographics[design['strata_variable']]['values'])
                 and all(type(v) in (int, float) and v > 0 for v in allocation.values())
                 and abs(sum(allocation.values())-1) < 1e-9, 'allocation shares must cover strata and sum to 1')
    else:
        _require(set(design) == {'type', 'issued_sample'}, 'srs design keys')

    response = spec['response_process']
    _require(isinstance(response, dict) and set(response) == {'base_logit', 'effects', 'latent_effects'}, 'response_process keys')
    _number(response['base_logit'], 'base_logit', -10, 10)
    for demographic, coefficients in response['effects'].items():
        _require(demographic in demographics and set(coefficients) <= set(demographics[demographic]['values']),
                 'response effects')
    for latent, coefficient in response['latent_effects'].items():
        _require(latent in latents, 'response latent_effects')
        _number(coefficient, 'response latent effect', -10, 10)

    stressors = spec['intended_stressors']
    _require(isinstance(stressors, list) and set(stressors) <= STRESSORS, 'unknown intended stressor')
    missing = set(stressors) - realized_stressors(spec)
    _require(not missing, 'intended stressors not produced by the declared process: '+', '.join(sorted(missing)))

    bindings = spec['scenario_bindings']
    _require(isinstance(bindings, dict) and set(bindings) <= SCENARIO_KEYS, 'scenario_bindings keys')
    for name in ('causal_group_variables', 'media_group_variables'):
        _require(set(bindings.get(name, [])) <= set(demographics), name)
    for name in ('media_outcome_blocks', 'purchase_outcome_blocks'):
        _require(set(bindings.get(name, [])) <= block_ids, name)
    for item, values in bindings.get('purchase_values', {}).items():
        _require(item in single_values and item not in demographics and set(values) <= set(single_values[item]),
                 'purchase_values')
    return spec
