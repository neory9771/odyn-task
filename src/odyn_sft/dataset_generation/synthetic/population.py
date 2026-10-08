"""Finite synthetic population, its latent truth and exact interventional quantities.

Truth is defined at three separate levels:

* item truth: the finite-population share giving a response, among members
  eligible under the questionnaire routing, before sampling, nonresponse and
  measurement error;
* causal truth: whether a demographic is a DGP ancestor of the item's construct,
  and the average effect of setting group membership (an intervention on the
  whole population, with exogenous noise held fixed);
* observed evidence, which only the survey process and Survey API produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .spec import Spec, descendants, exogenous, items, latent_order

SALT_POPULATION = 1
SALT_RESPONSES = 2


@dataclass
class Population:
    spec: Spec
    demographics: dict[str, np.ndarray]  # integer codes into each demographic's values
    noise: dict[str, np.ndarray]         # standard-normal exogenous noise per latent
    latents: dict[str, np.ndarray]
    responses: dict[str, np.ndarray]     # true item responses: bool or integer codes
    _intervened: dict[tuple[str, int], dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    @property
    def size(self) -> int:
        return self.spec['population_size']


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-x))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits-logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights/weights.sum(axis=1, keepdims=True)


def _categorical(probabilities: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    cumulative = probabilities.cumsum(axis=1)
    cumulative[:, -1] = 1.0
    return (uniform[:, None] > cumulative).sum(axis=1)


def _endogenous(spec: Spec, demographics: dict[str, np.ndarray], noise: dict[str, np.ndarray],
                latents: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute non-exogenous latents in causal order from fixed noise."""
    result = dict(latents)
    cards = {d['id']: d for d in spec['demographics']}
    by_id = {l['id']: l for l in spec['latent_variables']}
    for name in latent_order(spec):
        latent = by_id[name]
        if exogenous(latent):
            continue
        value = latent['intercept'] + latent['noise_sd']*noise[name]
        for demographic, coefficients in latent.get('effects', {}).items():
            table = np.array([coefficients.get(v, 0.0) for v in cards[demographic]['values']])
            value = value + table[demographics[demographic]]
        for parent, coefficient in latent.get('parents', {}).items():
            value = value + coefficient*result[parent]
        result[name] = value
    return result


def category_probabilities(block: dict, item: dict, construct: np.ndarray) -> np.ndarray:
    """P(response category | construct); select-all returns one selection column."""
    kind = block['answer_type']
    if kind == 'select_all':
        return _sigmoid(item['intercept'] + item['loading']*construct)[:, None]
    if kind in {'single', 'rank'}:
        logits = np.asarray(item['value_intercepts'])[None, :] + np.asarray(item['value_loadings'])[None, :]*construct[:, None]
        return _softmax(logits)
    cuts = np.asarray(item['cutpoints'])
    at_most = _sigmoid(cuts[None, :] - item['loading']*construct[:, None])
    bounds = np.concatenate([np.zeros((len(construct), 1)), at_most, np.ones((len(construct), 1))], axis=1)
    return np.diff(bounds, axis=1)


def simulate(spec: Spec) -> Population:
    """Deterministic population from the spec seed alone."""
    n = spec['population_size']
    rng = np.random.default_rng([spec['seed'], SALT_POPULATION])
    by_id = {l['id']: l for l in spec['latent_variables']}
    noise = {name: rng.standard_normal(n) for name in latent_order(spec)}
    latents = {name: by_id[name]['intercept'] + by_id[name]['noise_sd']*noise[name]
               for name in latent_order(spec) if exogenous(by_id[name])}
    demographics = {}
    for demographic in spec['demographics']:
        logits = np.tile(np.asarray(demographic['base_logits'], dtype=float), (n, 1))
        for latent, coefficients in demographic.get('depends_on', {}).items():
            logits = logits + latents[latent][:, None]*np.asarray(coefficients)[None, :]
        demographics[demographic['id']] = _categorical(_softmax(logits), rng.random(n))
    latents = _endogenous(spec, demographics, noise, latents)
    responses = {}
    response_rng = np.random.default_rng([spec['seed'], SALT_RESPONSES])
    for block, item in items(spec):
        probabilities = category_probabilities(block, item, latents[item['construct']])
        uniform = response_rng.random(n)
        responses[item['id']] = (uniform < probabilities[:, 0]) if block['answer_type'] == 'select_all' else _categorical(probabilities, uniform)
    return Population(spec, demographics, noise, latents, responses)


def item_lookup(spec: Spec) -> dict[str, tuple[dict, dict]]:
    return {item['id']: (block, item) for block, item in items(spec)}


def recorded_values(spec: Spec, variable: str) -> list[Any]:
    """Category labels indexed by population response codes."""
    demographics = {d['id']: d for d in spec['demographics']}
    if variable in demographics:
        return demographics[variable]['values']
    block, item = item_lookup(spec)[variable]
    return [True] if block['answer_type'] == 'select_all' else item['values'] if block['answer_type'] in {'single', 'rank'} else block['scale']


def true_labels(population: Population, variable: str) -> np.ndarray:
    """True labels for a demographic or a single/ordered item."""
    spec = population.spec
    codes = population.demographics.get(variable)
    if codes is None:
        codes = population.responses[variable]
    return np.asarray(recorded_values(spec, variable), dtype=object)[codes]


def routing_eligible(population: Population, block: dict) -> np.ndarray:
    routing = block.get('routing')
    if routing is None:
        return np.ones(population.size, dtype=bool)
    return np.isin(true_labels(population, routing['variable']), routing['values'])


def _measure_index(spec: Spec, variable: str, operation: str) -> tuple[dict, dict, list[int]]:
    """Category columns counted by a Survey API measure operation."""
    block, item = item_lookup(spec)[variable]
    if operation == 'selection':
        return block, item, [0]
    if operation == 'top_box_2':
        k = len(block['scale'])
        return block, item, [k-2, k-1]
    if operation.startswith('value='):
        value = operation[len('value='):]
        return block, item, [item['values'].index(value)]
    raise ValueError('Unsupported measure operation for truth: '+operation)


def selected(population: Population, variable: str, operation: str) -> np.ndarray:
    block, _, columns = _measure_index(population.spec, variable, operation)
    response = population.responses[variable]
    return response.astype(bool) if block['answer_type'] == 'select_all' else np.isin(response, columns)


def group_shares(population: Population, variable: str, operation: str, group_variable: str,
                 groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Finite-population item truth for each labelled group."""
    block, _, _ = _measure_index(population.spec, variable, operation)
    base = routing_eligible(population, block)
    hit = selected(population, variable, operation)
    labels = true_labels(population, group_variable)
    rows = []
    for label, values in groups.items():
        mask = base & np.isin(labels, values)
        n = int(mask.sum())
        rows.append({'group': label, 'population_n': n, 'share': float(hit[mask].mean()) if n else None})
    return rows


def _intervened(population: Population, demographic: str, code: int) -> dict[str, np.ndarray]:
    key = (demographic, code)
    if key not in population._intervened:
        forced = dict(population.demographics)
        forced[demographic] = np.full(population.size, code)
        exogenous_values = {k: v for k, v in population.latents.items()
                            if exogenous({l['id']: l for l in population.spec['latent_variables']}[k])}
        population._intervened[key] = _endogenous(population.spec, forced, population.noise, exogenous_values)
    return population._intervened[key]


def interventional_share(population: Population, variable: str, operation: str, demographic: str,
                         values: list[str]) -> float:
    """E[P(response) | do(demographic in group)], mixing values by their population shares.

    Routing is ignored: the causal estimand concerns everyone's potential response,
    not the realised routed base.
    """
    spec = population.spec
    block, item, columns = _measure_index(spec, variable, operation)
    labels = recorded_values(spec, demographic)
    counts = np.array([(population.demographics[demographic] == labels.index(v)).sum() for v in values], dtype=float)
    weights = counts/counts.sum() if counts.sum() else np.full(len(values), 1/len(values))
    total = 0.0
    for weight, value in zip(weights, values):
        latents = _intervened(population, demographic, labels.index(value))
        probabilities = category_probabilities(block, item, latents[item['construct']])
        total += float(weight)*float(probabilities[:, columns].sum(axis=1).mean())
    return total


def causal_truth(population: Population, variable: str, operation: str, demographic: str,
                 groups: dict[str, list[str]]) -> dict[str, Any]:
    spec = population.spec
    _, item = item_lookup(spec)[variable]
    path = item['construct'] in descendants(spec, demographic)
    confounders = sorted({l for d in spec['demographics'] if d['id'] == demographic
                          for l, c in d.get('depends_on', {}).items() if any(c)} &
                         {a for a in (l['id'] for l in spec['latent_variables']) if item['construct'] in descendants(spec, a)})
    (a, va), (b, vb) = list(groups.items())
    effect = interventional_share(population, variable, operation, demographic, va) - \
        interventional_share(population, variable, operation, demographic, vb)
    media = [l for l in spec['measurement_model']['unmeasured_constructs']
             if l in descendants(spec, demographic) and item['construct'] in descendants(spec, l)]
    return {'demographic_is_causal_ancestor': path, 'average_causal_effect': effect,
            'contrast': f'do({a}) - do({b})', 'confounding_latents': confounders,
            'mediating_unmeasured_constructs': media,
            'estimand': 'difference in mean response probability under intervention on group membership; routing ignored'}


def population_frame(population: Population) -> pd.DataFrame:
    """Hidden population table: demographics, true responses and latents."""
    spec = population.spec
    columns: dict[str, Any] = {'population_index': np.arange(population.size)}
    for demographic in spec['demographics']:
        columns[demographic['id']] = pd.Categorical.from_codes(population.demographics[demographic['id']], demographic['values'])
    for block, item in items(spec):
        response = population.responses[item['id']]
        columns[item['id']] = response if block['answer_type'] == 'select_all' else \
            pd.Categorical.from_codes(response, recorded_values(spec, item['id']))
    for name, value in population.latents.items():
        columns[name] = value
    return pd.DataFrame(columns)
