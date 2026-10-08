"""Seeded StudySpec templates.

A domain fixes the questionnaire and variable identifiers, so several studies of
one domain share analysis families ("familiar analysis type in a new study").
Effect signs and sizes, zero effects, confounding, design and stressors are drawn
per study, so no claim direction is predictable from the domain alone.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from .spec import SPEC_VERSION, Spec, realized_stressors, validate_spec

AGE = ['18-24', '25-34', '35-44', '45-54', '55-64', '65 and above']
AGE_CONTRASTS = [{'18-34': ['18-24', '25-34'], '55+': ['55-64', '65 and above']},
                 {'Under 45': ['18-24', '25-34', '35-44'], '45+': ['45-54', '55-64', '65 and above']}]
GENDER_CONTRASTS = [{'Female': ['Female'], 'Male': ['Male']}]
AGREEMENT = ['Strongly disagree', 'Disagree', 'Neither agree nor disagree', 'Agree', 'Strongly agree']
REGIONS = ['Northland', 'Westmark', 'Eastvale', 'Southmere']


class _Draw:
    """Rounded seeded draws keep specs readable and byte-stable."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def signed(self, low: float = .25, high: float = .9, p_zero: float = .25) -> float:
        if self.rng.random() < p_zero:
            return 0.0
        return round(float(self.rng.choice([-1, 1]) * self.rng.uniform(low, high)), 3)

    def uniform(self, low: float, high: float) -> float:
        return round(float(self.rng.uniform(low, high)), 3)

    def choice(self, values: list) -> Any:
        return values[int(self.rng.integers(len(values)))]

    def trend(self, values: list[str], p_zero: float = .25) -> dict[str, float]:
        """Monotone ordinal effect, centred so the slope sign is the direction."""
        slope = self.signed(.08, .3, p_zero)
        middle = (len(values)-1)/2
        return {v: round(slope*(i-middle), 3) for i, v in enumerate(values)}

    def categorical(self, values: list[str], p_zero: float = .3) -> dict[str, float]:
        return {v: (0.0 if i == 0 else self.signed(.2, .8, p_zero)) for i, v in enumerate(values)}

    def logits(self, base: list[float], jitter: float = .15) -> list[float]:
        return [round(b+float(self.rng.normal(0, jitter)), 3) for b in base]


def _common(draw: _Draw, study_id: str, domain: str, label: str, seed: int, population_size: int,
            unit: str, region: str) -> Spec:
    year = draw.choice([2023, 2024, 2025])
    return {'spec_version': SPEC_VERSION, 'study_id': study_id, 'domain': domain, 'label': label, 'seed': seed,
            'target_population': {'description': f'{unit}s aged 18 and over living in {region}', 'region': region,
                                  'unit': unit},
            'fieldwork': {'year': year, 'period': f'{year}-05-02..{year}-05-16', 'mode': 'online panel'},
            'population_size': population_size}


def _design(draw: _Draw, strata: str, values: list[str], oversample: str) -> dict:
    issued = draw.choice([3000, 4500, 6000])
    if draw.rng.random() < .5:
        return {'type': 'srs', 'issued_sample': issued}
    raw = {v: (2.5 if v == oversample else 1.0)*draw.uniform(.8, 1.2) for v in values}
    total = sum(raw.values())
    shares = {v: round(x/total, 6) for v, x in raw.items()}
    # Make the shares sum to exactly one after rounding.
    last = values[-1]
    shares[last] = round(1-sum(x for v, x in shares.items() if v != last), 6)
    return {'type': 'stratified', 'issued_sample': issued, 'strata_variable': strata, 'allocation': shares}


def _measurement(draw: _Draw, proxies: list[dict], unmeasured: list[str]) -> dict:
    return {'error_rate': draw.choice([0.0, 0.02, 0.05]), 'item_nonresponse': draw.choice([0.0, 0.02, 0.04]),
            'refusal_rate': draw.choice([0.0, 0.02]), 'checkbox_encoding': draw.choice(['true_or_blank', 'true_false']),
            'proxies': proxies, 'unmeasured_constructs': unmeasured}


def _finish(spec: Spec) -> Spec:
    # Detach shared module constants so callers can edit one spec safely.
    spec = copy.deepcopy(spec)
    spec['intended_stressors'] = sorted(realized_stressors(spec))
    return validate_spec(spec)


def brand_tracking(study_id: str, seed: int, population_size: int = 200_000) -> Spec:
    d = _Draw(seed)
    region_name = d.choice(REGIONS)
    regions = ['North', 'South', 'East', 'West', 'Islands']
    settlements = ['Urban', 'Suburban', 'Rural']
    spec = _common(d, study_id, 'brand_tracking', f'Synthetic snack brand tracker ({region_name})', seed,
                   population_size, 'adult', region_name)
    spec['demographics'] = [
        {'id': 'age_band', 'label': 'Age band', 'values': AGE, 'base_logits': d.logits([0, .2, .1, .1, .05, .1]),
         'contrasts': AGE_CONTRASTS, 'ordered': True},
        {'id': 'gender', 'label': 'Gender', 'values': ['Female', 'Male'], 'base_logits': d.logits([0, 0]),
         'contrasts': GENDER_CONTRASTS},
        # Islands is deliberately rare: small bases and suppression.
        {'id': 'region', 'label': 'Region of residence', 'values': regions, 'base_logits': [0, -.1, .1, 0, -3.6],
         'depends_on': {'lat_u_lifestyle': [0, d.signed(.2, .6, .3), 0, d.signed(.2, .6, .3), 0]},
         'contrasts': [{'North': ['North'], 'South': ['South']}, {'Islands': ['Islands'], 'East': ['East']}]},
        {'id': 'settlement', 'label': 'Settlement type', 'values': settlements, 'base_logits': d.logits([.3, .4, 0]),
         'depends_on': {'lat_u_lifestyle': [d.signed(.3, .8, .2), 0, d.signed(.3, .8, .2)]},
         'contrasts': [{'Urban': ['Urban'], 'Rural': ['Rural']}, {'Suburban': ['Suburban'], 'Urban': ['Urban']}]},
    ]
    aware = {}
    for brand in 'abcd':
        aware[brand] = {'id': f'lat_aware_{brand}', 'label': f'Awareness of Brand {brand.upper()}', 'intercept': 0.0,
                        'noise_sd': 1.0, 'effects': {'age_band': d.trend(AGE), 'region': d.categorical(regions)},
                        'parents': {'lat_media_exposure': d.uniform(.2, .9), 'lat_u_lifestyle': d.signed(.2, .6, .5)}}
    spec['latent_variables'] = [
        {'id': 'lat_u_lifestyle', 'label': 'Unobserved lifestyle orientation', 'intercept': 0.0, 'noise_sd': 1.0},
        {'id': 'lat_media_exposure', 'label': 'Exposure to brand advertising', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'age_band': d.trend(AGE, .1), 'gender': d.categorical(['Female', 'Male']),
                     'settlement': d.categorical(settlements)}},
        {'id': 'lat_category_engagement', 'label': 'Snack category engagement', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'age_band': d.trend(AGE), 'gender': d.categorical(['Female', 'Male'])},
         'parents': {'lat_u_lifestyle': d.signed(.2, .6, .3)}},
        *aware.values(),
        {'id': 'lat_consider_a', 'label': 'Consideration of Brand A', 'intercept': 0.0, 'noise_sd': 1.0,
         'parents': {'lat_aware_a': d.uniform(.4, .9), 'lat_category_engagement': d.uniform(.1, .5)}},
        {'id': 'lat_consider_b', 'label': 'Consideration of Brand B', 'intercept': 0.0, 'noise_sd': 1.0,
         'parents': {'lat_aware_b': d.uniform(.4, .9), 'lat_category_engagement': d.uniform(.1, .5)}},
        {'id': 'lat_ad_recall', 'label': 'Recall of Brand A advertising', 'intercept': 0.0, 'noise_sd': 1.0,
         'parents': {'lat_media_exposure': d.uniform(.5, 1.0)}},
        {'id': 'lat_value_perception', 'label': 'Perceived value of Brand A', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'gender': d.categorical(['Female', 'Male']), 'age_band': d.trend(AGE)},
         'parents': {'lat_u_lifestyle': d.signed(.2, .5, .4)}},
    ]
    routed = {'variable': 'category_user', 'values': ['Yes'], 'documented': bool(d.rng.random() < .5)}
    spec['questionnaire'] = {'blocks': [
        {'id': 'category_user', 'label': 'Have you bought packaged snacks in the past month?', 'section': 'category',
         'answer_type': 'single', 'items': [{'id': 'category_user', 'label': 'Bought packaged snacks in the past month',
          'construct': 'lat_category_engagement', 'values': ['Yes', 'No'], 'value_intercepts': [d.uniform(.2, .9), 0.0],
          'value_loadings': [1.2, 0.0], 'refusal': False}]},
        {'id': 'brand_awareness', 'label': 'Which of these snack brands have you heard of?', 'section': 'brand',
         'answer_type': 'select_all', 'items': [
            {'id': f'aware_brand_{b}', 'label': f'Brand {b.upper()}', 'construct': f'lat_aware_{b}',
             'intercept': d.uniform(-1.2, 1.2), 'loading': 1.3} for b in 'abcd']},
        {'id': 'brand_consideration', 'label': 'Which of these brands would you consider buying?', 'section': 'brand',
         'answer_type': 'select_all', 'routing': routed, 'items': [
            {'id': f'consider_brand_{b}', 'label': f'Brand {b.upper()}', 'construct': f'lat_consider_{b}',
             'intercept': d.uniform(-1.0, .6), 'loading': 1.2} for b in 'ab']},
        {'id': 'ad_channels', 'label': 'Where have you seen advertising for Brand A recently?', 'section': 'media',
         'answer_type': 'select_all', 'items': [
            {'id': f'ad_seen_{c}', 'label': label, 'construct': 'lat_ad_recall', 'intercept': d.uniform(-1.6, -.2),
             'loading': 1.1} for c, label in [('tv', 'Television'), ('social', 'Social media'), ('outdoor', 'Outdoor posters')]]},
        {'id': 'next_purchase', 'label': 'Which brand are you most likely to buy next?', 'section': 'purchase intent',
         'answer_type': 'single', 'routing': {'variable': 'category_user', 'values': ['Yes'], 'documented': True},
         'items': [{'id': 'next_brand', 'label': 'Most likely next brand', 'construct': 'lat_consider_a',
                    'values': ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'None of these'],
                    'value_intercepts': [0.0, d.uniform(-.3, .3), d.uniform(-.6, 0), d.uniform(-.8, 0), d.uniform(-1.5, -.5)],
                    'value_loadings': [1.0, -.3, -.3, -.3, -.5], 'refusal': True}]},
        {'id': 'brand_attitudes', 'label': 'How far do you agree or disagree with these statements about Brand A?',
         'section': 'attitudes', 'answer_type': 'ordered_scale', 'scale': AGREEMENT, 'items': [
            {'id': 'att_good_value', 'label': 'Brand A is good value for money', 'construct': 'lat_value_perception',
             'loading': 1.0, 'cutpoints': [-2.2, -1.0, .2, 1.6], 'refusal': True},
            {'id': 'att_memorable_ads', 'label': 'Brand A advertising is memorable', 'construct': 'lat_ad_recall',
             'loading': 1.0, 'cutpoints': [-2.0, -.8, .4, 1.8], 'refusal': True}]},
    ]}
    spec['measurement_model'] = _measurement(d, [{'concept': 'actual purchase', 'item': 'next_brand',
        'rationale': 'Stated next-purchase intention, not observed purchasing.'}], ['lat_media_exposure'])
    spec['sampling_design'] = _design(d, 'region', regions, 'Islands')
    spec['response_process'] = {'base_logit': d.uniform(-.6, .2),
        'effects': {'age_band': {v: c for v, c in d.trend(AGE, .2).items() if c}},
        'latent_effects': {'lat_category_engagement': d.signed(.1, .4, .4)}}
    spec['scenario_bindings'] = {'causal_group_variables': ['age_band', 'gender', 'region', 'settlement'],
                                 'media_group_variables': ['age_band', 'gender'],
                                 'media_outcome_blocks': ['brand_awareness'],
                                 'purchase_outcome_blocks': ['next_purchase']}
    return _finish(spec)


def transit_service(study_id: str, seed: int, population_size: int = 200_000) -> Spec:
    d = _Draw(seed)
    region_name = d.choice(REGIONS)
    districts = ['Central', 'Riverside', 'Hillside', 'Harbour', 'Outskirts']
    employment = ['Employed', 'Student', 'Retired', 'Other']
    spec = _common(d, study_id, 'transit_service', f'Synthetic public transport survey ({region_name})', seed,
                   population_size, 'resident', region_name)
    spec['demographics'] = [
        {'id': 'age_band', 'label': 'Age band', 'values': AGE, 'base_logits': d.logits([.1, .2, .1, .1, 0, .05]),
         'contrasts': AGE_CONTRASTS, 'ordered': True},
        {'id': 'gender', 'label': 'Gender', 'values': ['Female', 'Male'], 'base_logits': d.logits([0, 0]),
         'contrasts': GENDER_CONTRASTS},
        {'id': 'district', 'label': 'District of residence', 'values': districts, 'base_logits': [.3, 0, 0, -.2, -3.4],
         'depends_on': {'lat_u_commute_need': [d.signed(.2, .6, .2), 0, 0, 0, d.signed(.2, .6, .2)]},
         'contrasts': [{'Central': ['Central'], 'Hillside': ['Hillside']}, {'Outskirts': ['Outskirts'], 'Riverside': ['Riverside']}]},
        {'id': 'employment', 'label': 'Employment status', 'values': employment, 'base_logits': d.logits([1.0, -.6, -.1, -.5]),
         'depends_on': {'lat_u_commute_need': [d.signed(.2, .7, .2), 0, 0, 0]},
         'contrasts': [{'Employed': ['Employed'], 'Retired': ['Retired']}, {'Student': ['Student'], 'Employed': ['Employed']}]},
    ]
    spec['latent_variables'] = [
        {'id': 'lat_u_commute_need', 'label': 'Unobserved commuting need', 'intercept': 0.0, 'noise_sd': 1.0},
        {'id': 'lat_info_exposure', 'label': 'Exposure to travel information campaigns', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'age_band': d.trend(AGE, .1), 'gender': d.categorical(['Female', 'Male'])}},
        {'id': 'lat_service_coverage', 'label': 'Local service coverage', 'intercept': 0.0, 'noise_sd': .6,
         'effects': {'district': d.categorical(districts, .1)}},
        {'id': 'lat_transit_use', 'label': 'Public transport use', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'age_band': d.trend(AGE), 'employment': d.categorical(employment)},
         'parents': {'lat_service_coverage': d.uniform(.3, .8), 'lat_u_commute_need': d.signed(.3, .8, .2)}},
        {'id': 'lat_reliability', 'label': 'Experienced service reliability', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'district': d.categorical(districts)}, 'parents': {'lat_service_coverage': d.uniform(.2, .6)}},
        {'id': 'lat_fare_sensitivity', 'label': 'Fare sensitivity', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'employment': d.categorical(employment), 'age_band': d.trend(AGE)}},
        {'id': 'lat_safety', 'label': 'Perceived travel safety', 'intercept': 0.0, 'noise_sd': 1.0,
         'effects': {'gender': d.categorical(['Female', 'Male']), 'district': d.categorical(districts)}},
        {'id': 'lat_app_awareness', 'label': 'Awareness of travel information services', 'intercept': 0.0, 'noise_sd': 1.0,
         'parents': {'lat_info_exposure': d.uniform(.4, 1.0), 'lat_transit_use': d.uniform(.1, .4)}},
    ]
    routed = {'variable': 'uses_transit', 'values': ['Yes'], 'documented': bool(d.rng.random() < .5)}
    spec['questionnaire'] = {'blocks': [
        {'id': 'uses_transit', 'label': 'Have you used public transport in the past four weeks?', 'section': 'usage',
         'answer_type': 'single', 'items': [{'id': 'uses_transit', 'label': 'Used public transport in the past four weeks',
          'construct': 'lat_transit_use', 'values': ['Yes', 'No'], 'value_intercepts': [d.uniform(-.2, .6), 0.0],
          'value_loadings': [1.3, 0.0], 'refusal': False}]},
        {'id': 'modes_used', 'label': 'Which of these services have you used in the past four weeks?', 'section': 'usage',
         'answer_type': 'select_all', 'routing': routed, 'items': [
            {'id': f'mode_{m}', 'label': label, 'construct': 'lat_transit_use', 'intercept': d.uniform(-1.5, .8),
             'loading': 1.0} for m, label in [('bus', 'Bus'), ('tram', 'Tram'), ('rail', 'Commuter rail'), ('ferry', 'Ferry')]]},
        {'id': 'problems', 'label': 'Which of these problems have you experienced on public transport recently?',
         'section': 'experience', 'answer_type': 'select_all', 'routing': routed, 'items': [
            {'id': f'problem_{p}', 'label': label, 'construct': 'lat_reliability', 'intercept': d.uniform(-1.2, .2),
             'loading': -1.1} for p, label in [('delays', 'Delays'), ('crowding', 'Overcrowding'),
                                               ('connections', 'Missed connections'), ('information', 'Unclear information')]]},
        {'id': 'info_services', 'label': 'Which of these travel information services have you heard of?',
         'section': 'information', 'answer_type': 'select_all', 'items': [
            {'id': f'heard_{s}', 'label': label, 'construct': 'lat_app_awareness', 'intercept': d.uniform(-1.3, .5),
             'loading': 1.2} for s, label in [('app', 'TransitNow app'), ('planner', 'Journey planner website'),
                                              ('alerts', 'Text alerts')]]},
        {'id': 'ticket_plan', 'label': 'Which ticket do you plan to buy next month?', 'section': 'purchase intent',
         'answer_type': 'single', 'items': [{'id': 'ticket_plan', 'label': 'Planned ticket next month',
          'construct': 'lat_transit_use', 'values': ['Monthly pass', 'Pay as you go', 'Annual pass', 'No ticket'],
          'value_intercepts': [d.uniform(-.6, 0), 0.0, d.uniform(-1.6, -.8), d.uniform(.2, .8)],
          'value_loadings': [1.1, .5, .9, -1.0], 'refusal': True}]},
        {'id': 'transit_attitudes', 'label': 'How far do you agree or disagree with these statements about local public transport?',
         'section': 'attitudes', 'answer_type': 'ordered_scale', 'scale': AGREEMENT, 'items': [
            {'id': 'att_reliable', 'label': 'Services are reliable', 'construct': 'lat_reliability', 'loading': 1.0,
             'cutpoints': [-1.8, -.6, .5, 1.9], 'refusal': True},
            {'id': 'att_fares', 'label': 'Fares are good value', 'construct': 'lat_fare_sensitivity', 'loading': -1.0,
             'cutpoints': [-1.6, -.4, .6, 2.0], 'refusal': True},
            {'id': 'att_safe', 'label': 'I feel safe travelling', 'construct': 'lat_safety', 'loading': 1.0,
             'cutpoints': [-2.4, -1.2, 0.0, 1.4], 'refusal': True}]},
    ]}
    spec['measurement_model'] = _measurement(d, [{'concept': 'actual ticket purchase', 'item': 'ticket_plan',
        'rationale': 'Stated ticket plan, not observed ticket sales.'}], ['lat_info_exposure'])
    spec['sampling_design'] = _design(d, 'district', districts, 'Outskirts')
    spec['response_process'] = {'base_logit': d.uniform(-.6, .2),
        'effects': {'employment': {v: c for v, c in d.categorical(employment, .4).items() if c}},
        'latent_effects': {'lat_transit_use': d.signed(.1, .4, .4)}}
    spec['scenario_bindings'] = {'causal_group_variables': ['age_band', 'gender', 'district', 'employment'],
                                 'media_group_variables': ['age_band', 'gender'],
                                 'media_outcome_blocks': ['info_services'],
                                 'purchase_outcome_blocks': ['ticket_plan']}
    return _finish(spec)


DOMAINS: dict[str, Callable[..., Spec]] = {'brand_tracking': brand_tracking, 'transit_service': transit_service}


def template_spec(domain: str, study_id: str, seed: int, population_size: int = 200_000) -> Spec:
    if domain not in DOMAINS:
        raise ValueError('Unknown synthetic domain: '+domain+'; choose '+', '.join(sorted(DOMAINS)))
    return DOMAINS[domain](study_id, seed, population_size)
