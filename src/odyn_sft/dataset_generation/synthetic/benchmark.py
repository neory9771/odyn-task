"""Multi-study benchmark: study-level and analysis-family-level isolated splits.

generate studies -> split studies -> own analysis families globally ->
per study: result-blind families -> reference execution -> selection -> gold ->
paired Alpaca records. The existing case pipeline (execute_family, candidates,
select, materialize, contracts, prompts) is reused unchanged; only study-specific
wording is localised. Latent truth is attached to private gold as diagnostics and
never enters candidate inputs or reference outputs.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import copy
import re
import shutil
from typing import Any, Callable

from ...survey_api import run_reference
from ...survey_metadata.catalogue import numeric_program, numeric_references
from ..contracts import analysis_contract, synthesis_contract
from ..dataset.families import execute_family, training_claim
from ..provenance import implementation_hashes as case_hashes
from ..generation import (STYLES, candidates, fixed_direction,
                                     materialize, reference_perspective, validate_reference)
from ...prompts.survey import ANALYSIS_INSTRUCTION, PROMPT_VERSION, SYNTHESIS_INSTRUCTION, analysis_input, synthesis_input
from ..selection import coverage, select
from ..study import atomic_key, measure_key
from ...common.storage import atomic_text, digest, file_hash, read_json, read_jsonl, write_json, write_jsonl
from .population import Population, causal_truth, group_shares
from .spec import REFUSAL, Spec, validate_spec
from .survey import HIDDEN_FILES, VISIBLE_FILES, generate_study
from .templates import DOMAINS, template_spec

VERSION = 'synthetic-benchmark-0.1.0'
SPLITS = ('train', 'validation', 'test')
FAMILIAR = 'test_familiar'
# Wording of the real 2022 US survey frozen in shared templates; none may survive localisation.
REAL_SURVEY_FRAGMENTS = ('unweighted 2022', '2022 survey', '2020 and 2022', 'United States', 'US consumers',
                         'US population', 'quant_us_2022')
LANGUAGE_STYLES = ['', 'business', 'informal', 'formal', 'presentation', 'working_interpretation']
# Strings that must never reach a candidate-facing file.
HIDDEN_MARKERS = (re.compile(r'(?<![A-Za-z0-9])lat_'), 'oracle_weight', 'inclusion_probability', 'response_propensity', 'population_index',
                  'latent_truth', 'average_causal_effect', 'population_share')


def implementation_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {**case_hashes(), **{'synthetic/'+p.name: file_hash(p) for p in sorted(here.glob('*.py'))}}


def load_config(path: str | Path) -> dict[str, Any]:
    """Resolve paths relative to the config workspace and validate study isolation."""
    path = Path(path).resolve()
    config = read_json(path)
    required = {'workspace', 'seed', 'studies', 'population_size', 'family_partition_ratios', 'splits',
                'train_variants', 'familiar_test_cases_per_study'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError('Synthetic benchmark config keys must be exactly: '+', '.join(sorted(required)))
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(config['population_size']) is not int or config['population_size'] < 1000:
        raise ValueError('population_size must be an integer >= 1000')
    if type(config['train_variants']) is not int or not 1 <= config['train_variants'] <= len(STYLES):
        raise ValueError('Invalid train_variants')
    if type(config['familiar_test_cases_per_study']) is not int or config['familiar_test_cases_per_study'] < 0:
        raise ValueError('familiar_test_cases_per_study must be a nonnegative integer')
    ratios = config['family_partition_ratios']
    if set(ratios) != set(SPLITS) or any(type(v) not in (int, float) or v <= 0 for v in ratios.values()):
        raise ValueError('family_partition_ratios need positive train/validation/test values')
    if set(config['studies']) != set(SPLITS) or set(config['splits']) != set(SPLITS):
        raise ValueError('studies and splits must name exactly train/validation/test')
    workspace = (path.parent/config['workspace']).resolve()
    seen = set()
    for split in SPLITS:
        policy = config['splits'][split]
        if set(policy) != {'cases_per_study', 'coverage_minima', 'interaction_minima'} or \
                type(policy['cases_per_study']) is not int or policy['cases_per_study'] < 1:
            raise ValueError('Invalid split policy: '+split)
        entries = config['studies'][split]
        if not isinstance(entries, list) or not entries:
            raise ValueError('Each split needs at least one study: '+split)
        for entry in entries:
            if not isinstance(entry, dict) or 'study_id' not in entry:
                raise ValueError('Study entries need study_id')
            if entry['study_id'] in seen:
                # Pairwise disjoint studies, not merely an empty triple intersection.
                raise ValueError('Study appears more than once: '+entry['study_id'])
            seen.add(entry['study_id'])
            if set(entry) == {'study_id', 'domain', 'seed'}:
                if entry['domain'] not in DOMAINS or type(entry['seed']) is not int:
                    raise ValueError('Invalid template study: '+entry['study_id'])
            elif set(entry) == {'study_id', 'spec'}:
                entry['spec'] = str((workspace/entry['spec']).resolve())
            else:
                raise ValueError('Study entries are {study_id, domain, seed} or {study_id, spec}')
    config['workspace'] = str(workspace)
    return config


def study_spec(entry: dict[str, Any], population_size: int) -> Spec:
    if 'spec' in entry:
        spec = validate_spec(read_json(entry['spec']))
        if spec['study_id'] != entry['study_id']:
            raise ValueError('Spec study_id differs from config: '+entry['study_id'])
        return spec
    return template_spec(entry['domain'], entry['study_id'], entry['seed'], population_size)


def study_seed(seed: int, study_id: str) -> int:
    return int(digest({'seed': seed, 'study': study_id, 'purpose': 'study-cases'})[:8], 16)


def family_owner(core: str, ratios: dict[str, float], seed: int) -> str:
    """Study-independent ownership: one analysis family belongs to one split everywhere."""
    total = sum(ratios.values())
    u = int(digest({'core': core, 'seed': seed, 'purpose': 'synthetic-family-partition'})[:16], 16)/2**64
    return 'train' if u < ratios['train']/total else 'validation' if u < (ratios['train']+ratios['validation'])/total else 'test'


def enumerate_study_families(metadata: dict[str, Any], spec: Spec, seed: int) -> list[dict[str, Any]]:
    """Metadata-only inventory with study-declared contrasts; no responses touched.

    The core hash names the analysis meaning (variable, response, groups,
    estimand) independently of the study, so it identifies a family across
    studies that share a questionnaire. Claim direction is committed per study.
    """
    cards = {c['id']: c for c in metadata['variables']}
    contrasts = [(d['id'], c) for d in spec['demographics'] for c in d['contrasts']]
    result, seen = [], set()
    for card in cards.values():
        kind = card['answer_type']
        if kind not in {'select_all', 'single', 'ordered_scale'} or card['section'] == 'demographics':
            continue
        values = [None] if kind != 'single' else [v for v in card['values'] if v != REFUSAL]
        for value in values:
            measure = measure_key(card['id'], kind, value)
            for gv, groups in contrasts:
                for threshold in (False, True):
                    target = dict(list(groups.items())[:1]) if threshold else groups
                    core = atomic_key(measure, gv, target, 'threshold' if threshold else 'difference')
                    if core in seen:
                        continue
                    seen.add(core)
                    direction = fixed_direction(core, seed)
                    result.append({'core': core, 'row': {'card': card, 'value': value, 'measure': measure,
                                   'group_variable': gv, 'groups': target}, 'threshold': threshold, 'direction': direction,
                                   'direction_commitment': digest({'core': core, 'direction': direction, 'seed': seed})})
    return result


def scenario_rules(spec: Spec) -> dict[str, Any]:
    """Translate spec block ids into metadata block ids (the question wording)."""
    labels = {b['id']: b['label'] for b in spec['questionnaire']['blocks']}
    bindings = spec['scenario_bindings']
    rules = {k: list(bindings.get(k, [])) for k in ('causal_group_variables', 'media_group_variables')}
    for name in ('media_outcome_blocks', 'purchase_outcome_blocks'):
        rules[name] = [labels[b] for b in bindings.get(name, [])]
    rules['purchase_values'] = dict(bindings.get('purchase_values', {}))
    return rules


def survey_stressors(spec: Spec, record: dict[str, Any]) -> list[str]:
    """Select-all S tags only where the recorded format really leaves them ambiguous."""
    card = record['row']['card']
    if card['answer_type'] != 'select_all':
        return []
    block = next(b for b in spec['questionnaire']['blocks'] if b['label'] == card['block_id'])
    undocumented = block.get('routing') is not None and not block['routing']['documented']
    if spec['measurement_model']['checkbox_encoding'] == 'true_or_blank':
        return ['routing_ambiguity', 'denominator_ambiguity']
    return ['routing_ambiguity'] if undocumented else []


def _replacements(spec: Spec) -> list[tuple[str, str]]:
    year, population = spec['fieldwork']['year'], spec['target_population']
    return [('observed unweighted 2022 analysis bases', 'observed analysis bases of the supplied study'),
            ('between 2020 and 2022', f'between {year-2} and {year}'),
            ('Only the 2022 survey is supplied', f'Only the {year} survey is supplied'),
            ('Observed unweighted 2022 analysis base', f'Observed unweighted {year} analysis base'),
            ('These percentages also describe all US consumers.', f"These percentages also describe all {population['description']}."),
            ('identifies US population prevalence', f"identifies {population['region']} population prevalence"),
            ('applies to consumers outside the United States', f"applies to {population['unit']}s outside {population['region']}")]


def localize(value: Any, spec: Spec) -> Any:
    """Replace the real-survey wording frozen in the shared case templates."""
    if isinstance(value, str):
        for old, new in _replacements(spec):
            value = value.replace(old, new)
        if any(fragment in value for fragment in REAL_SURVEY_FRAGMENTS):
            raise ValueError('Unlocalised real-survey wording in synthetic case: '+value[:120])
        return value
    if isinstance(value, list):
        return [localize(v, spec) for v in value]
    if isinstance(value, dict):
        return {k: localize(v, spec) for k, v in value.items()}
    return value


def _relation(holds: bool | None, evidence: str) -> str:
    if holds is None:
        return 'population_truth_undefined'
    if evidence == 'support':
        return 'evidence_matches_truth' if holds else 'evidence_supports_false_claim'
    if evidence == 'contradict':
        return 'evidence_matches_truth' if not holds else 'evidence_contradicts_true_claim'
    if evidence == 'inconclusive':
        return 'true_but_inconclusive' if holds else 'false_and_inconclusive'
    return 'true_but_unavailable' if holds else 'false_and_unavailable'


def case_truth(candidate: dict[str, Any], case: dict[str, Any], population: Population) -> dict[str, Any]:
    """Hidden diagnostic layer: population and causal truth beside supplied evidence."""
    by_core = {a['core']: a for a in candidate['analyses']}
    components, causal = [], None
    for component in case['components']:
        if 'analytical_core' not in component:
            continue
        analysis = by_core[component['analytical_core']]
        row = analysis['row']
        measure = row['measure']
        shares = group_shares(population, measure['variable'], measure['operation'], row['group_variable'], row['groups'])
        values = [s['share'] for s in shares]
        delta = None if None in values else values[0]-(0.5 if analysis['threshold'] else values[1])
        holds = None if delta is None or delta == 0 else (delta > 0) == (analysis['direction'] > 0)
        item = {'component_id': component['id'], 'population_shares': shares, 'population_delta': delta,
                'claim_direction_holds_in_population': holds, 'E': component['E'],
                'relation': _relation(holds, component['E'])}
        if not analysis['threshold']:
            item['causal'] = causal_truth(population, measure['variable'], measure['operation'],
                                          row['group_variable'], row['groups'])
            causal = causal or (item['causal'], analysis['direction'])
        components.append(item)
    scenario = case['scenario']
    limit: dict[str, Any] | None = None
    if scenario in {'causal', 'media'} and causal is not None:
        truth, direction = causal
        effect = truth['average_causal_effect']
        real = truth['demographic_is_causal_ancestor'] and abs(effect) >= 0.005 and (effect > 0) == (direction > 0)
        if scenario == 'media':
            real = real and bool(truth['mediating_unmeasured_constructs'])
        limit = {'component_id': 'limit', 'claim_true_in_dgp': real, 'identified_by_supplied_study': False,
                 'relation': 'true_but_not_identified' if real else 'false_but_superficially_plausible'}
    elif scenario == 'population' and components:
        estimates = [r for r in case['evidence']['results'] if r['kind'] == 'estimate']
        gaps = [abs(x['value']-s['share']) for r in estimates for x in r['estimates'] for s in components[0]['population_shares']
                if x['group'] == s['group'] and x['value'] is not None and s['share'] is not None]
        limit = {'component_id': 'limit', 'max_abs_sample_minus_population_share': max(gaps) if gaps else None,
                 'claim_true_in_dgp': bool(gaps) and max(gaps) <= .02, 'identified_by_supplied_study': False}
        limit['relation'] = 'true_but_not_identified' if limit['claim_true_in_dgp'] else 'false_but_superficially_plausible'
    elif scenario in {'behaviour', 'transactions', 'transport', 'historical'}:
        limit = {'component_id': 'limit', 'claim_true_in_dgp': None, 'identified_by_supplied_study': False,
                 'relation': 'not_modeled_by_dgp'}
    return {'components': components, 'limit': limit,
            'gold_policy': 'Gold follows the supplied study evidence; latent truth is diagnostic and never shown to candidates.'}


def _stage(output: Path) -> Path:
    staging = output.with_name('.'+output.name+'.partial')
    if output.exists() or staging.exists():
        raise ValueError('Use a new output directory (and remove any stale .partial staging directory)')
    return staging


def build(config_path: str | Path, output: str | Path, progress: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Generate studies and isolated cases; publish only if every split is feasible."""
    config = load_config(config_path)
    output = Path(output)
    staging = _stage(output)
    try:
        result = _build(config, staging, progress)
        staging.rename(output)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _build(config: dict[str, Any], root: Path, progress: Callable[[dict], None] | None) -> dict[str, Any]:
    ratios, seed = config['family_partition_ratios'], config['seed']
    study_split = {e['study_id']: s for s in SPLITS for e in config['studies'][s]}
    entries = {e['study_id']: e for s in SPLITS for e in config['studies'][s]}
    owners: dict[str, str] = {}
    gold, reports, shortages = [], {}, []
    for study_id, split in study_split.items():
        spec = study_spec(entries[study_id], config['population_size'])
        data, metadata, population = generate_study(spec, root/'studies'/study_id, root/'hidden'/'studies'/study_id)
        write_json(root/'studies'/study_id/'numeric_references.json', numeric_references(metadata['variables']))
        local_seed = study_seed(seed, study_id)
        families = enumerate_study_families(metadata, spec, local_seed)
        for family in families:
            owners.setdefault(family['core'], family_owner(family['core'], ratios, seed))
        # Ownership is fixed before any reference program executes.
        slices = [(split, split, config['splits'][split]['cases_per_study'], config['splits'][split])]
        if split == 'test' and config['familiar_test_cases_per_study']:
            slices.append((FAMILIAR, 'train', config['familiar_test_cases_per_study'],
                           {'coverage_minima': {}, 'interaction_minima': {}}))
        rules = scenario_rules(spec)
        for offset, (slice_name, owner, size, policy) in enumerate(slices):
            analyses = []
            for family in families:
                if owners[family['core']] != owner:
                    continue
                record = execute_family(family, data, metadata, local_seed)
                record['S'] = survey_stressors(spec, record)
                analyses.append(record)
            pool = candidates(analyses, local_seed+offset, set(), rules)
            chosen, _, missing = select(pool, size, local_seed+100+offset, policy['coverage_minima'],
                                        policy['interaction_minima'], [])
            report = coverage(chosen, size, policy['coverage_minima'], policy['interaction_minima'], [])
            report.update(candidate_cases=len(pool), owned_families=len(analyses),
                          scenario_counts_diagnostic_only=dict(Counter(c['scenario'] for c in chosen)))
            reports[f'{study_id}/{slice_name}'] = report
            shortages += [{'study_id': study_id, 'slice': slice_name, **m} for m in missing]
            for candidate in chosen:
                gold += _cases(candidate, slice_name, spec, metadata, data, population, local_seed,
                               config['train_variants'] if slice_name == 'train' else 1)
        if progress:
            progress({'stage': 'study_complete', 'study_id': study_id, 'split': split, 'respondents': len(data),
                      'families': len(families)})
    if shortages:
        raise ValueError(f'{len(shortages)} coverage shortages; no benchmark published: '+str(shortages[:5]))
    assert_isolation(gold, owners, study_split)
    plan = {'version': VERSION, 'prompt_version': PROMPT_VERSION, 'config': config, 'implementation_hashes': implementation_hashes(),
            'study_splits': study_split, 'family_owners': owners, 'family_partition_hash': digest(owners),
            'partition_before_reference_execution': True, 'coverage': reports,
            'selected_family_ids': {s: sorted({c['family_id'] for c in gold if c['slice'] == s}) for s in (*SPLITS, FAMILIAR)},
            'llm_calls': 0, 'status': 'synthetic_draft_pending_review'}
    write_json(root/'plan.json', plan)
    write_jsonl(root/'hidden'/'private_gold.jsonl', gold)
    export_alpaca(root, gold)
    write_json(root/'diversity.json', diversity(gold))
    atomic_text(root/'README.md', README)
    leakage_audit(root)
    write_manifest(root)
    return {'studies': dict(Counter(study_split.values())),
            'records': dict(Counter(c['slice'] for c in gold)),
            'families': {s: len(v) for s, v in plan['selected_family_ids'].items()},
            'truth_relations': dict(Counter(x['relation'] for c in gold if c['variant'] == 0 for x in c['latent_truth']['components'])),
            'llm_calls': 0, 'status': plan['status']}


def _cases(candidate: dict[str, Any], slice_name: str, spec: Spec, metadata: dict[str, Any], data: Any,
           population: Population, seed: int, variants: int) -> list[dict[str, Any]]:
    split = 'test' if slice_name == FAMILIAR else slice_name
    case = materialize(candidate, split, seed, metadata)
    case['study_id'] = spec['study_id']
    case = localize(case, spec)
    analysis_family = case['family_id']
    case.update(study_id=spec['study_id'], analysis_family_id=analysis_family, slice=slice_name,
                family_id=digest({'study': spec['study_id'], 'family': analysis_family}),
                case_id=f"syn_{spec['study_id']}_{analysis_family[:16]}",
                draft_status='synthetic_pending_expert_and_language_review')
    if 'context_measure' in case:
        case['context_measure'].update(source_period=str(spec['fieldwork']['year']),
            requested_prior_period=str(spec['fieldwork']['year']-2) if case['scenario'] == 'historical' else None)
    case['evidence'] = run_reference(case['program'], data, metadata)
    validate_reference(case, case['evidence'])
    case['analysis_contract'] = analysis_contract(case)
    case['synthesis_contract'] = synthesis_contract(case)
    case['reference_perspective'] = localize(reference_perspective(case), spec)
    case['reference_language_status'] = 'template_pending_expert_review'
    case['latent_truth'] = case_truth(candidate, case, population)
    case['slice_kind'] = 'unseen_study_familiar_family' if slice_name == FAMILIAR else \
        'seen_study_type' if slice_name == 'train' else 'unseen_study_unseen_family'
    records = []
    for variant in range(variants):
        record = copy.deepcopy(case)
        record['claim'] = training_claim(record, variant)
        record['language_style'] = 'primary' if variant == 0 else LANGUAGE_STYLES[variant]
        record['record_id'] = record['case_id']+f'.v{variant}'
        record['variant'] = variant
        records.append(record)
    return records


def assert_isolation(gold: list[dict[str, Any]], owners: dict[str, str], study_split: dict[str, str]) -> None:
    """Studies are disjoint; every atom has the owner its slice requires."""
    family_slice = {}
    absence_keys = {k for c in gold for k in c['scenario_keys']}
    for case in gold:
        slice_name = case['slice']
        split = 'test' if slice_name == FAMILIAR else slice_name
        if study_split[case['study_id']] != split:
            raise ValueError('Case study crosses splits: '+case['case_id'])
        required = 'train' if slice_name == FAMILIAR else split
        if family_slice.setdefault(case['family_id'], slice_name) != slice_name:
            raise ValueError('Case family crosses slices: '+case['case_id'])
        for core in case['component_keys']:
            owner = owners.get(core)
            if owner is not None and owner != required:
                raise ValueError('Analysis family ownership violated: '+case['case_id'])
            if owner is None and core not in absence_keys:
                raise ValueError('Unowned analysis key: '+core)


def _alpaca(case: dict[str, Any], metadata: dict[str, Any], references: dict[str, Any]) -> tuple[dict, dict]:
    analysis = {'instruction': ANALYSIS_INSTRUCTION, 'input': analysis_input(case['claim'], metadata),
                'output': numeric_program(case['program'], references)}
    synthesis = {'instruction': SYNTHESIS_INSTRUCTION,
                 'input': synthesis_input(case['claim'], case['program'], case['evidence'], metadata),
                 'output': case['reference_perspective']}
    return analysis, synthesis


def export_alpaca(root: Path, gold: list[dict[str, Any]]) -> None:
    """Strict three-field Alpaca records; study and family identities in lineage sidecars."""
    studies = {}
    for slice_name in (*SPLITS, FAMILIAR):
        records = [c for c in gold if c['slice'] == slice_name]
        if not records:
            continue
        analysis, synthesis, lineage = [], [], []
        for case in records:
            if case['study_id'] not in studies:
                folder = root/'studies'/case['study_id']
                studies[case['study_id']] = (read_json(folder/'metadata.json'), read_json(folder/'numeric_references.json'))
            a, s = _alpaca(case, *studies[case['study_id']])
            analysis.append(a)
            synthesis.append(s)
            lineage.append({'record_id': case['record_id'], 'study_id': case['study_id'],
                            'study_dir': 'studies/'+case['study_id'], 'family_id': case['family_id'],
                            'analysis_family_id': case['analysis_family_id'], 'component_keys': case['component_keys'],
                            'variant': case['variant'], 'slice_kind': case['slice_kind']})
        private = slice_name in ('test', FAMILIAR)
        suffix = '_gold' if private else ''
        write_jsonl(root/f'{slice_name}_analysis{suffix}.jsonl', analysis)
        write_jsonl(root/f'{slice_name}_synthesis{suffix}.jsonl', synthesis)
        write_jsonl(root/f'{slice_name}_lineage.jsonl', lineage)
        if private:
            write_jsonl(root/f'{slice_name}_analysis_inputs.jsonl', [{k: v for k, v in r.items() if k != 'output'} for r in analysis])
            write_jsonl(root/f'{slice_name}_synthesis_oracle_inputs.jsonl', [{k: v for k, v in r.items() if k != 'output'} for r in synthesis])


def _counts(counter: Counter) -> dict[str, int]:
    """Sorted keys keep reports byte-stable despite set iteration order."""
    return dict(sorted(counter.items()))


def diversity(gold: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for slice_name in (*SPLITS, FAMILIAR):
        primary = [c for c in gold if c['slice'] == slice_name and c['variant'] == 0]
        if not primary:
            continue
        result[slice_name] = {'cases': len(primary), 'rows': sum(c['slice'] == slice_name for c in gold),
            'studies': _counts(Counter(c['study_id'] for c in primary)),
            'dimensions': {d: _counts(Counter(v for c in primary for v in (set(c[d]) if isinstance(c[d], list) else [c[d]])))
                           for d in ('C', 'D', 'A', 'E', 'R', 'S')},
            'scenarios': _counts(Counter(c['scenario'] for c in primary)),
            'truth_relations_hidden_diagnostic': _counts(Counter(x['relation'] for c in primary for x in c['latent_truth']['components'])),
            'limit_relations_hidden_diagnostic': _counts(Counter(c['latent_truth']['limit']['relation'] for c in primary
                                                              if c['latent_truth']['limit']))}
    return result


def leakage_audit(root: Path) -> dict[str, int]:
    """Fail if hidden DGP markers appear in any candidate-facing file."""
    checked = 0
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or relative.parts[0] == 'hidden' or path.suffix == '.parquet' or path.name in {'plan.json', 'diversity.json', 'manifest.json'}:
            continue
        text = path.read_text()
        for marker in HIDDEN_MARKERS:
            if (marker.search(text) if isinstance(marker, re.Pattern) else marker in text):
                raise ValueError(f'Hidden benchmark marker {getattr(marker, "pattern", marker)!r} leaked into {relative}')
        checked += 1
    return {'files_checked': checked}


def write_manifest(root: Path) -> None:
    files = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
    write_json(root/'manifest.json', {'version': VERSION, 'prompt_version': PROMPT_VERSION, 'files': files,
        'candidate_visible': ['studies/*/'+name for name in VISIBLE_FILES]+['{train,validation}_*.jsonl', '*_inputs.jsonl'],
        'benchmark_only': ['hidden/private_gold.jsonl']+['hidden/studies/*/'+name for name in HIDDEN_FILES]+['*_gold.jsonl', 'plan.json'],
        'release_status': 'synthetic_draft_pending_expert_review', 'expert_review_complete': False, 'llm_calls': 0})


def validate_benchmark(directory: str | Path) -> dict[str, Any]:
    """Recheck hashes, study/family isolation, Alpaca alignment and leakage."""
    root = Path(directory)
    manifest = read_json(root/'manifest.json')
    if manifest.get('version') != VERSION or manifest.get('prompt_version') != PROMPT_VERSION:
        raise ValueError('Unsupported synthetic benchmark or prompt version')
    for name, expected in manifest['files'].items():
        if file_hash(root/name) != expected:
            raise ValueError('Artifact hash mismatch: '+name)
    present = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if present != set(manifest['files']):
        raise ValueError('Unlisted or missing benchmark files')
    plan = read_json(root/'plan.json')
    gold = read_jsonl(root/'hidden'/'private_gold.jsonl')
    assert_isolation(gold, plan['family_owners'], plan['study_splits'])
    studies = {}
    for slice_name in (*SPLITS, FAMILIAR):
        records = [c for c in gold if c['slice'] == slice_name]
        if not records:
            continue
        private = slice_name in ('test', FAMILIAR)
        suffix = '_gold' if private else ''
        tables = {task: read_jsonl(root/f'{slice_name}_{task}{suffix}.jsonl') for task in ('analysis', 'synthesis')}
        lineage = read_jsonl(root/f'{slice_name}_lineage.jsonl')
        if [r['record_id'] for r in lineage] != [c['record_id'] for c in records] or any(len(t) != len(records) for t in tables.values()):
            raise ValueError('Lineage/Alpaca alignment mismatch: '+slice_name)
        for case, a, s in zip(records, tables['analysis'], tables['synthesis']):
            if case['study_id'] not in studies:
                folder = root/'studies'/case['study_id']
                studies[case['study_id']] = (read_json(folder/'metadata.json'), read_json(folder/'numeric_references.json'))
            if (a, s) != _alpaca(case, *studies[case['study_id']]):
                raise ValueError('Alpaca gold/input alignment failure: '+case['record_id'])
        if private:
            for task, public in (('analysis', 'analysis_inputs'), ('synthesis', 'synthesis_oracle_inputs')):
                if read_jsonl(root/f'{slice_name}_{public}.jsonl') != [{k: v for k, v in r.items() if k != 'output'} for r in tables[task]]:
                    raise ValueError('Private test inputs expose or mismatch outputs')
    audit = leakage_audit(root)
    return {'kind': 'synthetic_benchmark', 'integrity_valid': True, 'studies': dict(Counter(plan['study_splits'].values())),
            'records': dict(Counter(c['slice'] for c in gold)), **audit, 'llm_calls': 0}


README = '''# Synthetic multi-study benchmark (draft)

Generated locally from StudySpecs with no LLM calls.

* `studies/<id>/` - candidate-visible survey artefacts (respondents, metadata, codebook, description).
* `{train,validation}_{analysis,synthesis}.jsonl` - paired Alpaca records; `*_lineage.jsonl` maps each row to its study.
* `test_*` - strict hold-out: unseen studies and analysis families unseen in training.
* `test_familiar_*` - diagnostic slice: unseen test studies, analysis families owned by train. Never train on it.
* `hidden/` - benchmark-only truth: StudySpecs, DGP, population, sampling/measurement truth and private gold
  with latent-truth diagnostics. Gold labels follow the supplied evidence, not hidden truth.

Status: synthetic draft pending expert and language review.
'''
