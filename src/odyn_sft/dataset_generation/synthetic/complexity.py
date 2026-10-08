"""Cheap, versioned complexity heuristics for filtering generated experiments.

No LLMs, pairwise candidate searches, code interpretation or difficulty fitting.
Study summaries are cached once. Case labels use catalogue size, supplied labels
and a few counts from the declared analysis and generated reference program.
These replace the heavier revision-2 mapping rule at the user's request.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import re
import unicodedata
from typing import Any

RULE_VERSION = 'complexity-heuristic-1.0.0'
_STUDIES: dict[str, dict[str, Any]] = {}


def normalize(text: str) -> str:
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def block_groups(metadata: dict[str, Any]) -> dict[str,list[dict[str,Any]]]:
    blocks: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for card in metadata['variables']:
        blocks[card['block_id']].append(card)
    return dict(blocks)


def duplicate_blocks(metadata: dict[str, Any]) -> set[str]:
    signatures: dict[tuple[str,...],list[str]] = defaultdict(list)
    for block,cards in block_groups(metadata).items():
        signatures[tuple(sorted(normalize(c['label']) for c in cards))].append(block)
    return {b for group in signatures.values() if len(group)>1 for b in group}


def routed_blocks(metadata: dict[str, Any]) -> set[str]:
    note = metadata['study'].get('routing_note','')
    return {b for b,cards in block_groups(metadata).items()
            if b in note or any('eligibility_unknown' in c.get('flags',[]) for c in cards)}


def study_features(metadata: dict[str, Any]) -> dict[str,Any]:
    key = metadata.get('hash')
    if key and key in _STUDIES:
        return _STUDIES[key]['features']
    blocks = block_groups(metadata)
    labels = Counter(normalize(c['label']) for c in metadata['variables'])
    types = Counter(c['answer_type'] for c in metadata['variables'])
    features = {'variable_count':len(metadata['variables']),'block_count':len(blocks),
        'answer_format_count':len(types),'answer_format_counts':dict(types),
        'near_duplicate_block_count':len(duplicate_blocks(metadata)),
        'duplicate_label_group_count':sum(n>1 for n in labels.values()),
        'routed_block_count':len(routed_blocks(metadata)),
        'select_all_block_count':sum(any(c['answer_type']=='select_all' for c in cards) for cards in blocks.values())}
    if key:
        _STUDIES[key] = {'features':features,'cards':{c['id']:c for c in metadata['variables']}}
    return features


def mapping_features(claim: str, metadata: dict[str,Any], required: list[str],
                     *, recode: bool = False, proxy_or_gap: bool = False) -> dict[str,Any]:
    features = study_features(metadata)
    cached = _STUDIES.get(metadata.get('hash',''))
    cards = cached['cards'] if cached else {c['id']:c for c in metadata['variables']}
    text = ' '+normalize(claim)+' '
    return {'catalogue_variables':features['variable_count'],'mapping_required':bool(required),
            'label_exposed':bool(required) and all(' '+normalize(cards[v]['label'])+' ' in text for v in required),
            'recode_required':recode,'proxy_or_gap':proxy_or_gap}


def q_level(features: dict[str,Any]) -> str:
    """Explicit target labels or <=64 variables: low; hidden labels in >=256: high."""
    if not features['mapping_required'] or features['label_exposed'] or features['catalogue_variables']<=64:
        return 'low'
    return 'high' if features['catalogue_variables']>=256 else 'medium'


def program_features(program: str, contract: dict[str,Any]) -> dict[str,Any]:
    # References are generator-produced Python; this is just a lexical count,
    # not an interpreter. Executability is validated separately by the runtime.
    calls = Counter(re.findall(r'\b([A-Za-z_]\w*)\s*\(',program))
    analyses = contract.get('required_analyses',[])
    recodes = {tuple(values) for a in analyses for values in a.get('groups',{}).values()
               if isinstance(values,list) and len(values)>=2}
    return {'n_subclaims':calls['subclaim'],'n_variables':len(contract.get('required_variables',[])),
        'n_filters':calls['where']+calls['eligible']+program.count('among='),
        'n_recodes':len(recodes),'n_groups':calls['group'],
        'variable_defined_groups':sum(bool(a.get('variable_defined_groups')) for a in analyses),
        'n_estimands':calls['proportion']+calls['top_box']+calls['mean_score'],
        'n_comparisons':calls['compare'],
        'n_threshold_comparisons':sum(a.get('threshold') is not None for a in analyses),
        'n_gap_calls':calls['not_measured']+calls['proxy'],
        'relation_depth':contract.get('relation_depth',0),
        'api_operations':sorted(set(calls)&set(contract.get('available_functions',calls)))}


def p_level(features: dict[str,Any]) -> str:
    if (features['n_subclaims']>=3 or features['n_estimands']>=2 and
        (features['n_filters'] or features['variable_defined_groups']) or
        features['n_gap_calls']>=2 and features['n_estimands']>=2):
        return 'high'
    if features['n_estimands']<=1 and features['n_subclaims']<=1 and not any(features[k] for k in ('n_filters','n_recodes','n_gap_calls')):
        return 'low'
    return 'medium'


def annotate(claim: str, program: str, metadata: dict[str,Any], contract: dict[str,Any]) -> dict[str,Any]:
    features = program_features(program,contract)
    mapping = mapping_features(claim,metadata,contract.get('required_variables',[]),
        recode=features['n_recodes']>0,proxy_or_gap=features['n_gap_calls']>0)
    return {'difficulty_rule_version':RULE_VERSION,
        'difficulty':{'Q_study':study_features(metadata),'Q_case':q_level(mapping),'P':p_level(features)},
        'difficulty_features':{'Q_case':mapping,'P':features}}
