"""Independent Pandas count checks; never call API filter/base implementations."""
from __future__ import annotations
from typing import Any
import pandas as pd


def base(data: pd.DataFrame, cards: dict[str, Any], variable: str) -> pd.Series:
    card = cards[variable]
    if card['answer_type'] == 'select_all':
        siblings = [v for v,c in cards.items() if c['block_id'] == card['block_id']]
        return data[siblings].notna().any(axis=1)
    return data[variable].notna() & ~data[variable].isin(['Prefer not to say'])


def predicate(spec: dict[str, Any], data: pd.DataFrame, cards: dict[str, Any]) -> tuple[pd.Series,pd.Series]:
    """Return membership and universe, including the API's intersection OR rule."""
    op = spec['op']
    if op in {'eq','eligible'}:
        universe = base(data,cards,spec['variable'])
        return (universe & data[spec['variable']].isin(spec['values']) if op=='eq' else universe), universe
    if op == 'not':
        mask, universe = predicate(spec['arg'],data,cards)
        return universe & ~mask, universe
    (left,lu),(right,ru) = [predicate(arg,data,cards) for arg in spec['args']]
    universe = lu & ru
    return (left & right if op=='and' else left | right) & universe, universe


def counts(required: dict[str, Any], data: pd.DataFrame, metadata: dict[str, Any]) -> list[tuple[int,int]]:
    cards = {c['id']:c for c in metadata['variables']}
    v, operation = required['variable'],required['operation']
    eligible = base(data,cards,v)
    if operation.startswith('top_box_'):
        order = cards[v]['ordered_values']
        eligible &= data[v].isin(order)
        selected = data[v].isin(order[-int(operation.rsplit('_',1)[1]):])
    else:
        selected = data[v].eq(True if operation=='selection' else operation[6:]).fillna(False)
    if required.get('population'):
        eligible &= predicate(required['population'],data,cards)[0]
    groups = (list(required['group_filters'].values()) if required.get('group_filters') else
              [{'op':'eq','variable':required['group_variable'],'values':vs} for vs in required['groups'].values()])
    return [(int((eligible & predicate(g,data,cards)[0]).sum()),
             int((eligible & predicate(g,data,cards)[0] & selected).sum())) for g in groups]
