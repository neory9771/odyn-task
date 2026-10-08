"""Audit the synthetic train/val and client-only test package before tokenization."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ..common.storage import file_hash, read_json

VERSION = 'synthetic-client-sft-1'

def validate_training_suite(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    manifest = read_json(root/'manifest.json')
    if manifest.get('version') != VERSION:
        raise ValueError('Unsupported hybrid training package')
    for name, expected in manifest['files'].items():
        if file_hash(root/name) != expected:
            raise ValueError('Dataset file changed: '+name)
    identities: dict[tuple[str,str], str] = {}
    for split in ('train','validation','test'):
        path = root/f'{split}_lineage.jsonl'
        lineage = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(lineage) != manifest['counts'][split]:
            raise ValueError('Unexpected split size: '+split)
        for item in lineage:
            source = 'client_2022' if split == 'test' else 'synthetic'
            if item['source'] != source or item['split'] != split:
                raise ValueError('Client data entered fitting, or split label is incorrect')
            for kind, keys in [('record', [item['record_id']]), ('study', [item['study_id']]),
                               ('family',[item['family_id']]), ('atomic',item['component_keys'])]:
                for key in keys:
                    previous = identities.get((kind,key))
                    if previous is not None and (previous != split or kind == 'record'):
                        raise ValueError('Dataset identity overlap: '+kind)
                    identities[(kind,key)] = split
        filename = f'{split}_analysis'+('_gold' if split=='test' else '')+'.jsonl'
        rows = (json.loads(line) for line in (root/filename).open() if line.strip())
        for row, item in zip(rows,lineage,strict=True):
            if row['metadata'] != item:
                raise ValueError('Record and lineage metadata differ')
            ast.parse(row['output'])
        if split == 'test':
            inputs = [json.loads(line) for line in (root/'test_analysis_inputs.jsonl').open() if line.strip()]
            if len(inputs) != len(lineage) or any('output' in row for row in inputs):
                raise ValueError('Test gold entered test input')
    return {'passed':True,'counts':manifest['counts'],'study_family_atomic_disjoint':True,
            'client_used_for_fitting':False,'test_used_for_selection':False,
            'expert_review_complete':False}


def validate_small_training_suite(directory: str | Path) -> dict[str, Any]:
    """Audit the legacy smallest-case training/validation package, without test access."""
    root = Path(directory)
    manifest = read_json(root/'manifest.json')
    if manifest.get('version') != 'synthetic-smallest100-sft-1':
        raise ValueError('Unsupported small training package')
    for name, expected in manifest['files'].items():
        if file_hash(root/name) != expected:
            raise ValueError('Dataset file changed: '+name)
    owners: dict[tuple[str,str], str] = {}
    atomic_ids_available = True
    for split in ('train','validation'):
        lineage = [json.loads(l) for l in (root/f'{split}_lineage.jsonl').read_text().splitlines() if l.strip()]
        rows = [json.loads(l) for l in (root/f'{split}_analysis.jsonl').read_text().splitlines() if l.strip()]
        if len(lineage) != manifest['counts'][split] or not lineage:
            raise ValueError('Unexpected small split size: '+split)
        for row,item in zip(rows,lineage,strict=True):
            if item['split'] != split or item['source'] != 'synthetic' or row['metadata'] != item:
                raise ValueError('Small record/lineage source mismatch')
            ast.parse(row['output'])
            atomic_ids_available = atomic_ids_available and bool(item.get('component_keys'))
            for kind,keys in [('record',[item['record_id']]),('study',[item['study_id']]),
                              ('family',[item['family_id']]),('atomic',item.get('component_keys', []))]:
                for key in keys:
                    previous=owners.get((kind,key))
                    if previous is not None and (previous != split or kind=='record'):
                        raise ValueError('Small dataset identity overlap: '+kind)
                    owners[(kind,key)]=split
    return {'passed':True,'counts':manifest['counts'],'study_family_disjoint':True,'atomic_disjoint_verified':atomic_ids_available,
            'client_used_for_fitting':False,'test_used_for_selection':False,
            'expert_review_complete':False}
