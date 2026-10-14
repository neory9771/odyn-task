"""Assemble a raw curriculum into the trainer package `synthetic-client-sft-1`.

Synthetic train/validation come from an existing curriculum build; the test
split is rebuilt from the client survey only. No LLM calls or model outputs.
"""
from __future__ import annotations

import os
from pathlib import Path
import random
import shutil
from typing import Any

import pandas as pd

from ...common.storage import digest, file_hash, read_json, write_json, write_jsonl
from ...prompts.survey import ANALYSIS_INSTRUCTION, PROMPT_VERSION, SYNTHESIS_INSTRUCTION, analysis_input, synthesis_input
from ...survey_api import CALLS, normalize_data, run_reference
from ...survey_metadata.catalogue import numeric_program, numeric_references
from ...tasks.sft1_suite import VERSION, validate_training_suite
from .complexity import annotate
from .curriculum import Progress, iter_rows, load_config, validate


def build_client_test(root: Path, config: dict[str, Any], size: int, *, smoke: bool = False,
                      progress: Progress | None = None) -> None:
    """Select reference-checked client families without consulting any model predictions."""
    from ..contracts import analysis_contract, synthesis_contract
    from ..dataset.families import execute_family
    from ..generation import candidates, materialize, reference_perspective, validate_reference
    from ..partition import enumerate_family_specs
    from ..selection import DEFAULT_MINIMA, scaled_minima, select

    metadata = read_json(config['client_metadata'])
    data = normalize_data(pd.read_parquet(config['client_data']), metadata)
    specs = enumerate_family_specs(metadata, config['seed'])
    random.Random(config['seed']).shuffle(specs)
    # A frozen metadata-only prefix is selected before computing outcomes.
    plan = specs[:min(len(specs), 100 if smoke else max(2500, size*3))]
    write_json(root/'client_execution_plan.json', {'seed':config['seed'], 'family_ids':[r['core'] for r in plan],
               'selection_before_execution':True, 'model_outputs_used':False})
    inventory = []
    for i, spec in enumerate(plan):
        inventory.append(execute_family(spec, data, metadata, config['seed']))
        if progress and i % 100 == 0:
            progress({'stage':'client_inventory', 'completed':i, 'total':len(plan)})
    pool = candidates(inventory, config['seed']+3, set(), config['client_scenario_rules'])
    minima = {} if smoke else scaled_minima({k:v for k,v in DEFAULT_MINIMA.items() if k!='holdout'}, size/1000)
    chosen, _, shortages = select(pool, size, config['seed']+4, minima, {}, [])
    if shortages or len(chosen) != size:
        raise ValueError('Client test coverage/capacity insufficient: '+str(shortages))
    write_json(root/'client_test_selection.json', {'minima':minima, 'shortages':shortages,
               'pool_cases':len(pool), 'selected':len(chosen), 'model_outputs_used':False,
               'expert_review_complete':False, 'judge_validated':False})
    references = numeric_references(metadata['variables'])
    analysis, synthesis, gold, lineage = [], [], [], []
    for i, candidate in enumerate(chosen):
        case = materialize(candidate, 'test', config['seed'], metadata)
        record_id = 'client_'+digest({'family':case['family_id'], 'scenario':case['scenario'],
                                     'keys':case['component_keys']})[:24]
        case['case_id'] = record_id
        evidence = run_reference(case['program'], data, metadata)
        checks = validate_reference(case, evidence)
        case['evidence'] = evidence
        contract = analysis_contract(case)
        contract['required_variables'] = sorted({r['variable'] for r in contract['required_analyses']})
        contract['available_functions'] = sorted(CALLS)
        item = {'record_id':record_id, 'study_id':'quant_us_2022', 'source':'client_2022',
                'split':'test', 'family_id':case['family_id'], 'component_keys':case['component_keys'],
                'task_type':'research_claim', 'use_case':case['scenario'],
                'benchmark_dimensions':candidate['dimensions'],
                **annotate(case['claim'], case['program'], metadata, contract),
                'generation':{'seed':config['seed'], 'method':'contract_first_client_test',
                              'llm_calls':0, 'independent_count_checks':checks},
                'allowed_for_sft':False, 'review_status':'draft_pending_expert_review'}
        lineage.append(item)
        answer = reference_perspective(case)
        analysis.append({'instruction':ANALYSIS_INSTRUCTION, 'input':analysis_input(case['claim'], metadata),
                         'output':numeric_program(case['program'], references), 'metadata':item})
        synthesis.append({'instruction':SYNTHESIS_INSTRUCTION,
                          'input':synthesis_input(case['claim'], case['program'], evidence, metadata),
                          'output':answer, 'metadata':item})
        gold.append({**case, 'record_id':record_id, 'split':'test', 'metadata':item,
                     'analysis_contract':contract, 'synthesis_contract':synthesis_contract(case),
                     'reference_perspective':answer})
        if progress and i % 100 == 0:
            progress({'stage':'client_gold', 'completed':i, 'total':size})
    for name, rows in [('test_analysis_gold.jsonl',analysis), ('test_synthesis_gold.jsonl',synthesis),
                       ('test_lineage.jsonl',lineage), ('private_gold.jsonl',gold)]:
        write_jsonl(root/name, rows)
    write_jsonl(root/'test_analysis_inputs.jsonl', [{k:v for k,v in r.items() if k!='output'} for r in analysis])
    write_jsonl(root/'test_synthesis_inputs.jsonl', [{k:v for k,v in r.items() if k!='output'} for r in synthesis])
    shutil.copyfile(config['client_data'], root/'respondents.parquet')
    write_json(root/'metadata.json', metadata)


def link_or_copy(source: str, target: str) -> None:
    """Study files are never rewritten, so hard links avoid duplicating large parquet data."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def assemble(curriculum: str | Path, config_path: str | Path, output: str | Path, *, test_count: int = 1000,
             smoke: bool = False, progress: Progress | None = None) -> dict[str, Any]:
    """Copy a validated curriculum, split out validation gold and add a client-only test split."""
    source, output = Path(curriculum).resolve(), Path(output).resolve()
    staging = output.with_name(output.name+'.partial')
    if output.exists() or staging.exists():
        raise ValueError('Use a fresh output directory (including its .partial staging path)')
    audit = validate(source)
    config = load_config(config_path)
    try:
        report = _assemble(source, staging, config, audit, test_count, smoke, progress)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    staging.rename(output)
    return {'output':str(output), 'validation':report}


def _assemble(source: Path, staging: Path, config: dict[str, Any], audit: dict[str, Any], test_count: int,
              smoke: bool, progress: Progress | None) -> dict[str, Any]:
    shutil.copytree(source, staging, ignore=shutil.ignore_patterns('studies'))
    if (source/'studies').exists():
        shutil.copytree(source/'studies', staging/'studies', copy_function=link_or_copy)
    (staging/'manifest.json').rename(staging/'synthetic_manifest.json')
    write_jsonl(staging/'validation_private_gold.jsonl',
                [r for r in iter_rows(staging/'hidden/private_gold.jsonl') if r['metadata']['split']=='validation'])
    build_client_test(staging, config, test_count, smoke=smoke, progress=progress)
    counts = {**read_json(staging/'synthetic_manifest.json')['counts'], 'test':test_count}
    manifest = {'version':VERSION, 'counts':counts, 'release_status':'draft_pending_expert_review',
                'prompt_version':PROMPT_VERSION, 'training_scope':'step1_analysis_code_generation',
                'source_policy':{'train':'synthetic', 'validation':'synthetic', 'test':'client_2022'},
                'source_curriculum':str(source), 'source_curriculum_manifest_sha256':file_hash(source/'manifest.json'),
                'fresh_base_required':True, 'llm_calls':0, 'test_used_for_fitting':False,
                'client_acceptance_certified':False, 'synthetic_build_audit':audit,
                'files':{str(p.relative_to(staging)):file_hash(p) for p in staging.rglob('*') if p.is_file()}}
    write_json(staging/'manifest.json', manifest)
    return validate_training_suite(staging)

