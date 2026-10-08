"""Generate, audit and filter a fresh two-call synthetic curriculum locally.

Metadata stays beside Alpaca fields and must never be appended to model inputs.
The client study supplies review drafts, never synthetic training observations.
"""
from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Iterator
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
from typing import Any, TextIO

import pandas as pd
import tiktoken

from ...survey_api import CALLS, VERSION as API_VERSION, IMPLEMENTATION_VERSION, implementation_hashes
from ...survey_metadata.catalogue import numeric_program, numeric_references
from ...prompts.survey import ANALYSIS_INSTRUCTION, SYNTHESIS_INSTRUCTION, PROMPT_VERSION, analysis_input, synthesis_input
from ...common.storage import digest, file_hash, read_json, write_json
from .complexity import annotate, program_features, study_features
from .curriculum_cases import lesson, run_case, study_blueprints, program_for
from .curriculum_studies import DOMAINS, minimal_study, scaled_spec, subset_study
from .survey import generate_study

VERSION = 'synthetic-curriculum-1.0.0'
SPLITS = ('train','validation','test')
DEFAULTS: dict[str, Any] = {
    'seed':20261007,'counts':{'train':5000,'validation':1000,'test':1000},
    'tutorial_fraction':.1,'cases_per_study':40,'widths':[20,64,128,256,392,640],
    'population_size':20000,'max_context_tokens':32768,'client_calibration_size':100,
    'client_pool_limit':5000,'client_data':'data/quant-data-example/data/respondents.parquet',
    'client_metadata':'runs/pilot/metadata.json','client_scenario_rules':{}}
ENCODING = 'o200k_base'
Row = dict[str, Any]
Progress = Callable[[Row], None]


def iter_rows(path: Path) -> Iterator[Row]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class Writer:
    """Stream paired records to bound memory even with large catalogues."""
    def __init__(self, root: Path):
        self.root = root
        self.handles: dict[str, TextIO] = {}

    def append(self, name: str, row: Row) -> None:
        if name not in self.handles:
            path = self.root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            self.handles[name] = path.open('w')
        self.handles[name].write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def load_config(path: str | Path) -> Row:
    supplied = read_json(path)
    unknown = set(supplied)-set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown curriculum config keys: '+str(sorted(unknown)))
    config = {**deepcopy(DEFAULTS),**supplied}
    if set(config['counts']) != set(SPLITS) or any(type(v) is not int or v < (0 if k == 'test' else 1) for k,v in config['counts'].items()):
        raise ValueError('counts must specify positive train/validation/test case counts')
    for key in ('cases_per_study','population_size','max_context_tokens'):
        if type(config[key]) is not int or config[key]<1:
            raise ValueError(key+' must be positive')
    if not 0 <= config['tutorial_fraction'] <= 1:
        raise ValueError('tutorial_fraction must be in [0,1]')
    if not config['widths'] or any(type(v) is not int or v<20 for v in config['widths']):
        raise ValueError('widths must contain integers >=20')
    if any(type(config[k]) is not int or config[k]<0 for k in ('client_calibration_size','client_pool_limit')):
        raise ValueError('client draft counts must be nonnegative integers')
    for key in ('client_data','client_metadata'):
        config[key] = str(Path(config[key]).resolve())
    return config


def component_keys(case: Row, study_id: str) -> list[str]:
    """Study-qualified atomic identity ignores prose, labels and assertion direction."""
    keys = []
    for a in case.get('analyses',[]):
        meaning = {k:v for k,v in a.items() if k not in {'component_id','expect','variable_defined_groups','groups','group_filters'}}
        meaning['group_sets'] = sorted(a.get('groups',{}).values(),key=lambda x:json.dumps(x,sort_keys=True))
        meaning['group_filters'] = sorted(a.get('group_filters',{}).values(),key=lambda x:json.dumps(x,sort_keys=True))
        keys.append(digest({'study':study_id,'meaning':meaning}))
    keys.extend(digest({'study':study_id,'gap':g['concept']}) for g in case.get('gaps',[]))
    if not keys:
        keys = [digest({'study':study_id,'lesson':case['pattern']})]
    return sorted(set(keys))


def check_realization(case: Row) -> None:
    """Check a controlled realization against its pre-execution blueprint.

    This catches generator drift; it is not expert validation of free-form prose.
    """
    if 'realization_components' not in case:
        return
    expected = ' '.join(case['realization_components'])+case['realization_population']
    if expected != case['claim']:
        raise ValueError('Controlled claim differs from its committed realization')
    for a,text in zip(case['analyses'],case['realization_components']):
        labels = list(a['groups'])
        if not all(label in text for label in labels):
            raise ValueError('Claim dropped a named group')
        direction = ('More' if a['expect']=='>' else 'Fewer') if a['threshold'] is not None else ('more often' if a['expect']=='>' else 'less often')
        if direction not in text:
            raise ValueError('Claim dropped its precommitted direction')


def dimensions(case: Row, metadata: Row) -> Row:
    """Evidence direction is observed after execution, never chosen to fit prose."""
    cards = {c['id']:c for c in metadata['variables']}
    measured = len(case.get('analyses',[]))
    gaps = case.get('gaps',[])
    if case['pattern'] in CALLS:
        measured = sum(r['kind']=='estimate' for r in case['evidence']['results'])
        gaps = [{'concept':r.get('concept','unmeasured construct')} for r in case['evidence']['results'] if r['kind']=='note' and r['label']=='not_measured']
    labels = {'consistent':'support','inconsistent':'contradict','no_clear_difference':'inconclusive','unavailable':'unavailable'}
    directions = sorted({labels[r['label']] for r in case['evidence']['results'] if r['kind']=='contrast' and r['label'] in labels})
    domains = {'statistical_inference'} if measured else set()
    domain_by_gap = {'causal effect':'causal_identification','population prevalence':'survey_population_inference',
                     'transportability':'transportability','temporal comparison':'temporal_comparability',
                     'observed transactions':'measurement_construct_validity'}
    domains.update(domain_by_gap.get(g['concept'],'measurement_construct_validity') for g in gaps)
    if gaps:
        directions.append('not_applicable' if measured else 'unavailable')
    required = case['analysis_contract']['required_variables']
    types = sorted({cards[v]['answer_type'] for v in required})
    stress = sorted({'denominator_ambiguity','routing_ambiguity'} if 'select_all' in types else set())
    return {'C':['linked' if gaps and measured else 'multiple' if measured>1 else 'single'],
            'D':sorted(domains),'A':['partially_available' if gaps and measured else 'unavailable' if gaps else 'available'],
            'E':directions,'R':types,'S':stress}


def emit(writer: Writer, case: Row, metadata: Row, split: str, study_dir: str,
         study_seed: int, tutorial: bool, config: Row, encoder: Any) -> Row:
    record_id = 'syn_'+digest({'study':metadata['study_id'],'family':case['family_id']})[:24]
    references = numeric_references(metadata['variables'])
    program = numeric_program(case['program'],references)
    analysis = {'instruction':ANALYSIS_INSTRUCTION,'input':analysis_input(case['claim'],metadata),'output':program}
    synthesis = {'instruction':SYNTHESIS_INSTRUCTION,'input':synthesis_input(case['claim'],case['program'],case['evidence'],metadata),
                 'output':case['reference_perspective']}
    token_counts = {}
    for name,row in (('analysis',analysis),('synthesis',synthesis)):
        input_count = len(encoder.encode(row['instruction']+'\n'+row['input']))
        output_count = len(encoder.encode(row['output']))
        token_counts[name+'_input'] = input_count
        token_counts[name+'_output'] = output_count
        if input_count+output_count > config['max_context_tokens']:
            raise ValueError(f'{record_id}: {name} exceeds the token budget; no truncation permitted')
    difficulty = annotate(case['claim'],case['program'],metadata,case['analysis_contract'])
    lineage = {'record_id':record_id,'study_id':metadata['study_id'],'study_dir':study_dir,
        'family_id':case['family_id'],'component_keys':component_keys(case,metadata['study_id']),
        'split':split,'source':'synthetic','domain':metadata['study'].get('label'),'task_type':'api_tutorial' if tutorial else 'research_claim',
        'use_case':case['pattern'],'api_operations':difficulty['difficulty_features']['P']['api_operations'],
        'benchmark_dimensions':dimensions(case,metadata),**difficulty,
        'generation':{'version':VERSION,'study_seed':study_seed,'case_seed':case['seed'],
            'method':'minimal_api_lesson' if tutorial else 'contract_first_controlled_realization',
            'study_generator':'seeded_minimal_survey' if tutorial else 'finite_population_sampling_response_measurement',
            'study_spec_path':None if tutorial else str(Path(study_dir).parent/'hidden/study_spec.json'),
            'family_selection':'seeded_case_pattern_mix' if not tutorial else 'round_robin_21_functions',
            'claim_style':'natural_paraphrase' if case.get('natural') else 'explicit_catalogue',
            'claim_validation':'controlled_checked_expert_pending','direction_commitment':case.get('direction_commitment'),
            'llm_calls':0,'reference_execution':'native_python',
            'reference_semantics_passed':case['reference_semantics_passed'],
            'independent_count_checks':case['independent_count_checks']},
        'tokens':token_counts,'token_encoding':ENCODING,'prompt_version':PROMPT_VERSION,
        'api_version':API_VERSION,'api_implementation_version':IMPLEMENTATION_VERSION,
        'review_status':'draft_pending_expert_review','allowed_for_sft':split=='train'}
    for task,row in (('analysis',analysis),('synthesis',synthesis)):
        row['metadata'] = lineage
        writer.append(f'{split}_{task}'+('_gold' if split=='test' else '')+'.jsonl',row)
        if split=='test':
            writer.append(f'test_{task}_inputs.jsonl',{k:v for k,v in row.items() if k!='output'})
    writer.append(f'{split}_lineage.jsonl',lineage)
    writer.append('hidden/private_gold.jsonl',{'record_id':record_id,'metadata':lineage,'claim':case['claim'],
        'analysis_contract':case['analysis_contract'],'synthesis_contract':case['synthesis_contract'],
        'program':case['program'],'evidence':case['evidence'],'reference_perspective':case['reference_perspective']})
    return lineage


def fingerprints() -> Row:
    from ..provenance import implementation_hashes as generator_hashes

    return generator_hashes()


def summary(rows: list[Row]) -> Row:
    report = {}
    for split in SPLITS:
        selected = [r for r in rows if r['split']==split]
        if not selected:
            report[split] = {'cases':0, 'studies':0, 'status':'disabled'}
            continue
        cells = Counter(r['difficulty']['Q_case']+'/'+r['difficulty']['P'] for r in selected)
        report[split] = {'cases':len(selected),'studies':len({r['study_id'] for r in selected}),
            'task_types':dict(Counter(r['task_type'] for r in selected)),
            'use_cases':dict(Counter(r['use_case'] for r in selected)),
            'api_operations':dict(Counter(op for r in selected for op in r['api_operations'])),
            'Q_by_P':{q+'/'+p:cells[q+'/'+p] for q in ('low','medium','high') for p in ('low','medium','high')},
            'primary_cells_below_50':[q+'/'+p for q in ('low','medium','high') for p in ('low','medium','high') if cells[q+'/'+p]<50],
            'benchmark_dimensions':{d:dict(Counter(t for r in selected for t in r['benchmark_dimensions'][d])) for d in ('C','D','A','E','R','S')},
            'token_ranges':{k:{'min':min(r['tokens'][k] for r in selected),'max':max(r['tokens'][k] for r in selected)} for k in selected[0]['tokens']}}
    return report


def structural_envelope(rows: list[Row], client: Row | None) -> Row:
    """Report the requested G4 bracket; do not turn an unmet gate into a pass."""
    train = [r['difficulty']['Q_study'] for r in rows if r['split']=='train']
    features = ('variable_count','block_count','answer_format_count','near_duplicate_block_count',
                'duplicate_label_group_count','routed_block_count','select_all_block_count')
    bounds = {k:{'min':min(r[k] for r in train),'max':max(r[k] for r in train)} for k in features}
    target = study_features(client) if client else None
    checks = {k:bounds[k]['min'] <= target[k] <= bounds[k]['max'] for k in features} if target else {}
    return {'status':'passed' if checks and all(checks.values()) else 'unmet' if checks else 'not_checked',
            'train_bounds':bounds,'client_features':target,'checks':checks,
            'note':'Structural bracket is necessary, not proof of semantic transfer or representativeness.'}


def capability_matrix(rows: list[Row]) -> Row:
    """Full coverage needs judge validation; executed lessons alone are partial."""
    advanced = {'mean_score','trend','association','adjusted_compare'}
    return {op:{'api_implemented':True,'generator':'minimal_api_lesson',
        'reference_examples':{s:sum(op in r['api_operations'] for r in rows if r['split']==s) for s in SPLITS},
        'scorer':'population/estimand contract' if op in {'compare','group','proportion','top_box','where','eligible'} else 'record-shape check; semantic rubric pending',
        'judge_rubric_status':'pending_expert_validation','cell_status':'partial',
        'client_v1_scope':'excluded_auxiliary_teaching' if op in advanced else 'core'} for op in sorted(CALLS)}


def client_drafts(root: Path, writer: Writer, config: Row, progress: Progress | None) -> Row:
    """Freeze ownership before executions; export an unselected client pool.

    No LLM/judge/model calls occur. Expert review and the protocol freeze are
    required before selecting the final 1,000 client test cases.
    """
    if not config['client_calibration_size'] and not config['client_pool_limit']:
        return {'status':'disabled','final_test_selected':False,'calibration_cases':0,'candidate_cases':0}
    from random import Random
    from ...survey_api import run_reference, normalize_data
    from ..partition import enumerate_family_specs
    from ..dataset.families import execute_family
    from ..generation import candidates, materialize, validate_reference, reference_perspective
    from ..generation.candidates import keys
    from ..contracts import analysis_contract, synthesis_contract
    from ..selection import DEFAULT_MINIMA, scaled_minima, select
    metadata = read_json(config['client_metadata'])
    data = normalize_data(pd.read_parquet(config['client_data']),metadata)
    specs = enumerate_family_specs(metadata,config['seed'])
    Random(config['seed']).shuffle(specs)
    owners = {s['core']:('calibration' if int(digest({'core':s['core'],'seed':config['seed'],'purpose':'client-partition'})[:16],16)/2**64 < .1 else 'test_candidate') for s in specs}
    write_json(root/'client/family_owners.json',{'partition_before_execution':True,'owners':owners,'hash':digest(owners)})
    limits = {'calibration':max(800,config['client_calibration_size']*8) if config['client_calibration_size'] else 0,
              'test_candidate':max(2500,math.ceil(config['client_pool_limit']/2))}
    selected_specs = []
    sizes: Counter[str] = Counter()
    for spec in specs:
        owner = owners[spec['core']]
        if sizes[owner]<limits[owner]:
            selected_specs.append(spec)
            sizes[owner] += 1
    write_json(root/'client/execution_plan.json',{'result_blind':True,'owned_inventory':len(specs),
        'executed_family_ids':[s['core'] for s in selected_specs],'execution_limits':limits,
        'note':'A seeded metadata-only inventory prefix is executed; remaining owned families are unexecuted.'})
    inventory: dict[str,list[Row]] = {'calibration':[],'test_candidate':[]}
    for i,spec in enumerate(selected_specs):
        inventory[owners[spec['core']]].append(execute_family(spec,data,metadata,config['seed']))
        if progress and i%250==0:
            progress({'stage':'client_reference_inventory','completed':i,'total':len(selected_specs)})
    cal_pool = candidates(inventory['calibration'],config['seed']+1,set(),config['client_scenario_rules']) if config['client_calibration_size'] else []
    minima = scaled_minima({d:values for d,values in DEFAULT_MINIMA.items() if d!='holdout'},config['client_calibration_size']/1000)
    # These are calibration draft margins, not the final-test design.
    chosen,_,shortages = select(cal_pool,config['client_calibration_size'],config['seed']+2,minima,{},[])
    if len(chosen) != config['client_calibration_size'] or shortages:
        raise ValueError('Client calibration draft capacity insufficient: '+str(shortages))
    pool = candidates(inventory['test_candidate'],config['seed']+3,set(),config['client_scenario_rules'])[:config['client_pool_limit']]
    if len(pool)<config['client_pool_limit']:
        raise ValueError('Client candidate pool is smaller than the configured limit')
    reserved = {k for k,v in owners.items() if v=='calibration'}
    if any(set(keys(candidate)) & reserved for candidate in pool):
        raise ValueError('Client candidate reuses a calibration-owned atomic family')
    for split,cases in (('calibration',chosen),('test_candidate',pool)):
        for i,candidate in enumerate(cases):
            case = materialize(candidate,split,config['seed'],metadata)
            case['case_id'] = 'client_'+digest({'family':case['family_id'],'scenario':case['scenario'],'keys':case['component_keys']})[:24]
            case['evidence'] = run_reference(case['program'],data,metadata)
            checks = validate_reference(case,case['evidence'])
            gold = analysis_contract(case)
            gold['required_variables'] = sorted({a['variable'] for a in gold['required_analyses']})
            gold['available_functions'] = sorted(CALLS)
            lineage = {'record_id':case['case_id'],'study_id':'quant_us_2022','split':split,
                'source':'client_2022','family_id':case['family_id'],'component_keys':case['component_keys'],
                'task_type':'research_claim','use_case':case['scenario'],**annotate(case['claim'],case['program'],metadata,gold),
                'benchmark_dimensions':candidate['dimensions'],
                'generation':{'version':VERSION,'seed':config['seed'],'method':'client_metadata_family_partition_then_execution',
                    'llm_calls':0,'independent_count_checks':checks,'claim_validation':'controlled_template_pending_expert_review'},
                'review_status':'draft_pending_expert_and_judge_review','allowed_for_sft':False,'selected_for_final_test':False}
            private = {**case,'metadata':lineage,'analysis_contract':gold,'synthesis_contract':synthesis_contract(case),
                       'reference_perspective':reference_perspective(case)}
            writer.append(f'client/{split}_private_gold.jsonl',private)
            if split=='calibration':
                writer.append('client/calibration_analysis_inputs.jsonl',{'instruction':ANALYSIS_INSTRUCTION,
                    'input':analysis_input(case['claim'],metadata),'metadata':lineage})
                writer.append('client/calibration_synthesis_oracle_inputs.jsonl',{'instruction':SYNTHESIS_INSTRUCTION,
                    'input':synthesis_input(case['claim'],case['program'],case['evidence'],metadata),'metadata':lineage})
            if progress and i%250==0:
                progress({'stage':'client_drafts','split':split,'completed':i,'total':len(cases)})
    return {'status':'draft_pending_review','calibration_cases':len(chosen),'candidate_cases':len(pool),
        'candidate_pool_requested_limit':config['client_pool_limit'],'final_test_target':1000,'final_test_selected':False,
        'source_sha256':file_hash(config['client_data']),'metadata_sha256':file_hash(config['client_metadata']),
        'family_ownership_counts':dict(Counter(owners.values())),
        'calibration_coverage_minima':minima,'calibration_shortages':shortages,
        'note':'Pool cases may share atomic families with each other; final selection must enforce atomic disjointness, expert review, balanced margins and G0–G8 freeze.'}


def build(config_path: str | Path, output: str | Path, progress: Progress | None = None) -> Row:
    """Build in staging; publish only after paired alignment and isolation checks."""
    config = load_config(config_path)
    output = Path(output).resolve()
    staging = output.with_name(output.name+'.partial')
    if output.exists() or staging.exists():
        raise ValueError('Use a fresh output directory (including its .partial staging path)')
    staging.mkdir(parents=True)
    initial_hashes = fingerprints()
    writer, rows, encoder = Writer(staging), [], tiktoken.get_encoding(ENCODING)
    write_json(staging/'config.json',config)
    try:
        for split_index,split in enumerate(SPLITS):
            total = config['counts'][split]
            if total == 0:
                for name in ('test_lineage.jsonl','test_analysis_gold.jsonl','test_synthesis_gold.jsonl','test_analysis_inputs.jsonl','test_synthesis_inputs.jsonl'):
                    (staging/name).touch()
                continue
            tutorials = round(total*config['tutorial_fraction'])
            operations = sorted(CALLS)
            for index in range(tutorials):
                seed = config['seed']+split_index*100000+index
                study_id = f'{split}_lesson_{index:04d}'
                data,metadata,ids = minimal_study(study_id,seed)
                blueprint = lesson(operations[index%len(operations)],ids,seed)
                data,metadata = subset_study(data,metadata,blueprint['variables'])
                study_dir = 'studies/'+study_id+'/study'
                public = staging/study_dir
                public.mkdir(parents=True)
                data.to_parquet(public/'respondents.parquet',index=False)
                write_json(public/'metadata.json',metadata)
                write_json(public/'codebook.json',metadata['variables'])
                write_json(public/'study_description.json',metadata['study'])
                write_json(staging/'studies'/study_id/'hidden/generation.json',{'seed':seed,'generator':'minimal_study','method':'seeded_balanced_service_survey','status':'benchmark_only_not_candidate_input'})
                case = run_case(blueprint,data,metadata,tutorial=True)
                if blueprint['pattern'] not in program_features(case['program'],case['analysis_contract'])['api_operations']:
                    raise ValueError('Teaching program omitted the requested API call')
                rows.append(emit(writer,case,metadata,split,study_dir,seed,True,config,encoder))
                if progress and index%25==0:
                    progress({'stage':'tutorials','split':split,'completed':index,'total':tutorials})
            remaining = total-tutorials
            for index in range(math.ceil(remaining/config['cases_per_study'])):
                count = min(config['cases_per_study'],remaining-index*config['cases_per_study'])
                seed = config['seed']+split_index*100000+10000+index
                study_id = f'{split}_research_{index:04d}'
                width = config['widths'][index%len(config['widths'])]
                domain = DOMAINS[index%len(DOMAINS)]
                spec,roles = scaled_spec(study_id,seed,domain,width,config['population_size'])
                study_dir = 'studies/'+study_id+'/study'
                data,metadata,population = generate_study(spec,staging/study_dir,staging/'studies'/study_id/'hidden')
                del population
                blueprints = study_blueprints(metadata,roles,count,seed)
                write_json(staging/'studies'/study_id/'hidden/family_assignment.json',{'split':split,
                    'before_reference_execution':True,'study_id':study_id,
                    'families':{b['family_id']:component_keys(b,study_id) for b in blueprints}})
                for blueprint in blueprints:
                    check_realization(blueprint)
                    case = run_case(blueprint,data,metadata)
                    rows.append(emit(writer,case,metadata,split,study_dir,seed,False,config,encoder))
                if progress:
                    progress({'stage':'research','split':split,'study':index+1,'width':width,'completed':tutorials+min(remaining,(index+1)*config['cases_per_study']),'total':total})
        client = client_drafts(staging,writer,config,progress)
        writer.close()
        client_metadata = read_json(config['client_metadata']) if Path(config['client_metadata']).exists() else None
        audit = summary(rows)
        write_json(staging/'coverage.json',audit)
        structural = structural_envelope(rows,client_metadata)
        write_json(staging/'structural_envelope.json',structural)
        write_json(staging/'capability_matrix.json',capability_matrix(rows))
        write_json(staging/'gates.json',{'G0':'fresh_generation_no_prior_case_or_adapter_reuse',
            'G1':'partial_judge_validation_pending','G2':'controlled_checks_passed_expert_review_pending',
            'G3':'contract_and_count_checks_passed_free_language_review_pending','G4':structural['status'],
            'G5':'coverage_audit_available_transfer_validation_pending','G6':'client_calibration_draft_not_evaluated',
            'G7':'expert_gold_and_judge_validation_pending','G8':'protocol_and_model_freeze_pending',
            'final_client_selection_permitted':False})
        (staging/'README.md').write_text(readme())
        if fingerprints()!=initial_hashes:
            raise ValueError('Generator implementation changed during this build; rerun from frozen code')
        manifest = {'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),'counts':config['counts'],
            'record_unit':'one claim with paired analysis and synthesis tasks','llm_calls':0,'training_performed':False,
            'client_final_test_selected':False,'client':client,'prompt_version':PROMPT_VERSION,
            'implementation_hashes':initial_hashes,'token_encoding':ENCODING,'max_context_tokens':config['max_context_tokens'],
            'token_budget_note':'Measured using o200k_base; remeasure with the selected training tokenizer and chat template before fitting.',
            'family_policy':'Study-qualified families are isolated across splits. Shared abstract API skills across new studies intentionally test transfer.',
            'files':{str(p.relative_to(staging)):file_hash(p) for p in sorted(staging.rglob('*')) if p.is_file()}}
        write_json(staging/'manifest.json',manifest)
        validation = validate(staging)
        staging.rename(output)
        return {'output':str(output),'counts':config['counts'],'validation':validation,'client':client,'llm_calls':0}
    except BaseException:
        writer.close()
        # Preserve failure evidence for inspection; never present a partial run as complete.
        raise


def validate(directory: str | Path) -> Row:
    """Verify hashes, metadata/pair alignment and study/family/atomic isolation."""
    root = Path(directory)
    manifest = read_json(root/'manifest.json')
    expected_files = set(manifest['files']) | {'manifest.json'}
    actual_files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if expected_files != actual_files:
        raise ValueError('Manifest file inventory mismatch')
    for name,expected in manifest['files'].items():
        if file_hash(root/name) != expected:
            raise ValueError('File hash mismatch: '+name)
    ownership: dict[tuple[str,str], str] = {}
    ids: set[str] = set()
    checked = 0
    for split in SPLITS:
        lineage = list(iter_rows(root/f'{split}_lineage.jsonl'))
        if len(lineage) != manifest['counts'][split]:
            raise ValueError('Split case count mismatch: '+split)
        for item in lineage:
            if item['record_id'] in ids or item['source']!='synthetic' or item['split']!=split:
                raise ValueError('Repeated record or invalid split/source')
            ids.add(item['record_id'])
            for kind,values in (('study',[item['study_id']]),('family',[item['family_id']]),('atomic',item['component_keys'])):
                for value in values:
                    key = (kind,value)
                    if key in ownership and ownership[key]!=split:
                        raise ValueError('Cross-split '+kind+' overlap')
                    ownership[key] = split
        for task in ('analysis','synthesis'):
            name = f'{split}_{task}'+('_gold' if split=='test' else '')+'.jsonl'
            seen = 0
            for row,item in zip(iter_rows(root/name),lineage,strict=True):
                if row.get('metadata') != item or set(row)!={'instruction','input','output','metadata'}:
                    raise ValueError('Paired record/lineage mismatch')
                if task=='analysis':
                    ast.parse(row['output'])
                payload = json.loads(row['input'])
                if any(key in payload for key in ('metadata','difficulty','generation','family_id','analysis_contract','synthesis_contract')):
                    raise ValueError('Private lineage entered a model prompt')
                study = payload['survey_specification']['study']
                if any(key in study for key in ('latent_variables','population_truth','dgp','oracle_weight')):
                    raise ValueError('Hidden DGP entered the candidate study')
                seen += 1
            checked += seen
            if split=='test':
                for public,private in zip(iter_rows(root/f'test_{task}_inputs.jsonl'),iter_rows(root/name),strict=True):
                    if public!={k:v for k,v in private.items() if k!='output'}:
                        raise ValueError('Test input contains gold or has drifted')
    gold_ids = [r['record_id'] for r in iter_rows(root/'hidden/private_gold.jsonl')]
    if len(gold_ids)!=len(ids) or set(gold_ids)!=ids:
        raise ValueError('Private gold is not aligned with exported cases')
    client_rows: dict[str,list[Row]] = {}
    for split in ('calibration','test_candidate'):
        path = root/f'client/{split}_private_gold.jsonl'
        client_rows[split] = [row['metadata'] for row in iter_rows(path)] if path.exists() else []
        record_ids = [r['record_id'] for r in client_rows[split]]
        if len(set(record_ids))!=len(record_ids) or any(r['allowed_for_sft'] or r['selected_for_final_test'] for r in client_rows[split]):
            raise ValueError('Invalid client draft identity/release state')
    if len(client_rows['calibration'])!=manifest['client']['calibration_cases'] or len(client_rows['test_candidate'])!=manifest['client']['candidate_cases']:
        raise ValueError('Client draft count mismatch')
    calibration_atoms = {k for r in client_rows['calibration'] for k in r['component_keys']}
    candidate_atoms = {k for r in client_rows['test_candidate'] for k in r['component_keys']}
    if calibration_atoms & candidate_atoms:
        raise ValueError('Client calibration/test candidate atomic overlap')
    if manifest['client_final_test_selected']:
        raise ValueError('This generator cannot release a final client test')
    return {'passed':True,'cases':len(ids),'paired_records':checked,'studies':sum(k[0]=='study' for k in ownership),
            'checks':['file_hashes','exact_counts','paired_metadata_alignment','private_gold_alignment','study_family_atomic_isolation','public_test_without_outputs','no_top_level_private_prompt_fields'],
            'expert_gold_certified':False,'judge_validated':False}


def filter_subset(directory: str | Path, output: str | Path, *, split: str = 'train', limit: int = 10,
                  task_type: str | None = None, use_case: str | None = None, api_operation: str | None = None,
                  q: str | None = None, p: str | None = None, max_variables: int | None = None,
                  max_tokens: int | None = None) -> Row:
    """Select aligned tasks/gold and copy their required public/hidden studies."""
    source, destination = Path(directory).resolve(),Path(output).resolve()
    if split not in SPLITS or limit<1:
        raise ValueError('Choose a valid split and positive limit')
    if destination.exists():
        raise ValueError('Use a fresh subset directory')
    selected = []
    for row in iter_rows(source/f'{split}_lineage.jsonl'):
        if ((task_type and row['task_type']!=task_type) or (use_case and row['use_case']!=use_case)
            or (api_operation and api_operation not in row['api_operations'])
            or (q and row['difficulty']['Q_case']!=q) or (p and row['difficulty']['P']!=p)
            or (max_variables is not None and row['difficulty']['Q_study']['variable_count']>max_variables)
            or (max_tokens is not None and max(row['tokens'][t+'_input']+row['tokens'][t+'_output'] for t in ('analysis','synthesis'))>max_tokens)):
            continue
        selected.append(row)
        if len(selected)==limit:
            break
    if not selected:
        raise ValueError('No cases match the filters')
    ids = {r['record_id'] for r in selected}
    destination.mkdir(parents=True)
    writer = Writer(destination)
    try:
        names = [f'{split}_lineage.jsonl','hidden/private_gold.jsonl']
        names.extend(f'{split}_{task}'+('_gold' if split=='test' else '')+'.jsonl' for task in ('analysis','synthesis'))
        if split=='test':
            names.extend(f'test_{task}_inputs.jsonl' for task in ('analysis','synthesis'))
        for name in names:
            for row in iter_rows(source/name):
                if row.get('record_id',row.get('metadata',{}).get('record_id')) in ids:
                    writer.append(name,row)
        for study in sorted({r['study_id'] for r in selected}):
            shutil.copytree(source/'studies'/study,destination/'studies'/study)
    finally:
        writer.close()
    manifest = {'version':VERSION,'source':str(source),'source_manifest_sha256':file_hash(source/'manifest.json'),
                'split':split,'cases':len(selected),'paired_records':2*len(selected),
                'filters':{'task_type':task_type,'use_case':use_case,'api_operation':api_operation,'Q_case':q,'P':p,
                           'max_variables':max_variables,'max_tokens':max_tokens},
                'record_ids':[r['record_id'] for r in selected],
                'files':{str(f.relative_to(destination)):file_hash(f) for f in destination.rglob('*') if f.is_file()}}
    write_json(destination/'subset_manifest.json',manifest)
    return {'output':str(destination),'cases':len(selected),'paired_records':2*len(selected)}


def readme() -> str:
    return '''# Fresh synthetic two-call curriculum

Case counts count claims; each claim produces an analysis-code and an evidence-synthesis record.
`train_analysis.jsonl` and `train_synthesis.jsonl` have Alpaca instruction/input/output fields.
Validation uses the same format. Test inputs omit output; `test_*_gold.jsonl` and `hidden/` are evaluator-only.
Every record also has a `metadata` field, aligned with `*_lineage.jsonl`.
Feed ONLY instruction/input/output to training. Never concatenate metadata or private gold into prompts.

Metadata includes task_type (`research_claim` or `api_tutorial`), use_case, API operations, study/domain identifiers,
study/case seeds, generation method, Q_study features, Q_case/P heuristic levels and simple counts, observed evidence dimensions,
token counts and review state. Filters select both task records and their required study files.
Public study files live under `studies/<study_id>/study/`; hidden simulation truth is in the sibling `hidden/` folder.
Runtime loads the study associated with each row's metadata.study_dir; do not use one shared survey for all rows.

All programs execute against the API and all rich-case counts have independent Pandas checks.
Directions/family ownership are fixed before execution. Populations and reference synthesis come from recorded data,
never hidden population or causal truth. Minimal lessons cover all 21 API functions, including helper calls.
Advanced statistical lessons are auxiliary and are outside client-v1 capability claims.
Studies are isolated across train/validation/test; shared abstract API skills intentionally test transfer.
No earlier cases or trained adapters are reused. No LLM calls or model training occurred.

Coverage and unmet primary Q/P cells are in coverage.json. Cells below 50 are not primary reported results;
secondary cells below 30 require suppression/merging. This generation is not a model evaluation or a >80% claim.
G4 structural checks are in structural_envelope.json. Other outstanding gates are explicit in gates.json.
Client calibration is a review draft; the client candidate pool is unselected and MUST NOT be used for SFT.
The final client 1,000-case test requires expert review, judge validation, balanced selection and protocol/model freeze.
Controlled template validity and executable reference gold do not replace these gates.

Token budgets use o200k_base. Remeasure with the selected training tokenizer/chat template before fitting.
No example is truncated. Existing single-study training-package validators need multi-study adaptation before use.

Example filtering:

```bash
odyn-dataset synthetic filter \\
  --run RUN --output SUBSET --split train --limit 10 \\
  --task-type api_tutorial --max-variables 4
```
'''


def resume_client(config_path: str | Path, staging_path: str | Path, output: str | Path,
                  progress: Progress | None = None) -> Row:
    """Finish client drafts after correcting a client-only oracle issue.

    Reuses the already completed synthetic rows but replays every client family
    from the frozen metadata-only ownership plan and validates the final package.
    """
    config = load_config(config_path)
    staging, final = Path(staging_path).resolve(),Path(output).resolve()
    if not staging.is_dir() or final.exists():
        raise ValueError('Expected an existing partial run and a fresh final output')
    for split in SPLITS:
        if sum(1 for _ in iter_rows(staging/f'{split}_lineage.jsonl')) != config['counts'][split]:
            raise ValueError('Completed synthetic split is not reusable: '+split)
    # Client drafts were not exported by the failed first family; remove only
    # replayable generated client outputs, keeping ownership audit files.
    client_root = staging/'client'
    for name in ('calibration_private_gold.jsonl','calibration_analysis_inputs.jsonl',
                 'calibration_synthesis_oracle_inputs.jsonl','test_candidate_private_gold.jsonl'):
        path=client_root/name
        if path.exists():path.unlink()
    writer = Writer(staging)
    client = client_drafts(staging,writer,config,progress)
    writer.close()
    rows=[row for split in SPLITS for row in iter_rows(staging/f'{split}_lineage.jsonl')]
    write_json(staging/'coverage.json',summary(rows))
    client_metadata = read_json(config['client_metadata']) if Path(config['client_metadata']).exists() else None
    structural=structural_envelope(rows,client_metadata)
    write_json(staging/'structural_envelope.json',structural)
    write_json(staging/'capability_matrix.json',capability_matrix(rows))
    write_json(staging/'gates.json',{'G0':'fresh_generation_no_prior_case_or_adapter_reuse',
        'G1':'partial_judge_validation_pending','G2':'controlled_checks_passed_expert_review_pending',
        'G3':'contract_and_count_checks_passed_free_language_review_pending','G4':structural['status'],
        'G5':'coverage_audit_available_transfer_validation_pending','G6':'client_calibration_draft_not_evaluated',
        'G7':'expert_gold_and_judge_validation_pending','G8':'protocol_and_model_freeze_pending',
        'final_client_selection_permitted':False})
    (staging/'README.md').write_text(readme())
    manifest={'version':VERSION,'created_at':datetime.now(timezone.utc).isoformat(),'counts':config['counts'],
        'record_unit':'one claim with paired analysis and synthesis tasks','llm_calls':0,'training_performed':False,
        'client_final_test_selected':False,'client':client,'prompt_version':PROMPT_VERSION,
        'implementation_hashes':fingerprints(),'token_encoding':ENCODING,'max_context_tokens':config['max_context_tokens'],
        'token_budget_note':'Measured using o200k_base; remeasure with the selected training tokenizer and chat template before fitting.',
        'family_policy':'Study-qualified families are isolated across splits. Shared abstract API skills across new studies intentionally test transfer.',
        'files':{str(p.relative_to(staging)):file_hash(p) for p in sorted(staging.rglob('*')) if p.is_file()}}
    write_json(staging/'manifest.json',manifest)
    validation=validate(staging)
    staging.rename(final)
    return {'output':str(final),'counts':config['counts'],'validation':validation,'client':client,'llm_calls':0}
