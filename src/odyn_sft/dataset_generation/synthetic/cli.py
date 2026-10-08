"""Synthetic dataset actions exposed through the unified odyn-dataset CLI.

Parser construction imports no simulator or tokenizer. Dispatch performs only
local generation/validation; no provider calls or model training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from ...common.storage import read_json, write_json


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Attach synthetic subcommands to the public dataset parser."""
    parser.set_defaults(handler=dispatch)
    sub = parser.add_subparsers(dest='action', required=True)
    spec = sub.add_parser('spec', help='Write a seeded StudySpec from a domain template')
    spec.add_argument('--domain', dest='study_domain', required=True)
    spec.add_argument('--study-id', required=True)
    spec.add_argument('--seed', type=int, required=True)
    spec.add_argument('--population-size', type=int, default=200_000)
    spec.add_argument('--output', required=True)
    study = sub.add_parser('study', help='Generate one study (visible + hidden artefacts) from a StudySpec')
    study.add_argument('--spec', required=True)
    study.add_argument('--output', required=True)
    build = sub.add_parser('build', help='Generate studies and study/family-isolated benchmark splits')
    build.add_argument('--config', required=True)
    build.add_argument('--output', required=True)
    check = sub.add_parser('validate', help='Validate a generated synthetic benchmark')
    check.add_argument('--run', required=True)
    curriculum = sub.add_parser('curriculum', help='Generate a fresh paired curriculum, with no LLM calls')
    curriculum.add_argument('--config', required=True)
    curriculum.add_argument('--output', required=True)
    resume = sub.add_parser('resume-client', help='Finish client drafts from a completed partial curriculum')
    resume.add_argument('--config',required=True)
    resume.add_argument('--partial-run',required=True)
    resume.add_argument('--output',required=True)
    curriculum_check = sub.add_parser('validate-curriculum', help='Check hashes, counts, alignment and isolation')
    curriculum_check.add_argument('--run', required=True)
    filtered = sub.add_parser('filter', help='Export an aligned small experiment from curriculum metadata')
    filtered.add_argument('--run', required=True)
    filtered.add_argument('--output', required=True)
    filtered.add_argument('--split', choices=('train','validation','test'), default='train')
    filtered.add_argument('--limit', type=int, default=10)
    filtered.add_argument('--task-type', choices=('api_tutorial','research_claim'))
    filtered.add_argument('--use-case')
    filtered.add_argument('--api-operation')
    filtered.add_argument('--q', choices=('low','medium','high'))
    filtered.add_argument('--p', choices=('low','medium','high'))
    filtered.add_argument('--max-variables', type=int)
    filtered.add_argument('--max-tokens', type=int)


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the selected local synthetic stage."""
    if args.action == 'spec':
        from .templates import template_spec
        output = Path(args.output)
        if output.exists():
            raise ValueError('Refusing to overwrite '+str(output))
        value = template_spec(args.study_domain, args.study_id, args.seed, args.population_size)
        write_json(output, value)
        result = {'spec': str(output), 'stressors': value['intended_stressors']}
    elif args.action == 'study':
        from .spec import validate_spec
        from .survey import generate_study
        root = Path(args.output)
        data, metadata, _ = generate_study(validate_spec(read_json(args.spec)), root/'study', root/'hidden')
        result = {'study_id': metadata['study_id'], 'respondents': len(data), 'variables': len(metadata['variables'])}
    elif args.action == 'build':
        from .benchmark import build as run_build
        result = run_build(args.config, args.output, lambda event: print(json.dumps(event), file=sys.stderr, flush=True))
    elif args.action == 'curriculum':
        from .curriculum import build as build_curriculum
        result = build_curriculum(args.config,args.output,lambda event:print(json.dumps(event),file=sys.stderr,flush=True))
    elif args.action == 'resume-client':
        from .curriculum import resume_client
        result = resume_client(args.config,args.partial_run,args.output,lambda event:print(json.dumps(event),file=sys.stderr,flush=True))
    elif args.action == 'validate-curriculum':
        from .curriculum import validate
        result = validate(args.run)
    elif args.action == 'filter':
        from .curriculum import filter_subset
        result = filter_subset(args.run,args.output,split=args.split,limit=args.limit,
            task_type=args.task_type,use_case=args.use_case,api_operation=args.api_operation,
            q=args.q,p=args.p,max_variables=args.max_variables,max_tokens=args.max_tokens)
    elif args.action == 'validate':
        from .benchmark import validate_benchmark
        result = validate_benchmark(args.run)
    else:
        raise ValueError(f"Unknown synthetic action: {args.action}")
    return result


def main(argv: list[str] | None = None) -> int:
    """Compatibility module entry point; use the shared dataset CLI."""
    from ..cli import main as dataset_main

    return dataset_main(["synthetic", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
