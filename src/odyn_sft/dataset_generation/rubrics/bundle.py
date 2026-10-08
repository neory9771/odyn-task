"""Write and validate an immutable rubric bundle with source snapshots and hashes."""

import shutil
from collections import Counter
from pathlib import Path

from ...common.storage import (
    atomic_text,
    digest,
    file_hash,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from ...prompts.authoring import CURRENT_CONTRACT_VERSION, contract_manifest
from .criteria import DIMENSIONS, VERSION
from .judge_contract import JUDGE_INSTRUCTIONS, JudgeOutput
from .sources import methodology_rubric, pilot_rubric, use_case_rubric


def prepare(
    root,
    pilot="runs/pilot",
    methodology="runs/methodology-evaluation-v1",
    use_cases="runs/use-case-evaluation-v1",
):
    root, pilot, methodology, use_cases = map(Path, [root, pilot, methodology, use_cases])
    if root.exists():
        raise ValueError("Use a new contract directory; frozen artifacts are immutable")
    root.mkdir(parents=True)
    snapshots = root / "sources"
    snapshots.mkdir()
    originals = {
        "pilot_cases.jsonl": pilot / "cases.jsonl",
        "pilot_metadata.json": pilot / "metadata.json",
        "pilot_manifest.json": pilot / "manifest.json",
        "methodology_requests.jsonl": methodology / "requests.jsonl",
        "methodology_gold.jsonl": methodology / "gold.jsonl",
        "methodology_manifest.json": methodology / "manifest.json",
        "use_case_requests.jsonl": use_cases / "requests.jsonl",
        "use_case_gold.jsonl": use_cases / "gold.jsonl",
        "use_case_metadata.json": use_cases / "metadata.json",
        "use_case_manifest.json": use_cases / "manifest.json",
        "use_case_respondents.parquet": use_cases / "respondents.parquet",
    }
    for name, source in originals.items():
        shutil.copy2(source, snapshots / name)
    cases, rubrics = [], []
    metadata = read_json(pilot / "metadata.json")
    for case in read_jsonl(pilot / "cases.jsonl"):
        # Keep the canonical input definition. Existing paraphrased baseline inputs remain a separate snapshot/run.
        cases.append(
            {
                "case_id": case["case_id"],
                "suite": "pilot",
                "split": case["split"],
                "pair_id": case["family_id"],
                "input": {
                    "claim": case["spec"]["canonical_claim"],
                    "context_ref": "sources/pilot_metadata.json",
                },
                "input_provenance": "Existing canonical pilot spec; not a replacement for saved teacher paraphrases.",
            }
        )
        rubrics.append(pilot_rubric(case, metadata))
    mgold = {g["case_id"]: g for g in read_jsonl(methodology / "gold.jsonl")}
    for request in read_jsonl(methodology / "requests.jsonl"):
        g = mgold[request["case_id"]]
        cases.append(
            {**request, "suite": "methodology", "split": g["split"], "pair_id": g["pair_id"]}
        )
        rubrics.append(methodology_rubric(g, request))
    ugold = {g["case_id"]: g for g in read_jsonl(use_cases / "gold.jsonl")}
    for request in read_jsonl(use_cases / "requests.jsonl"):
        g = ugold[request["case_id"]]
        cases.append({**request, "suite": "use_case", "split": g["split"], "pair_id": g["pair_id"]})
        rubrics.append(use_case_rubric(g))
    ids = [c["case_id"] for c in cases]
    assert len(ids) == len(set(ids)) == len(rubrics)
    assert {r["case_id"] for r in rubrics} == set(ids)
    for r in rubrics:
        assert {c["id"] for c in r["criteria"]} == set(DIMENSIONS)
        assert r["required_findings"] and not r["human_calibrated"]
    write_jsonl(root / "cases.jsonl", cases)
    write_jsonl(root / "rubrics.jsonl", rubrics)
    policy = {
        "version": VERSION,
        "dimensions": DIMENSIONS,
        "judge_output_schema": JudgeOutput.model_json_schema(),
        "decisions": ["pass", "fail", "not_applicable"],
        "require_candidate_identity_blinding": True,
        "require_previous_score_blinding": True,
        "judge_context_required": [
            "Original claim/context and case-specific rubric",
            "Relevant full codebook/study metadata",
            "Candidate generated code and declared analysis meaning",
            "Actual execution status, estimates, tests, notes and filter definitions",
            "Candidate final response",
            "Verified reference facts, clearly distinct from candidate results",
        ],
        "answer_context_required": [
            "Original task and relevant metadata",
            "Actual execution status/evidence",
            "Declared variable/group/filter definitions or candidate program",
        ],
        "equivalence_rules": [
            "No exact program, wording, evidence-ID or cosmetic group-name match",
            "No standalone segment_sizes call unless distinct membership counts are requested",
            "Plain-language capability disclosure and valid composed analyses are accepted",
            "Optional alternatives never become required numerical targets",
            "Ambiguous proxy claims allow justified operationalisations or clarification",
        ],
        "numerical_scoring_rules": {
            "counts": "Exact integer membership/outcome base/numerator agreement for requested quantities.",
            "raw_probability_and_statistics_tolerance": {"absolute": 1e-8, "relative": 1e-7},
            "narrative_rounding": "Accept rounding consistent with the displayed precision; do not apply raw machine-value tolerance to rounded prose.",
            "unavailable_fields": "Suppressed/unavailable values stay null; group membership is separately reportable.",
            "group_semantics": "Numbers alone do not verify the population: judge candidate group/filter meaning using code, metadata and actual execution.",
        },
        "remaining_steps": [
            "Implement/verify complete answer and judge context wiring in a new run",
            "Calibrate decisions against independent expert labels",
            "Run stronger-model judging",
            "Reserve fresh untouched cases for confirmatory accuracy",
        ],
        "historical_scores_unchanged": True,
        "human_calibrated": False,
        "development_only": True,
        "candidate_independent_case_requirements": True,
        "rubric_revision_informed_by_previous_failures": True,
    }
    write_json(root / "policy.json", policy)
    write_json(root / "judge_prompt.json", {"instructions": JUDGE_INSTRUCTIONS})
    count = dict(Counter(c["suite"] for c in cases))
    write_json(
        root / "completion.json",
        {
            "step": 1,
            "status": "case_requirements_reviewed_and_frozen",
            "case_count": len(cases),
            "suite_counts": count,
            "all_cases_have_required_optional_prohibited_sections": True,
            "all_cases_have_seven_applicability_aware_criteria": True,
            "human_calibration_complete": False,
            "new_model_evaluation_run": False,
            "original_inputs_references_and_scores_preserved": True,
        },
    )
    lines = [
        "# Step 1: frozen evaluation cases and rubrics",
        "",
        f"Contract version: **{VERSION}**. Prepared {len(cases)} case-specific rubrics: 100 pilot, 30 methodology, 72 use-case.",
        "",
        "## Already implemented before this revision",
        "",
        "| Artifact | Existing state |",
        "|---|---|",
        "| Pilot | 100 canonical cases: 70 train, 10 validation, 20 test; reference programs/evidence and metadata |",
        "| Methodology | 30 controlled scenarios in 15 pairs; original 100 cases audited against 15 root domains |",
        "| Use cases | 72 synthetic development cases in 36 pairs; 20 analytical uses, 10 workflow uses, 6 boundaries |",
        "| Reference validation | All 72 use-case programs checked: 67 valid references and 5 deliberate validation-error examples; 99 independent count checks |",
        "| REST run | 72 generated programs, executed evidence, answers and automated reviews; 216 successful requests |",
        "| Results | 72/72 valid generated execution; strict numeric 29/39, separately audited equivalent numeric 39/39; provisional semantic 59/72 |",
        "| Freezing | Source snapshots, manifests, prompt/configuration and implementation hashes; separate request/reference files |",
        "",
        "## What this revision completes",
        "",
        "- Required findings are specific to the individual request, not inherited blindly from its use-case family or sibling.",
        "- Optional findings and prohibited inferences are explicit. Scope requirements are semantic, not exact wording or function-name checks.",
        "- Seven criteria have applicability rules and pass/fail/not_applicable outputs. Numerical/execution metrics remain separate.",
        "- Required numerical quantities exclude incidental reference group/count records and optional descriptive alternatives.",
        "- Correct composed distributions, rankings and binary relationship comparisons are allowed even when a dedicated helper is absent.",
        "- Explicitly scoped top-box/stratified alternatives are allowed without pretending they are means or adjusted effects.",
        "- Full catalogue facts and actual candidate filters are recognized as evidence. Default group labels cannot erase among= context.",
        "- Supplied hypothetical small-cell facts are distinguished from genuinely suppressed API results.",
        "- Invalid-request examples are distinguished from correct reference handling: the model should explain the invalid value/type rather than reproduce the error.",
        "- Source inputs and legacy results are preserved. The pilot portion uses its existing canonical specs; saved teacher paraphrases and baseline scores are not replaced.",
        "",
        "## Important corrections",
        "",
        "| Previous ambiguity | Frozen rule |",
        "|---|---|",
        "| Invalid transport request judged against top-two satisfaction control | Reject nominal-as-ordinal interpretation; optional top-one or top-two alternative is acceptable |",
        "| Correct conditional result labelled All observed respondents | Read candidate filter/program context; judge the actual conditional population |",
        "| Extra segment_sizes record required in simple comparisons | Estimate bases suffice unless membership counts are separately requested |",
        "| Proposed function absent implies no related calculation is possible | Accept valid compositions and clearly labelled alternatives; never claim absent helper ran |",
        "| Only execution output treated as evidence for discovery | Supplied catalogue names/types/values are valid metadata evidence |",
        "| Weight data assumed missing because API cannot read weights | The file contains unequal weights; the interpreter lacks weights access/interface |",
        "| Every small count treated as suppressed | Apply suppression to API outcomes; supplied scenario facts remain usable |",
        "| Directional claim described as a one-sided test | Preserve the executed two-sided statistical procedure |",
        "| Hidden proxy variable treated as unique truth | Accept justified proxies or clarification when the public concept is ambiguous |",
        "",
        "## Files and integrity",
        "",
        "- cases.jsonl: inputs and split/pair definitions; no judge rubric is included in model-facing input.",
        "- rubrics.jsonl: required/optional/prohibited findings, numerical requirements, seven criteria and reference policy.",
        "- policy.json: evidence context, equivalence, aggregation and blinding rules.",
        "- judge_prompt.json: generic, dataset-independent judge instructions.",
        "- sources/: unchanged copies of source definitions and fixture.",
        "- manifest.json: content hashes and source lineage; validate with validate_bundle().",
        "- completion.json: explicit completion and calibration status.",
        "",
        "## Remaining work belongs to later steps",
        "",
        "The rubric is agent-reviewed and frozen for a new development assessment. It is not independently human-calibrated. No expert labels, stronger-model scores or calibration agreement have been fabricated. Human calibration is step 6; complete context wiring and judge execution are later steps.",
        "",
        "The previous run did not pass full catalogue or candidate filter definitions to its final-answer stage. This revision specifies that requirement but does not silently modify the historical runner. A new runner/run must supply it.",
        "",
        "This revision was informed by observed development failures. It is not a preregistered evaluation of those old responses and cannot convert their scores into untouched test accuracy. The 100 pilot train/validation/test cases and 102 development challenges must be reported separately, never pooled as 202 held-out tests.",
        "",
        "The 36 use-case labels include overlapping analysis patterns; pairs are not 36 independent statistical observations. Root-domain coverage is not exhaustive coverage of all descendant threats.",
        "",
    ]
    atomic_text(root / "README.md", "\n".join(lines))
    # Manifest includes every artifact (including prose) and the generator used to construct it.
    write_json(
        root / "manifest.json",
        {
            "version": VERSION,
            "generator_sha256": file_hash(Path(__file__)),
            "files": {
                str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()
            },
            "source_hashes": {str(p): file_hash(p) for p in originals.values()},
            "case_count": len(cases),
            "suite_counts": count,
            "api_prompt_version": CURRENT_CONTRACT_VERSION,
            "api_contract_digest": digest(contract_manifest(CURRENT_CONTRACT_VERSION)),
            "step1_status": "complete_prepared_and_frozen",
            "human_calibrated": False,
            "model_rerun": False,
            "historical_scores_modified": False,
        },
    )
    return validate_bundle(root)


def validate_bundle(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    for name, fingerprint in manifest["files"].items():
        if file_hash(root / name) != fingerprint:
            raise ValueError("Frozen contract artifact changed: " + name)
    if file_hash(Path(__file__)) != manifest["generator_sha256"]:
        raise ValueError("Contract generator changed; use a new version")
    cases, rubrics = read_jsonl(root / "cases.jsonl"), read_jsonl(root / "rubrics.jsonl")
    if len(cases) != manifest["case_count"] or {c["case_id"] for c in cases} != {
        r["case_id"] for r in rubrics
    }:
        raise ValueError("Case/rubric registry mismatch")
    pairs = {}
    for c in cases:
        pairs.setdefault((c["suite"], c["pair_id"]), set()).add(c["split"])
    if any(len(splits) != 1 for splits in pairs.values()):
        raise ValueError("A pair/family crosses splits")
    return {
        "version": manifest["version"],
        "case_count": len(cases),
        "suite_counts": manifest["suite_counts"],
        "integrity_valid": True,
        "step1_status": manifest["step1_status"],
        "human_calibrated": False,
    }
