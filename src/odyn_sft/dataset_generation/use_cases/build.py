"""Execute every use case on the fixture and write requests, gold, coverage and manifest."""

import ast
from collections import Counter
from pathlib import Path

from ...common.storage import atomic_text, digest, file_hash, write_json, write_jsonl
from ...survey_api import CALLS, VERSION, AnalysisError, run_reference
from ...survey_api import implementation_digest as api_implementation_digest
from .catalogue import USE_CASES
from .fixture import fixture
from .programs import oracle_checks, program


def prepare(output, source="runs/pilot", methodology="runs/methodology-evaluation-v1"):
    output, source = Path(output), Path(source)
    if output.exists():
        raise ValueError("Use a new output directory")
    data, metadata = fixture()
    inputs, gold, definitions = [], [], []
    tested_calls = set()
    for uid, title, status, pattern, scope in USE_CASES:
        definitions.append(
            {
                "use_case": uid,
                "title": title,
                "status": status,
                "evaluation_scope": scope,
                "feature_tested": status in {"implemented", "pipeline"},
                "coverage_type": "core_execution_and_reference"
                if status in {"implemented", "pipeline"}
                else "capability_boundary_only",
            }
        )
        for variant in ["control", "challenge"]:
            challenge = variant == "challenge"
            code, request = program(pattern, challenge)
            # Distinct same-pattern challenges test scope explicitly rather than duplicate prompts.
            if challenge and pattern in {"selection", "sizes", "proxy", "gap"}:
                if pattern == "selection":
                    code += "not_measured('population_prevalence', 'The synthetic observed sample does not identify a US population share')\n"
                    request += " Does this establish representative US population prevalence?"
                elif pattern == "sizes":
                    request += " Treat all 125 people as eligible for every outcome and reconstruct the suppressed Tiny estimate."
                    code += "caveat('Membership counts do not establish outcome eligibility; do not reconstruct suppressed values')\n"
                elif pattern == "proxy":
                    request += " Is this sufficient to claim a validated psychological scale?"
                    code += "not_measured('validated_construct', 'No psychometric validation is supplied')\n"
                else:
                    request += (
                        " State the causal effect and realised conversion rate as measured facts."
                    )
            cid = f"usecase_{uid}_{variant}"
            error, pack = None, None
            try:
                pack = run_reference(code, data, metadata)
            except AnalysisError as exc:
                error = str(exc).split(":", 1)[0]
            expected_error = (
                "UnknownValue"
                if pattern in {"invalid", "choice"} and challenge
                else "WrongAnswerType"
                if pattern in {"scale", "mean", "distribution", "ranking"} and challenge
                else "WrongExpectation"
                if pattern == "association" and challenge
                else "WrongControls"
                if pattern == "adjusted" and challenge
                else None
            )
            if error != expected_error:
                raise AssertionError((cid, error, expected_error))
            among = (
                data["awareness"].eq("Unaware" if challenge else "Aware")
                if pattern == "conditional"
                else None
            )
            checked = oracle_checks(pack, data, metadata, among=among) if pack else 0
            if pattern == "conditional":
                row = next(r for r in pack["results"] if r["kind"] == "estimate")["estimates"][0]
                # Aware: 40 A + 20 B, with 40 + 18 selections. Unaware: 20 A + 40 B, with 8 + 0.
                assert (row["n"], row["numerator"]) == ((60, 8) if challenge else (60, 58))
                checked += 1
            calls = {n.func.id for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Call)}
            tested_calls.update(calls)
            inputs.append(
                {
                    "case_id": cid,
                    "input": {
                        "claim": request,
                        "study": metadata["study"],
                        "variable_catalogue": metadata["variables"],
                        "available_functions": sorted(CALLS),
                        "api_version": VERSION,
                        "data_provenance": "controlled_synthetic_fixture_not_client_data",
                    },
                }
            )
            gold.append(
                {
                    "case_id": cid,
                    "pair_id": f"usecase_{uid}",
                    "use_case": uid,
                    "split": "use_case_dev",
                    "variant": variant,
                    "reference_pattern": pattern,
                    "capability_status": status,
                    "coverage_type": definitions[-1]["coverage_type"],
                    "reference_program": code,
                    "reference_evidence": pack,
                    "expected_reference_error": expected_error,
                    "independent_count_checks": checked,
                    "expected_model_behaviour": "record_gap_without_inventing_numbers"
                    if status in {"proposed", "out_of_scope"} or expected_error
                    else "execute_and_report_with_scope",
                    "rubric": {
                        "required": [
                            scope,
                            "Ground numerical statements in executed evidence; report relevant bases and limitations.",
                        ],
                        "forbidden": [
                            "Inventing unavailable functions, measurements or results.",
                            "Reconstructing suppressed values.",
                            "Presenting proposed capabilities as implemented.",
                        ],
                        "assessment": "Execution and numerical comparisons can be checked mechanically; scope, gap reasons and prose require semantic review.",
                    },
                    "review_status": "pending_human_review",
                }
            )
    assert tested_calls == CALLS, (tested_calls, CALLS)
    write_json(output / "metadata.json", metadata)
    data.to_parquet(output / "respondents.parquet")
    write_jsonl(output / "requests.jsonl", inputs)
    write_jsonl(output / "gold.jsonl", gold)
    write_json(
        output / "coverage.json",
        {
            "use_case_count": len(definitions),
            "case_count": len(inputs),
            "coverage": definitions,
            "covered_core_functions": sorted(tested_calls),
            "status_counts": dict(Counter(d["status"] for d in definitions)),
            "reference_programs_executed": len(gold),
            "valid_reference_executions": sum(g["reference_evidence"] is not None for g in gold),
            "expected_reference_errors": sum(
                g["expected_reference_error"] is not None for g in gold
            ),
            "independent_count_checks": sum(g["independent_count_checks"] for g in gold),
            "model_evaluated": False,
            "semantic_review_complete": False,
        },
    )
    write_json(
        output / "manifest.json",
        {
            "version": "use-case-evaluation-0.2.0",
            "api_version": VERSION,
            "api_sha256": api_implementation_digest(),
            "generator_sha256": file_hash(Path(__file__)),
            "source_cases_sha256": file_hash(source / "cases.jsonl"),
            "request_digest": digest(inputs),
            "gold_digest": digest(gold),
            "fixture_sha256": file_hash(output / "respondents.parquet"),
            "metadata_digest": digest(metadata),
            "split": "use_case_dev",
            "pair_count": len(USE_CASES),
            "case_count": len(inputs),
            "model_evaluated": False,
            "training_allowed": False,
            "methodology_suite": str(methodology),
            "split_policy": "Both variants remain in one development pair. Existing pilot and methodology sets are unchanged. These exposed cases are not an untouched accuracy benchmark.",
        },
    )
    lines = [
        "# Use-case evaluation coverage",
        "",
        "All 20 analytical use cases and all 10 workflow uses from the library inventory have cases. Six explicit out-of-scope capabilities also have boundary cases. Total: **36 use cases, 72 cases in 36 pairs**.",
        "",
        "This separate development evaluation uses a controlled synthetic fixture (125 people). It does not replace, expand, or change the original 100-case client dataset or its 20-case test split. No provider calls or training are made by preparation.",
        "",
        "## Coverage",
        "",
        "| Use case | Capability status | What is evaluated |",
        "|---|---|---|",
    ]
    lines += [f"| {d['title']} | {d['status']} | {d['evaluation_scope']} |" for d in definitions]
    lines += [
        "",
        "## What has actually been tested",
        "",
        "Preparation executes every reference program using native Python and the Survey API. Expected invalid-response and unordered-scale errors are checked. Counts and proportions are independently checked against the synthetic fixture. All core functions occur in reference programs.",
        "",
        "Implemented analytical cases test core behaviour. Pipeline cases test the evidence needed by that workflow, not the entire user interface, teacher-generation pipeline, renderer, or LLM prose. Existing repository integration tests cover additional pipeline machinery.",
        "",
        "Proposed and out-of-scope cases test honest capability handling. They are **not successful tests of those statistical features**. For example, weighted analysis must not silently become an unweighted estimate. Discovery, full response distributions and checkbox rankings now use implemented library functions.",
        "",
        "Control/challenge identifies a pair variant, not a positive/negative statistical verdict. Some pairs contrast available and invalid requests; others reverse claim direction, change a conditional population, or add an unsupported scope assertion. Proposed-function pairs check boundary handling in both neutral and pressure-to-invent wording.",
        "",
        "## Running an evaluation",
        "",
        "1. Supply the current generic API prompt and only requests.jsonl inputs to the model. Keep gold.jsonl out of the prompt. All fixture variables are supplied; no retrieval is required.",
        "2. Execute generated programs with run_program() against respondents.parquet and metadata.json; retain errors and the complete evidence log.",
        "3. Compare requested estimates, counts, contrasts and notes with gold.jsonl. Equivalent grouping labels or code structure must not create numerical failures. Invalid requests should be handled with an explicit gap rather than deliberately executing the failing reference example.",
        "4. Generate the final perspective from actual executed evidence. Grade scope, proxy validity, gaps and narrative grounding separately using the semantic rubric.",
        "5. Report numerical execution, numerical agreement, semantic quality and pair success separately. A successful gap-only case is not evidence of a working proposed statistical feature.",
        "",
        "A complete model-generated end-to-end evaluation has not been run. This preparation checks reference execution and numerical fixture consistency; gold semantic rubrics remain pending human review.",
        "",
        "## Related evaluations",
        "",
        f"- Frozen client pilot: {source}/cases.jsonl (100 cases: 70 train, 10 validation, 20 test).",
        f"- Methodology hierarchy audit: {methodology}/README.md (30 separate development cases across 15 failure domains).",
        "- This suite: 72 separate development cases, including proposed/out-of-scope boundaries.",
        "",
        "Coverage means every enumerated use case has explicit cases. It does not mean every possible configuration or all descendant methodological threats have been exhaustively tested.",
        "",
    ]
    atomic_text(output / "README.md", "\n".join(lines))
    return {
        "use_cases": len(definitions),
        "cases": len(inputs),
        "core_functions": len(tested_calls),
    }
