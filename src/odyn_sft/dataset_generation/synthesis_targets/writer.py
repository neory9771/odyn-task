"""Write SFT2 training targets with a strong model, accepted only through the evaluation gate.

The target writer works from code-built facts and the expectation checklist, never
from the template answer. A target is accepted only if the deterministic checks
(D1-D4) and every judge expectation pass. A rejected target is rewritten with the
failed checks as feedback, up to MAX_ATTEMPTS.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict

from ...survey_evaluation import synthesis_judge as sj
from ...survey_evaluation.synthesis_checks import check_answer
from ..types import DataObject

VERSION = "synthesis-target-writer-1.0.1"
MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "medium"
MAX_OUTPUT_TOKENS = 8000
TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 3

INSTRUCTION = '''You write the answer a research assistant gives a client who asks whether their survey data support a claim.
Give a perspective, not a verdict: present the supporting, contradicting and inconclusive evidence so the client
can judge for themselves. Never say the claim, or any part of it, is true, false, proven, confirmed or refuted.

Use only the FACTS. Cite each finding with its evidence ID in square brackets, e.g. [c1.r3].
Structure:
- Go through the claim parts in order. For each part, give the supporting, then contradicting, then inconclusive
  findings, each with its ID. Say plainly what each finding means for that part.
- "inconclusive" means the data show no clear difference: say so, and do not present it as support or contradiction.
- State every gap (not_measured) and why it cannot be assessed; describe proxies as indirect, with their limits.
- If a finding is marked robust_after_adjustment=false, say it does not hold after multiple-testing adjustment.
- Describe results as the observed, unweighted survey sample, not the whole population. Do not claim causation.
- Never give a value for a suppressed group; you may say its sample was too small to report.
- End with the main limitation(s) in at most two sentences.
Numbers: only those in the FACTS. Percentages and percentage-point differences to 1 decimal place
(0.4681 -> 46.8%); intervals as "95% interval 44.4% to 49.3%"; sample sizes as given. No p-values.
Write plain prose for a business reader: no code, no JSON, no API terms such as "top_box_2" or "contrast".
Write as the analyst: never mention "facts", "supplied findings", the checklist or reviewers.
Length: 120 to 350 words; up to 450 when the claim has more than two parts.
The CHECKLIST lists what reviewers will verify; meet every item.'''


class WrittenAnswer(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    answer: str


def writer_facts(facts: DataObject) -> DataObject:
    """Facts with the numbers the answer may use; no raw estimator output beyond that."""
    finding_keys = ("evidence_id", "direction", "a", "b", "value", "value_scale", "ci", "robust_after_adjustment",
                    "estimate_id", "estimate_rows")
    return {"claim": facts["claim"],
            "study": {k: facts["study"].get(k) for k in ("label", "n", "period", "weights", "scope")},
            "parts": [{"part": p["text"],
                       "findings": [{k: f.get(k) for k in finding_keys} for f in p["findings"]],
                       "descriptive": p["descriptive"],
                       "gaps": p["gaps"],
                       "suppressed_groups": [{"group": s["group"], "n": s["n"]} for s in p["suppressed"]]}
                      for p in facts["parts"]]}


def payload(facts: DataObject, items: list[DataObject], feedback: list[str]) -> str:
    body: DataObject = {"FACTS": writer_facts(facts),
                        "CHECKLIST": [i["statement"] for i in items]}
    if feedback:
        body["PREVIOUS_ATTEMPT_FAILED"] = feedback
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def feedback_from(checks: DataObject, judged: DataObject | None, items: list[DataObject]) -> list[str]:
    """Concrete reasons the previous attempt was rejected."""
    notes = []
    c = checks["checks"]
    if c["D1_citations_valid"]["invalid_ids"]:
        notes.append(f"Cited IDs not in the facts: {c['D1_citations_valid']['invalid_ids']}")
    for u in c["D2_numbers_grounded"]["ungrounded"][:5]:
        notes.append(f"Number {u['number']!r} does not match the facts (sentence: {u['sentence'][:160]!r})")
    for h in c["D3_no_suppressed_values"]["hits"][:3]:
        notes.append(f"Gave a value for suppressed group {h['group']!r}")
    if c["D4_required_ids_cited"]["missing"]:
        notes.append(f"Findings not cited: {c['D4_required_ids_cited']['missing']}")
    if judged and judged["valid"]:
        statements = {i["id"]: i["statement"] for i in items}
        for item_id, r in judged["results"].items():
            if not r["passed"]:
                notes.append(f"Not met: {statements[item_id]} Reviewer: {r['explanation'][:200]}")
    return notes


def generate_target(writer: Any, judge: Any, record_id: str, facts: DataObject, items: list[DataObject],
                    evidence: DataObject) -> DataObject:
    """Up to MAX_ATTEMPTS writer calls, each gated by D-checks and the judge; returns the attempt log."""
    attempts: list[DataObject] = []
    feedback: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        row: DataObject = {"attempt": attempt}
        try:
            answer = writer.complete(INSTRUCTION, payload(facts, items, feedback), WrittenAnswer,
                                      f"{record_id}:write:a{attempt}").answer
        except Exception as exc:  # Ledger marks the request failed; the next attempt uses a new ID.
            row["error"] = f"writer {type(exc).__name__}: {exc}"[:300]
            attempts.append(row)
            continue
        checks = check_answer(answer, evidence, facts)
        row.update(answer=answer, words=len(answer.split()), checks=checks)
        judged = None
        if checks["passed"]:  # Judge only answers that pass the exact checks.
            try:
                judged = sj.judge(judge, facts, items, answer, f"{record_id}:gate:a{attempt}")
            except Exception as exc:
                row["error"] = f"judge {type(exc).__name__}: {exc}"[:300]
        row["judged"] = judged
        row["accepted"] = bool(checks["passed"] and judged and judged["valid"]
                               and all(r["passed"] for r in judged["results"].values()))
        attempts.append(row)
        if row["accepted"]:
            return {"accepted": True, "answer": answer, "attempts": attempts}
        feedback = feedback_from(checks, judged, items)
    return {"accepted": False, "answer": None, "attempts": attempts}
