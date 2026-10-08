"""Contract-first case blueprints and executable API lessons; no model calls."""
from __future__ import annotations

from copy import deepcopy
import ast
import json
from typing import Any

from ...survey_api import CALLS, execute_program
from ...survey_evaluation.meaning import VERSION as CONTRACT_VERSION, filter_value, expected_groups, semantics
from ...common.storage import digest
from ...survey_api.tracing import TracedSurveyAPI

USE_CASES = ("group_difference", "majority_threshold", "recode", "population_and", "population_or", "population_not",
             "variable_defined_groups", "observed_eligibility", "multiple_outcomes", "three_subclaims", "proxy_and_gap", "causal_gap", "population_gap", "transport_gap", "historical_gap", "unavailable_transactions")


def filter_code(spec: dict[str, Any]) -> str:
    """Render a generation contract's population filter as executable API code."""
    op = spec["op"]
    if op == "eq":
        return f"where(var({spec['variable']!r}), {spec['values']!r})"
    if op == "eligible":
        return f"eligible(var({spec['variable']!r}))"
    if op == "not":
        return f"(~{filter_code(spec['arg'])})"
    if op not in {"and", "or"}:
        raise ValueError(f"Unsupported filter operator: {op}")
    operator = "&" if op == "and" else "|"
    return f"({filter_code(spec['args'][0])} {operator} {filter_code(spec['args'][1])})"


def eq(variable: str, values: Any) -> dict[str, Any]:
    return {"op":"eq","variable":variable,"values":values if isinstance(values,list) else [values]}


def contract() -> dict[str, Any]:
    return {"version":CONTRACT_VERSION,"level":"analysis","required_analyses":[],"required_records":[],
            "required_gaps":[],"required_variables":[],"relations":[],"relation_depth":0,
            "available_functions":sorted(CALLS),"criteria":["coverage","measure_population","estimand_direction","gaps_proxy","execution"],
            "code_exact_match_required":False}


def study_blueprints(metadata: dict[str, Any], roles: dict[str, Any], count: int, seed: int) -> list[dict[str, Any]]:
    """Assign each family before reading responses; directions are seed hashes."""
    cards = [c for c in metadata["variables"] if c["section"] != "demographics" and c["answer_type"] != "rank"]
    demographic = [c for c in metadata["variables"] if c["section"] == "demographics"]
    simple = [c for c in cards if c['section']=='service attributes']
    inventory: list[dict[str, Any]] = []
    used: set[str] = set()
    for index in range(count * 10):
        pattern = USE_CASES[index % len(USE_CASES)]
        low_mapping = pattern in {'group_difference','majority_threshold','multiple_outcomes','three_subclaims'} and (index//len(USE_CASES))%2==0
        choices = simple if low_mapping and simple else cards
        main = choices[(index // len(USE_CASES) + seed % len(choices)) % len(choices)]
        gv = demographic[(index // (len(USE_CASES)*len(cards))) % len(demographic)]
        if pattern != "recode":
            gv = demographic[1]  # Simple two-category groups permit genuine low P.
        definitions = roles["demographic_contrasts"][gv["id"]]
        groups = deepcopy(definitions[(index // len(USE_CASES)) % len(definitions)])
        if pattern == "recode":
            gv = demographic[0]
            groups = deepcopy(roles["demographic_contrasts"][gv["id"]][0])
        if pattern == "majority_threshold":
            groups = dict(list(groups.items())[:1])
        analysis = {"component_id":"c1","variable":main["id"],"operation":"top_box_2" if main["answer_type"]=="ordered_scale"
                    else "selection" if main["answer_type"]=="select_all" else "value="+str(main["values"][0]),
                    "group_variable":gv["id"],"groups":groups,"population":None,"group_filters":{},
                    "threshold":.5 if pattern=="majority_threshold" else None,"expect":None,"variable_defined_groups":False}
        population_text = ""
        restriction = eq(demographic[0]["id"], demographic[0]["values"][:2])
        second = eq(demographic[-1]["id"], demographic[-1]["values"][0])
        if pattern in {"population_and", "population_or"}:
            analysis["population"] = {"op":"and" if pattern=="population_and" else "or","args":[restriction,second]}
            population_text = (f" Restrict to people answering {restriction['values']!r} for {demographic[0]['label']} "
                               f"{'and' if pattern=='population_and' else 'or'} {second['values']!r} for {demographic[-1]['label']}.")
        elif pattern == "population_not":
            analysis["population"] = {"op":"not","arg":restriction}
            population_text = f" Restrict to observed {demographic[0]['label']} answers outside {restriction['values']!r}."
        elif pattern == "observed_eligibility":
            other = next(c for c in cards if c["id"] != main["id"] and c["block_id"] != main["block_id"])
            analysis["population"] = {"op":"eligible","variable":other["id"]}
            population_text = f" Restrict to the observed analysis base for {other['label']!r} in {other['block_id']!r}, without assuming true routing."
        elif pattern == "variable_defined_groups":
            other = next(c for c in cards if c["answer_type"]=="single" and c["id"]!=main["id"])
            selected = eq(other["id"],other["values"][0])
            analysis["group_filters"] = {"First response":selected,"Other observed responses":{"op":"not","arg":selected}}
            analysis["variable_defined_groups"] = True
            analysis["groups"] = {"First response":[other["values"][0]],"Other observed responses":other["values"][1:]}
            population_text = f" Compare people answering {other['values'][0]!r} to {other['label']!r} with people giving other observed answers to that question."
        analyses = [analysis]
        if pattern in {"multiple_outcomes","three_subclaims","proxy_and_gap"}:
            total = 3 if pattern=="three_subclaims" else 2
            alternatives = [c for c in (simple if low_mapping and len(simple)>=3 else cards) if c["block_id"] != main["block_id"] and c["id"] != main["id"]]
            for offset, other in enumerate(alternatives[:total-1],2):
                extra = deepcopy(analysis)
                extra.update(component_id=f"c{offset}", variable=other["id"], operation="top_box_2" if other["answer_type"]=="ordered_scale"
                             else "selection" if other["answer_type"]=="select_all" else "value="+str(other["values"][0]))
                if pattern=="multiple_outcomes":
                    extra["population"] = restriction
                analyses.append(extra)
        if pattern == "proxy_and_gap":
            intent = next(c for c in cards if c["section"] == "purchase intent")
            analyses[1].update(variable=intent["id"], operation="value="+str(intent["values"][0]))
        key = digest({"study":metadata["study_id"],"analyses":analyses,"pattern":pattern})
        if key in used:
            continue
        used.add(key)
        for a in analyses:
            a["expect"] = ">" if int(digest({"family":key,"component":a["component_id"],"seed":seed})[:8],16)%2 else "<"
        # Alternate exact-label phrasing and controlled natural realisations.
        natural = not low_mapping and int(digest({"family":key,"style":seed})[:8],16) % 3 != 0
        texts = []
        by_id = {c["id"]:c for c in metadata["variables"]}
        for a in analyses:
            card = by_id[a["variable"]]
            phrase = roles["phrases"].get(a["variable"], f"report selecting {card['label']!r}") if natural else f"give the recorded response to {card['label']!r} in {card['block_id']!r}"
            if card["answer_type"]=="single":
                phrase = f"answer {a['operation'][6:]!r} when asked about {phrase.removeprefix('report ')}"
            if not natural and card["answer_type"]=="ordered_scale":
                phrase = f"give one of the two highest agreement responses to {card['label']!r}"
            labels = list(a["groups"])
            sentence = (f"{'More' if a['expect']=='>' else 'Fewer'} than half of {labels[0]} respondents {phrase}." if a["threshold"] is not None else
                        f"{labels[0]} respondents {phrase} {'more often' if a['expect']=='>' else 'less often'} than {labels[1]} respondents.")
            if a.get("population") and pattern=="multiple_outcomes":
                sentence += f" For this component only, restrict to {demographic[0]['label']} responses {restriction['values']!r}."
            texts.append(sentence)
        gaps = []
        proxy = None
        if pattern=="causal_gap":
            texts.append("The difference is caused by membership in those groups.")
            gaps.append({"concept":"causal effect","reason":"Cross-sectional self-report and observed comparisons do not identify a causal effect of group membership."})
        if pattern=="proxy_and_gap":
            intent = next(c for c in cards if c["section"]=="purchase intent")
            texts.append(f"Use {intent['label']!r} as a limited proxy for actual purchasing; explain what the study cannot establish about transactions or causality.")
            proxy = {"concept":"actual purchasing","variables":[intent["id"]],"rationale":"Stated future purchase intention is an indirect measure, not observed transactions."}
            gaps.extend([{"concept":"observed transactions","reason":"No linked transaction records are supplied."},
                         {"concept":"causal effect","reason":"No identifying causal design is supplied."}])
        extra_gaps = {
            "population_gap":("population prevalence", "The same percentages describe every resident in the target population.", "Selection probabilities and nonresponse adjustment are not supplied; unweighted sample shares do not identify population prevalence."),
            "transport_gap":("transportability", "The same pattern applies to residents in a different country.", "No observations or transport model for the proposed destination are supplied."),
            "historical_gap":("temporal comparison", "This pattern has strengthened since the previous year.", "Only one cross-section is supplied; no historical measurements or harmonisation evidence identify change."),
            "unavailable_transactions":("observed transactions", "The study's linked transaction records establish actual purchase rates.", "No linked transaction records are supplied.")}
        if pattern in extra_gaps:
            concept,sentence,reason = extra_gaps[pattern]
            if pattern == "unavailable_transactions":
                analyses, texts, population_text = [], [], ""
            texts.append(sentence)
            gaps.append({"concept":concept,"reason":reason})
        if pattern == 'unavailable_transactions':
            key = digest({'study':metadata['study_id'],'gap':'observed transactions'})
            if key in used:
                continue
            used.add(key)
        inventory.append({"family_id":key,"pattern":pattern,"analyses":analyses,"claim":" ".join(texts)+population_text,
                          "realization_components":texts,"realization_population":population_text,
                          "natural":natural,"gaps":gaps,"proxy":proxy,"seed":seed,
                          "direction_commitment":digest({"family":key,"directions":[a['expect'] for a in analyses]})})
        if len(inventory)==count:
            break
    if len(inventory) != count:
        raise ValueError(f"Only {len(inventory)} distinct case families available for {count} requested")
    return inventory


def program_for(blueprint: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gold = contract()
    gold["required_analyses"] = deepcopy(blueprint["analyses"])
    code, variables = [], set()
    for index,a in enumerate(blueprint["analyses"],1):
        if a.get("group_filters"):
            mapping = ", ".join(f"{label!r}: {filter_code(f)}" for label,f in a["group_filters"].items())
            code.append(f"g{index}=group({{{mapping}}})")
        else:
            code.append(f"g{index}=group(var({a['group_variable']!r}), {a['groups']!r})")
        with_scope = f", among={filter_code(a['population'])}" if a.get("population") else ""
        operation = a["operation"]
        call = f"top_box(var({a['variable']!r}), k=2, by=g{index}{with_scope})" if operation=="top_box_2" else (
            f"proportion(var({a['variable']!r}), by=g{index}{with_scope})" if operation=="selection" else
            f"proportion(var({a['variable']!r}), value={operation[6:]!r}, by=g{index}{with_scope})")
        labels = list(a["groups"])
        comparator = repr(a["threshold"]) if a["threshold"] is not None else repr(labels[1])
        code.extend([f"with subclaim({a['component_id']!r}, 'Recorded evidence for component {index}'):",
                     f"    e{index}={call}",f"    compare(e{index}, {labels[0]!r}, {comparator}, expect={a['expect']!r})"])
        variables.add(a["variable"])
    if blueprint.get("proxy"):
        p = blueprint["proxy"]
        code.append(f"proxy({p['concept']!r}, {[v for v in p['variables']]!r}, {p['rationale']!r})")
        variables.update(p["variables"])
        gold["required_records"].append({"component_id":"proxy","kind":"note","fields":{"label":"proxy_variable","variables":p["variables"]}})
    for i,gap in enumerate(blueprint["gaps"]):
        code.extend([f"with subclaim('gap{i}', {gap['concept']!r}):",f"    not_measured({gap['concept']!r}, {gap['reason']!r})"])
        gold["required_gaps"].append(gap)
        gold["required_records"].append({"component_id":f"gap{i}","kind":"note","fields":{"label":"not_measured","concept":gap["concept"]}})
    code.append("caveat('Observed unweighted self-report; preserve the actual bases and avoid population or causal overclaims.')")
    gold["required_variables"] = sorted(variables)
    return "\n".join(code)+"\n", gold


def lesson(operation: str, ids: dict[str, str], seed: int) -> dict[str, Any]:
    """One minimal teaching task per callable API operation, including helpers."""
    seg, selected, other, scale = (ids[k] for k in ("segment","selected","other","scale"))
    bind = f"s=var({selected!r})\nx=var({scale!r})\nr=var({seg!r})\n"
    group = "g=group(r, {'North':['North'],'South':['South']})\n"
    examples = {
        "var": ("Resolve the selected-service variable and estimate its recorded share.","proportion(s)",[selected,other]),
        "find_variables": ("Find the selected-service measure by its catalogue label, then estimate its share.","matches=find_variables('Selected service',k=1)\nproportion(var(matches[0]))",[selected,other]),
        "describe": ("Inspect the selected-service metadata and its observed base, then estimate the share.","import json\ncard=json.loads(describe(s))\nassert card['answer_type']=='select_all'\nproportion(s)",[selected,other]),
        "values": ("Inspect allowed region responses and estimate the recorded North share.","codes=values(r)\nproportion(r,codes[0])",[seg]),
        "eligible": ("Estimate agreement among the observed checkbox block base; do not infer true routing.","proportion(x,'Agree',among=eligible(s))",[selected,other,scale]),
        "where": ("Estimate agreement among people reporting North.","proportion(x,'Agree',among=where(r,'North'))",[seg,scale]),
        "group": ("Compare service-selection rates between North and South.",group+"e=proportion(s,by=g)\ncompare(e,'North','South',expect='<')",[seg,selected,other]),
        "subclaim": ("Record service selection and agreement under separate proposition identifiers.","with subclaim('selection','Service selection'):\n    proportion(s)\nwith subclaim('agreement','Agreement'):\n    proportion(x,'Agree')",[selected,other,scale]),
        "proportion": ("Estimate the share selecting the service with its observed base and uncertainty.","proportion(s)",[selected,other]),
        "distribution": ("Report the complete region distribution.","distribution(r)",[seg]),
        "rank_options": ("Rank the two services by observed selection share without claiming a significant difference.","rank_options('Which services do you use?')",[selected,other]),
        "top_box": ("Estimate the proportion in the highest agreement category.","top_box(x,k=1)",[scale]),
        "mean_score": ("Estimate mean quality-agreement position, stating the equal-spacing assumption.","mean_score(x)",[scale]),
        "compare": ("Is selection in North lower than selection in South? Report a difference and uncertainty.",group+"e=proportion(s,by=g)\ncompare(e,'North','South',expect='<')",[seg,selected,other]),
        "trend": ("Does service selection increase across the declared North, Central, South order? This is a declared grouping order, not geographic distance.","g=group(r)\ne=proportion(s,by=g)\ntrend(e,['North','Central','South'],expect='increasing')",[seg,selected,other]),
        "association": ("Is agreement positively associated with selecting the service? Do not claim causation.","association(x,s,expect='positive')",[scale,selected,other]),
        "adjusted_compare": ("Compare North and South selection after stratifying by agreement, without treating adjustment as causal identification.",group+"e=proportion(s,by=g)\nadjusted_compare(e,'North','South',controls=x,expect='<')",[seg,selected,other,scale]),
        "not_measured": ("Does recorded selection establish actual transactions? Describe selection and explain the gap.","proportion(s)\nnot_measured('actual transactions','No linked purchase records are supplied.')",[selected,other]),
        "proxy": ("Use reported agreement as a narrow proxy for perceived quality, while distinguishing it from objective performance.","proxy('perceived quality',[x],'A single agreement response is a narrow self-report measure, not objective service performance.')\nproportion(x,'Agree')",[scale]),
        "caveat": ("Describe service selection and state its observational scope.","proportion(s)\ncaveat('Unweighted observed self-report; no representative population or causal inference.')",[selected,other]),
        "segment_sizes": ("Report North and South membership counts before any outcome-base restriction.",group+"segment_sizes(g)",[seg]),
    }
    claim, body, needed = examples[operation]
    # Bind only names needed in this minimal catalogue; undefined unused names
    # are not allowed even if the body would never reference them.
    code = "\n".join(line for line in bind.splitlines() if any(repr(v) in line for v in needed)) + "\n" + body + "\n"
    if "subclaim(" not in body:
        lines = code.splitlines()
        code = "with subclaim('lesson', 'Minimal API example'):\n"+"\n".join("    "+line for line in lines)+"\n"
    return {"claim":claim,"program":code,"variables":needed,"pattern":operation,"family_id":digest({"operation":operation,"ids":ids,"seed":seed}),
            "seed":seed,"natural":False,"gaps":[],"direction_commitment":None}


def run_case(blueprint: dict[str, Any], data: Any, metadata: dict[str, Any], *, tutorial: bool = False) -> dict[str, Any]:
    code, gold = (blueprint["program"], contract()) if tutorial else program_for(blueprint)
    api = TracedSurveyAPI(data, metadata)
    pack = execute_program(code, api)
    pack["analysis_trace"] = api.trace
    if tutorial:
        gold["required_variables"] = blueprint["variables"]
        # Actual result shapes are retained in private gold for mechanical checks.
        for row in pack["results"]:
            if row["kind"] == "subclaim":
                continue
            fields = {k:row[k] for k in ("variable","operation","contrast_type","variables","controls","block_id","segments","label","concept","measure","expect","order") if k in row}
            target = {"component_id":row["id"],"kind":row["kind"],"fields":fields}
            if row['kind']=='estimate' and row['id'] in api.trace:
                target['expected_trace'] = api.trace[row['id']]
            gold["required_records"].append(target)
    checks = semantics(gold, pack, data, metadata)
    if not checks["passed"]:
        raise ValueError("Reference does not satisfy its population/estimand contract: "+str(checks))
    # For measured claims, independently recompute counts with Pandas masks.
    independent_checks = 0
    for a in gold["required_analyses"]:
        from .oracle import counts
        record = next(r for r in pack["results"] if r["kind"]=="estimate" and r["subclaim"]==a["component_id"])
        for row,(n,k) in zip(record["estimates"],counts(a,data,metadata),strict=True):
            if row["n"] != n or row["numerator"] != (k if n>=10 else None):
                raise ValueError("Independent respondent-count check failed")
            independent_checks += 1
    gold.update(reference_program=code,reference_evidence=pack)
    return {**blueprint,"program":code,"evidence":pack,"analysis_contract":gold,
            "synthesis_contract":{"level":"synthesis","criteria":["coverage","grounding","inference","scope_stress","perspective"],
                "required_findings":pack["results"],"required_gaps":gold["required_gaps"],
                "prohibited":["Whole-claim truth verdict","Invented calculations or studies","Unidentified causal or population generalisations"],
                "candidate_conditioning":"Ground numbers in candidate execution; never substitute private gold for missing candidate evidence."},
            "reference_perspective":perspective(pack),"independent_count_checks":independent_checks,
            "reference_semantics_passed":True,"reference_language_status":"controlled_template_pending_expert_review"}


def perspective(pack: dict[str, Any]) -> str:
    """Evidence-first synthesis; no hidden DGP truth or unexecuted statistics."""
    lines = []
    for row in pack["results"]:
        ref = f"[{row['id']}]"
        if row["kind"] == "estimate":
            for r in row["estimates"]:
                if r["value"] is None:
                    lines.append(f"{ref} {row['variable_label']}, {r['group']}: observed base n={r['n']}; outcome values are suppressed.")
                elif row["operation"] == "mean_score":
                    lines.append(f"{ref} {row['variable_label']}, {r['group']}: mean position {r['value']:.3f}, n={r['n']}, 95% interval {r['ci']}; assumes equally spaced categories.")
                else:
                    lines.append(f"{ref} {row['variable_label']} ({row['operation']}), {r['group']}: {r['numerator']}/{r['n']} ({100*r['value']:.2f}%), 95% interval {r['ci']}.")
        elif row["kind"] == "contrast":
            direction = {"consistent":"supports the declared direction", "inconsistent":"contradicts the declared direction",
                         "no_clear_difference":"does not establish a clear difference", "not_tested":"has no declared expectation",
                         "unavailable":"cannot support the requested inference"}.get(row["label"],row["label"])
            lines.append(f"{ref} {row.get('contrast_type','comparison')} {direction}; estimate={row['value']}, 95% interval={row['ci']}, two-sided p={row['p_value']}, Holm p={row.get('p_holm')}. These labels use unadjusted uncertainty.")
        elif row["kind"] == "note":
            lines.append(f"{ref} {row.get('concept','Scope')}: {row['reason']}")
        elif row["kind"] == "segment_sizes":
            lines.append(f"{ref} Segment membership counts: {row['segments']}; these precede outcome eligibility.")
        elif row["kind"] == "ranking":
            lines.append(f"{ref} Descriptive option ranks: "+str({label:[(r['variable_label'],r['rank']) for r in rows] for label,rows in row['rankings'].items()})+". Ranking alone does not establish significant differences.")
    lines.append("These findings describe the unweighted observed study and its recorded measures. They do not establish a whole-claim verdict, representative population prevalence, equivalence or a causal explanation.")
    return "\n".join(lines)
