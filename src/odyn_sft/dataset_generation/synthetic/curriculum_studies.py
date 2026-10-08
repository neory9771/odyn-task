"""Small API teaching studies and scalable studies with readable construct decoys.

Entities and source IDs vary by study. Repeated blocks have explicit reference
periods, so difficult mapping remains answerable rather than arbitrarily ambiguous.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any
import numpy as np
import pandas as pd

from ...common.storage import digest
from .spec import realized_stressors, validate_spec
from .templates import template_spec

DOMAINS = ("brand_tracking", "transit_service", "streaming_services", "mobile_services")
ENTITIES = ("Alder", "Birch", "Cedar", "Elm", "Maple", "Oak", "Pine", "Willow")


def scaled_spec(study_id: str, seed: int, domain: str, width: int,
                population_size: int = 20_000) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build declared measurement items, not extra random noise columns."""
    base_domain = "transit_service" if domain == "transit_service" else "brand_tracking"
    spec = template_spec(base_domain, study_id, seed, population_size)
    spec["domain"] = domain
    spec["label"] = f"Synthetic {domain.replace('_', ' ')} research"
    rng = np.random.default_rng(seed)
    names = list(rng.choice(ENTITIES, size=4, replace=False))
    replacements = {f"Brand {b}": name for b, name in zip("ABCD", names)}
    replacements.update({"packaged snacks": "streaming subscriptions", "snack": "streaming", "brands": "providers", "brand": "provider"}
                        if domain == "streaming_services" else
                        {"packaged snacks": "mobile subscriptions", "snack": "mobile", "brands": "networks", "brand": "network"}
                        if domain == "mobile_services" else {})
    def text(value: str) -> str:
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    # Qualify checkbox labels with measurement context; entity names alone do
    # not distinguish awareness, consideration and use.
    phrases: dict[str, str] = {}
    for block in spec["questionnaire"]["blocks"]:
        block["label"] = text(block["label"])
        for item in block["items"]:
            item["label"] = text(item["label"])
            if "values" in item:
                item["values"] = [text(v) for v in item["values"]]
            original = item["id"]
            label = item["label"]
            if original.startswith("aware_brand_"):
                phrases[original] = f"say they have heard of {label}"
                item["label"] = f"Recognition of {label}"
            elif original.startswith("consider_brand_"):
                phrases[original] = f"would consider buying from {label}"
                item["label"] = f"Purchase consideration for {label}"
            elif original.startswith("ad_seen_"):
                synonyms = {"Television": "TV", "Social media": "social platforms", "Outdoor posters": "outdoor advertising"}
                phrases[original] = f"recall advertising through {synonyms.get(label, label)}"
                item["label"] = f"Advertising recall via {label}"
            elif original.startswith("mode_"):
                phrases[original] = f"report travelling by {label.lower()}"
                item["label"] = f"Recent use of {label}"
            elif original.startswith("problem_"):
                synonyms = {"Delays": "late-running services", "Overcrowding": "crowded vehicles", "Missed connections": "failed connections", "Unclear information": "confusing travel information"}
                phrases[original] = f"report experiencing {synonyms.get(label, label.lower())}"
                item["label"] = f"Reported problem: {label}"
            elif original.startswith("heard_"):
                phrases[original] = f"are familiar with {label}"
                item["label"] = f"Recognition of {label}"
            elif block["answer_type"] == "ordered_scale":
                phrases[original] = {"att_good_value": f"agree that {names[0]} offers value for its price",
                    "att_memorable_ads": f"agree that advertisements for {names[0]} stay in their memory",
                    "att_reliable": "agree that public transport can be depended on",
                    "att_fares": "agree that ticket prices offer value",
                    "att_safe": "agree that travelling feels secure"}.get(original, f"agree with {label.lower()}")
            else:
                phrases[original] = {"category_user":"recent purchases in the category covered by the study", "next_brand":"the provider intended for their next purchase", "uses_transit":"whether they have travelled on public transport recently", "ticket_plan":"the fare product intended for the coming month"}.get(original, f"report {label.lower()}")
    base_blocks = deepcopy(spec["questionnaire"]["blocks"])
    # Duplicate option label sets are distinguishable by disclosed period.
    clone_sources = [b for b in base_blocks if b["answer_type"] == "select_all"]
    count = len(spec["demographics"]) + sum(len(b["items"]) for b in base_blocks)
    cycle = 0
    while count < width:
        source = clone_sources[cycle % len(clone_sources)]
        clone = deepcopy(source)
        clone["id"] = f"repeat_{cycle:04d}"
        period = 7 * (cycle + 2)
        clone["label"] = source["label"] + f" Reference window: previous {period} days."
        clone["items"] = clone["items"][:width-count]
        for item in clone["items"]:
            original = item["id"]
            item["id"] = f"repeat_{cycle:04d}_{original}"
            item["intercept"] += float(rng.uniform(-.3, .3))
            phrases[item["id"]] = phrases[original] + f" during the previous {period} days"
        spec["questionnaire"]["blocks"].append(clone)
        count += len(clone["items"])
        cycle += 1
    # Three distinct, unrouted scales provide genuine low-mapping/high-program
    # tasks without removing their complete question wording from the catalogue.
    prototype = next(i for b in base_blocks if b['answer_type']=='ordered_scale' for i in b['items'])
    for j,(label,statement) in enumerate((('Clarity','Information is easy to understand'),
                                        ('Timeliness','Responses arrive promptly'),
                                        ('Accessibility','Assistance is easy to obtain'))):
        idx = next(i for i in range(len(spec['questionnaire']['blocks'])-1,-1,-1)
                   if spec['questionnaire']['blocks'][i]['answer_type']=='select_all')
        spec['questionnaire']['blocks'][idx]['items'].pop()
        if not spec['questionnaire']['blocks'][idx]['items']:
            spec['questionnaire']['blocks'].pop(idx)
        item = deepcopy(prototype)
        item.update(id=f'attribute_{j}',label=label)
        spec['questionnaire']['blocks'].append({'id':f'attribute_block_{j}',
            'label':f'How strongly do you agree: {statement}?', 'section':'service attributes',
            'answer_type':'ordered_scale','scale':deepcopy(next(b['scale'] for b in base_blocks if b['answer_type']=='ordered_scale')),'items':[item]})
        phrases[item['id']] = f'agree that {statement.lower()}'
    if width >= 256:
        decoys = [b for b in spec["questionnaire"]["blocks"] if b["id"].startswith("repeat_")][-16:]
        for j,block in enumerate(decoys):
            for k,item in enumerate(block["items"]):
                label = f"Community service {j//2+1}, option {k+1}"
                item["label"] = label
                period = block["label"].split("Reference window:")[-1].strip()
                phrases[item["id"]] = f"report selecting {label} in the question covering {period}"
    for block in spec["questionnaire"]["blocks"]:
        if not block["id"].startswith("repeat_") and block["answer_type"] == "select_all":
            block["label"] += " Reference window: current survey period."
            for item in block["items"]:
                phrases[item["id"]] += " in the current survey period"
    # Large catalogues include rank items as contextual decoys and explicit
    # unsupported-rank lessons, not as ordered scales with invented direction.
    if width >= 256:
        for i in range(6):
            index = next(j for j in range(len(spec["questionnaire"]["blocks"])-1, -1, -1)
                         if spec["questionnaire"]["blocks"][j]["answer_type"] == "select_all")
            removed = spec["questionnaire"]["blocks"][index]["items"].pop()
            if not spec["questionnaire"]["blocks"][index]["items"]:
                spec["questionnaire"]["blocks"].pop(index)
            spec["questionnaire"]["blocks"].append({"id": f"ranking_{i}", "label": f"Rank preferred service option {i+1}",
                "section": "preferences", "answer_type": "rank", "items": [{"id": f"ranking_item_{i}",
                "label": f"Service preference rank {i+1}", "construct": removed["construct"],
                "values": [str(k) for k in range(1, 7)], "value_intercepts": [0.] * 6,
                "value_loadings": [.1, .05, 0., -.05, -.1, -.15], "refusal": False}]})
    surviving_blocks = {b['id'] for b in spec['questionnaire']['blocks']}
    for key in ('media_outcome_blocks','purchase_outcome_blocks'):
        spec['scenario_bindings'][key] = [b for b in spec['scenario_bindings'].get(key,[]) if b in surviving_blocks]
    identifiers = [d["id"] for d in spec["demographics"]] + [i["id"] for b in spec["questionnaire"]["blocks"] for i in b["items"]]
    opaque = {v: f"q{index:04d}_{digest({'study':study_id,'variable':v})[:6]}" for index, v in enumerate(identifiers, 1)}
    # Rename every observed-variable reference, preserving hidden latent names.
    for d in spec["demographics"]:
        d["id"] = opaque[d["id"]]
    for latent in spec["latent_variables"]:
        latent["effects"] = {opaque[k]: v for k, v in latent.get("effects", {}).items()}
    for block in spec["questionnaire"]["blocks"]:
        if block.get("routing"):
            block["routing"]["variable"] = opaque.get(block["routing"]["variable"],block["routing"]["variable"])
        for item in block["items"]:
            item["id"] = opaque[item["id"]]
    spec["response_process"]["effects"] = {opaque[k]:v for k,v in spec["response_process"]["effects"].items()}
    if "strata_variable" in spec["sampling_design"]:
        spec["sampling_design"]["strata_variable"] = opaque[spec["sampling_design"]["strata_variable"]]
    for proxy in spec["measurement_model"]["proxies"]:
        proxy["item"] = opaque[proxy["item"]]
    for k in ("causal_group_variables", "media_group_variables"):
        spec["scenario_bindings"][k] = [opaque[v] for v in spec["scenario_bindings"].get(k, [])]
    spec["intended_stressors"] = sorted(realized_stressors(spec))
    # Routing disclosure resolves labels from the entire questionnaire, including
    # earlier single-choice item IDs that differ from their block IDs.
    roles = {"identifiers": opaque, "phrases": {opaque[v]: phrase for v, phrase in phrases.items() if v in opaque},
             "demographic_contrasts": {opaque[d_old]: next(d for d in spec["demographics"] if d["id"]==opaque[d_old])["contrasts"]
                                       for d_old in identifiers[:len(spec["demographics"])]}}
    return validate_spec(spec), roles


def minimal_study(study_id: str, seed: int) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    """Four-item balanced teaching survey with real uncertainty and missingness."""
    rng = np.random.default_rng(seed)
    n = 240
    ids = {name: f"q{index}_{digest({'study':study_id,'name':name})[:6]}"
           for index, name in enumerate(("segment", "selected", "other", "scale"), 1)}
    segment = np.array(["North", "Central", "South"])[np.arange(n) % 3]
    probabilities = np.array([.25, .5, .75])[np.arange(n) % 3]
    selected = rng.random(n) < probabilities
    scale = np.array(["Disagree", "Neutral", "Agree"])[np.minimum(2, (rng.random(n)+probabilities)*2).astype(int)]
    data = pd.DataFrame({"respondent_id": [f"{study_id}_r{i}" for i in range(n)], ids["segment"]: segment,
        ids["selected"]: pd.array(selected, dtype="boolean"), ids["other"]: pd.array(~selected, dtype="boolean"),
        ids["scale"]: scale})
    # One entire checkbox block is unobserved; explicit False elsewhere is not NA.
    data.loc[230:, [ids["selected"], ids["other"]]] = pd.NA
    cards = []
    for name, label, kind, values, ordered, block in (
        ("segment", "Region", "single", ["North", "Central", "South"], None, "Region"),
        ("selected", "Selected service", "select_all", [True, False], None, "Which services do you use?"),
        ("other", "Alternative service", "select_all", [True, False], None, "Which services do you use?"),
        ("scale", "Agreement with service quality", "ordered_scale", ["Disagree", "Neutral", "Agree"], ["Disagree", "Neutral", "Agree"], "How do you rate service quality?")):
        cards.append({"id":ids[name],"label":label,"block_id":block,"section":"demographics" if name=="segment" else "service",
                      "answer_type":kind,"values":values,"ordered_values":ordered,"flags":["unweighted"],
                      "denominator_rule":"observed_block_respondents" if kind=="select_all" else "answered_item_excluding_refusals"})
    metadata = {"study_id":study_id,"variables":cards,"study":{"label":"Small synthetic service survey","n":n,
        "period":"synthetic cross-section","weights":None,"scope":"Synthetic unweighted self-report; no causal or population identification.",
        "missingness_note":"False means not selected in an observed checkbox block; all-blank blocks are excluded."}}
    metadata["hash"] = digest(metadata)
    return data, metadata, ids


def subset_study(data: pd.DataFrame, metadata: dict[str, Any], variables: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Include all checkbox siblings needed to preserve the observed denominator."""
    needed = set(variables)
    blocks = {c["block_id"] for c in metadata["variables"] if c["id"] in needed and c["answer_type"] == "select_all"}
    needed.update(c["id"] for c in metadata["variables"] if c["block_id"] in blocks)
    result = deepcopy(metadata)
    result["variables"] = [c for c in result["variables"] if c["id"] in needed]
    result["hash"] = digest({k:v for k,v in result.items() if k!="hash"})
    return data[["respondent_id"]+[c["id"] for c in result["variables"]]].copy(), result
