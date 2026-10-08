"""Tabular variable metadata with shared definitions and no variable retrieval."""

import ast
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

ANALYSIS_FIELDS = ["id", "label", "block_id", "section", "answer_type", "values", "ordered_values"]
BLOCK_CATALOGUE_GUIDE = (
    "catalogue.blocks groups related variables. Block fields are defaults inherited "
    "by their variables; variable fields override those defaults. id identifies a "
    "dataset column, label is its readable name, and block_id identifies the group. "
    "Use variable IDs in variable functions; rank_options alone takes a select-all block_id. "
    "ordered true uses values as the sequence; false means no usable order; a list gives "
    "the ordered subset and sequence. Sequence alone does not establish scale direction."
)
NUMERIC_BLOCK_CATALOGUE_GUIDE = (
    "catalogue.blocks groups related variables. Block fields are defaults inherited "
    "by their variables; variable fields override those defaults. Variable id and "
    "block_id are integers, unique within their respective namespaces in this study. "
    "Labels retain readable meaning; a block label, when present, gives its wording. "
    "Use variable IDs in variable functions, e.g. var(87); rank_options alone takes a "
    "select-all block_id, e.g. rank_options(12). "
    "ordered true uses values as the sequence; false means no usable order; a list gives "
    "the ordered subset and sequence. Sequence alone does not establish scale direction. "
    "The runtime resolves these references to the original dataset columns."
)


def numeric_references(cards: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    """Stable one-based references for a frozen, ordered set of variable cards.

    JSON object keys are strings. Catalogue references themselves are integers.
    Variable and block numbers have separate namespaces and are study-local.
    This mapping is a runtime sidecar, never part of the model's catalogue.
    """
    if not cards or len({c["id"] for c in cards}) != len(cards):
        raise ValueError("Numeric references require nonempty, unique variable IDs")
    return {
        "variables": {str(i): c["id"] for i, c in enumerate(cards, 1)},
        "blocks": {
            str(i): block for i, block in enumerate(dict.fromkeys(c["block_id"] for c in cards), 1)
        },
    }


def numeric_catalogue(
    cards: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Replace identifiers only; preserve labels, responses, orders and overrides."""
    references = numeric_references(cards)
    variables = {original: int(ref) for ref, original in references["variables"].items()}
    blocks = {original: int(ref) for ref, original in references["blocks"].items()}
    catalogue = grouped_catalogue(cards)
    catalogue["format"] = "variable-blocks-0.2.0"
    for block in catalogue["blocks"]:
        original = block["block_id"]
        block["block_id"] = blocks[original]
        # Some source identifiers are actual question wording. Retain that
        # meaning once, as a block label, rather than treating it as an opaque ID.
        if any(char.isspace() for char in original):
            block["label"] = original
        for variable in block["variables"]:
            variable["id"] = variables[variable["id"]]
    return catalogue, references


def numeric_program(
    program: str, references: dict[str, dict[str, str]], *, allow_invalid: bool = False
) -> str:
    """Encode API variable literals in deterministic reference targets.

    Only variable argument positions are changed. Response values, group labels,
    prose, thresholds and unrelated numbers keep their original meaning.
    Candidate execution uses runtime resolution, not AST rewriting.
    """
    reverse = {original: int(ref) for ref, original in references["variables"].items()}
    blocks = {original: int(ref) for ref, original in references["blocks"].items()}
    try:
        tree = ast.parse(program)
    except SyntaxError:
        if allow_invalid:
            return program  # Synthesis still needs the failed candidate's text.
        raise
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name in {
            "var",
            "eligible",
            "where",
            "group",
            "proportion",
            "top_box",
            "mean_score",
            "describe",
            "values",
            "distribution",
        }:
            keyword = "identifier" if name == "var" else "v"
            value = (
                node.args[0]
                if node.args
                else next((k.value for k in node.keywords if k.arg == keyword), None)
            )
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.value in reverse
            ):
                value.value = reverse[value.value]
        elif name == "rank_options":
            value = (
                node.args[0]
                if node.args
                else next((k.value for k in node.keywords if k.arg == "block"), None)
            )
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, str)
                and value.value in blocks
            ):
                value.value = blocks[value.value]
        elif name == "association":
            for position, keyword in ((0, "x"), (1, "y")):
                value = (
                    node.args[position]
                    if len(node.args) > position
                    else next((k.value for k in node.keywords if k.arg == keyword), None)
                )
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value in reverse
                ):
                    value.value = reverse[value.value]
        elif name in {"proxy", "adjusted_compare"}:
            position, keyword = (1, "variables") if name == "proxy" else (3, "controls")
            values = (
                node.args[position]
                if len(node.args) > position
                else next((k.value for k in node.keywords if k.arg == keyword), None)
            )
            if (
                isinstance(values, ast.Constant)
                and isinstance(values.value, str)
                and values.value in reverse
            ):
                values.value = reverse[values.value]
            if isinstance(values, ast.List):
                for value in values.elts:
                    if (
                        isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and value.value in reverse
                    ):
                        value.value = reverse[value.value]
    return ast.unparse(tree)


def numeric_evidence(
    evidence: dict[str, Any], references: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Use catalogue references in synthesis inputs; retain canonical saved evidence.

    Structured variable fields are mapped; evidence IDs, labels and values are
    unchanged. A deep copy keeps gold evidence and semantic traces intact.
    """
    reverse = {original: int(ref) for ref, original in references["variables"].items()}
    result = deepcopy(evidence)
    blocks = {original: int(ref) for ref, original in references["blocks"].items()}
    for row in [
        row for name in ("results", "estimates", "comparisons") for row in result.get(name, [])
    ]:
        if "variable" in row:
            row["variable"] = reverse.get(row["variable"], row["variable"])
        for field in ("variables", "controls"):
            if isinstance(row.get(field), list):
                row[field] = [reverse.get(v, v) for v in row[field]]
        if row.get("kind") == "ranking":
            row["block_id"] = blocks.get(row["block_id"], row["block_id"])
            for options in row["rankings"].values():
                for option in options:
                    option["variable"] = reverse.get(option["variable"], option["variable"])
    for trace in result.get("analysis_trace", {}).values():
        trace["variable"] = reverse.get(trace["variable"], trace["variable"])
    if result.get("study", {}).get("caveats_note"):
        result["study"]["caveats_note"] = numeric_reference_text(
            result["study"]["caveats_note"], references
        )
    return result


def numeric_reference_text(text: str, references: dict[str, dict[str, str]]) -> str:
    """Keep study warnings attached to numeric references without changing labels."""
    replacements = {
        original: f"{kind[:-1]} {ref}"
        for kind, mapping in references.items()
        for ref, original in mapping.items()
    }
    # Prefer variable references if a singleton block uses the same source ID.
    replacements.update(
        {original: f"variable {ref}" for ref, original in references["variables"].items()}
    )
    pattern = (
        r"(?<!\w)(?:"
        + "|".join(re.escape(k) for k in sorted(replacements, key=len, reverse=True))
        + r")(?!\w)"
    )
    return re.sub(pattern, lambda match: replacements[match.group()], text)


def grouped_catalogue(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """Readable block defaults plus exact variable IDs and necessary overrides.

    Defaults are chosen from the most common field value within each block.
    JSON keys distinguish values such as True and 1 when finding shared values.
    The original metadata and category sequences are never modified.
    """
    if not cards:
        raise ValueError("Catalogue is empty")
    groups: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        order = card["ordered_values"]
        variable = {
            k: card[k]
            for k in ANALYSIS_FIELDS
            if k in card and k not in {"block_id", "ordered_values"}
        }
        variable["ordered"] = False if order is None else True if order == card["values"] else order
        groups.setdefault(card["block_id"], []).append(variable)
    blocks = []
    for block_id, variables in groups.items():
        defaults = {}
        default_keys = {}
        for field in ["section", "answer_type", "values", "ordered"]:
            if not all(field in variable for variable in variables):
                continue
            keys = [
                json.dumps(variable[field], sort_keys=True, ensure_ascii=False)
                for variable in variables
            ]
            key = Counter(keys).most_common(1)[0][0]
            defaults[field] = variables[keys.index(key)][field]
            default_keys[field] = key
        children = [
            {
                k: v
                for k, v in variable.items()
                if k not in defaults
                or json.dumps(v, sort_keys=True, ensure_ascii=False) != default_keys[k]
            }
            for variable in variables
        ]
        blocks.append({"block_id": block_id, **defaults, "variables": children})
    return {"format": "variable-blocks-0.1.0", "blocks": blocks}
