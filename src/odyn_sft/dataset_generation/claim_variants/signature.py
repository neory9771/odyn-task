"""Claim meaning signatures parsed from reference Survey API programs.

The reference program fixes what a claim means (measures, groups, populations,
comparison, direction, gaps). A paraphrase may change the wording, never this signature.
"""
from __future__ import annotations

import ast
from typing import Any

from ..types import DataObject

VERSION = "claim-signature-1.0.1"
CAUSAL_CONCEPTS = ("causal",)


class SignatureError(ValueError):
    pass


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _kwargs(call: ast.Call) -> dict[str, ast.AST]:
    return {k.arg: k.value for k in call.keywords}


class _Parser:
    def __init__(self, variables: dict[str, DataObject]):
        self.variables = variables
        self.groups: dict[str, DataObject] = {}
        self.estimates: dict[str, DataObject] = {}

    def label(self, node: ast.AST) -> str:
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "var"):
            raise SignatureError("Expected var(...)")
        vid = _literal(node.args[0])
        if vid not in self.variables:
            raise SignatureError(f"Unknown variable {vid}")
        return self.variables[vid]["label"]

    def population(self, node: ast.AST) -> str:
        """Readable filter expression; operators keep their logic explicit."""
        if isinstance(node, ast.Call) and node.func.id == "where":
            values = _literal(node.args[1])
            values = values if isinstance(values, list) else [values]
            return f"{self.label(node.args[0])} is {' or '.join(map(str, values))}"
        if isinstance(node, ast.Call) and node.func.id == "eligible":
            return f"answered {self.label(node.args[0])}"
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.BitAnd)):
            joiner = " OR " if isinstance(node.op, ast.BitOr) else " AND "
            return f"({self.population(node.left)}{joiner}{self.population(node.right)})"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return f"NOT ({self.population(node.operand)})"
        raise SignatureError(f"Unsupported population expression: {ast.dump(node)[:80]}")

    def group(self, call: ast.Call) -> DataObject:
        if isinstance(call.args[0], ast.Dict):  # group({label: Filter})
            return {label: self.population(f) for label, f in
                    zip((_literal(k) for k in call.args[0].keys), call.args[0].values)}
        variable = self.label(call.args[0])
        mapping = _literal(call.args[1])
        return {label: f"{variable} is {' or '.join(map(str, values if isinstance(values, list) else [values]))}"
                for label, values in mapping.items()}

    def estimate(self, call: ast.Call) -> DataObject:
        kind, kw = call.func.id, _kwargs(call)
        variable = self.label(call.args[0])
        if kind == "top_box":
            measure = f"gives one of the {_literal(kw.get('k', ast.Constant(2)))} highest responses to {variable!r}"
        elif "value" in kw:
            measure = f"answers {_literal(kw['value'])!r} to {variable!r}"
        else:
            measure = f"selects {variable!r}"
        return {"measure": measure,
                "groups": self.groups[kw["by"].id] if "by" in kw else None,
                "population": self.population(kw["among"]) if "among" in kw else None}


def signature(program: str, variables: list[DataObject], claim: str) -> DataObject:
    """Parse a reference program into claim parts, gaps and proxies."""
    parser = _Parser({v["id"]: v for v in variables})
    parts: list[DataObject] = []
    gaps: list[DataObject] = []
    proxies: list[DataObject] = []

    def visit(statements: list[ast.stmt], subclaim: str | None) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.With):
                call = stmt.items[0].context_expr
                visit(stmt.body, _literal(call.args[0]))
                continue
            value = stmt.value if isinstance(stmt, (ast.Assign, ast.Expr)) else None
            if not isinstance(value, ast.Call):
                continue
            name = value.func.id
            if name == "group" and isinstance(stmt, ast.Assign):
                parser.groups[stmt.targets[0].id] = parser.group(value)
            elif name in {"proportion", "top_box"} and isinstance(stmt, ast.Assign):
                parser.estimates[stmt.targets[0].id] = parser.estimate(value)
            elif name == "compare":
                est = parser.estimates[value.args[0].id]
                a, b = _literal(value.args[1]), _literal(value.args[2])
                expect = _literal(_kwargs(value)["expect"])
                groups = est["groups"] or {}
                if isinstance(b, (int, float)):
                    comparison = {"type": "threshold", "threshold": b,
                                  "direction": {">": "more than", "<": "fewer than"}[expect]}
                    involved = {a: groups.get(a)}
                else:
                    comparison = {"type": "difference",
                                  "direction": {">": "more often than", "<": "less often than", "!=": "differently from"}[expect]}
                    involved = {a: groups.get(a), b: groups.get(b)}
                parts.append({"part_id": subclaim, "measure": est["measure"], "groups": involved,
                              "population": est["population"], "comparison": comparison,
                              "a": a, "b": b, "expect": expect})
            elif name == "not_measured":
                gaps.append({"part_id": subclaim, "concept": _literal(value.args[0]), "reason": _literal(value.args[1])})
            elif name == "proxy":
                proxies.append({"concept": _literal(value.args[0]),
                                "variables": [parser.variables[v]["label"] for v in _literal(value.args[1])],
                                "reason": _literal(value.args[2])})

    visit(ast.parse(program).body, None)
    if not parts and not gaps:
        raise SignatureError("Program has no claim parts")
    return {"version": VERSION, "reference_claim": claim, "parts": parts, "gaps": gaps, "proxies": proxies,
            "causal_clause": any(c in g["concept"] for g in gaps for c in CAUSAL_CONCEPTS),
            "label_text": _label_text(parts, proxies, claim)}


def _label_text(parts: list[DataObject], proxies: list[DataObject], claim: str) -> str:
    """All text a paraphrase may legitimately take numbers from (labels, values, claim)."""
    pieces = [claim]
    for p in parts:
        pieces += [p["measure"], p["population"] or "", *(str(g) for g in p["groups"].values()), str(p["a"]), str(p["b"])]
    for x in proxies:
        pieces += x["variables"]
    return " | ".join(pieces)


def render(sig: DataObject) -> list[DataObject]:
    """Numbered meaning items for the paraphrase writer and the meaning checker."""
    items = []
    for i, p in enumerate(sig["parts"], 1):
        c = p["comparison"]
        if c["type"] == "threshold":
            text = f"{c['direction']} {c['threshold']:.0%} of {p['a']} {p['measure']}"
        else:
            text = f"{p['a']} {p['measure']} {c['direction']} {p['b']}"
        definitions = "; ".join(f"{label} means {definition}" for label, definition in p["groups"].items() if definition)
        items.append({"id": f"P{i}", "kind": "part", "statement": text,
                      "group_definitions": definitions or None,
                      "population": f"only among respondents where {p['population']}" if p["population"] else None})
    for i, g in enumerate(sig["gaps"], 1):
        items.append({"id": f"G{i}", "kind": "gap", "statement": f"the claim also asserts {g['concept']}",
                      "note": g["reason"]})
    for i, x in enumerate(sig["proxies"], 1):
        items.append({"id": f"X{i}", "kind": "proxy",
                      "statement": f"{', '.join(x['variables'])} is offered as an indirect measure of {x['concept']}"})
    return items
