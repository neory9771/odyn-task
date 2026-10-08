"""Evidence records: gaps, proxies, caveats, segment sizes and the final evidence pack."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from . import statistics as st
from .constants import EXPECT_SYMBOLS
from .errors import AnalysisError
from .filters import Filter
from .types import (
    EvidencePack,
    Groups,
    Result,
)
from .version import VERSION


class EvidenceMixin:
    """Evidence records: gaps, proxies, caveats, segment sizes and the final evidence pack. Mixed into SurveyAPI."""

    def append(self, kind: str, **fields: Any) -> Result:
        """Collect a stable evidence ID; this helper is not a program callable."""
        sid = self.current or "scope"
        self.counts[sid] = self.counts.get(sid, 0) + 1
        row = {
            "id": f"{sid}.r{self.counts[sid]}",
            "subclaim": sid,
            "kind": kind,
            **fields,
        }
        self.log.append(row)
        return row

    def not_measured(self, concept: str, reason: str) -> Result:
        """Record missing measurement or identification evidence, with a reason."""
        self._text(concept, "concept")
        self._text(reason, "reason")
        return self.append("note", concept=concept, reason=reason, label="not_measured")

    def proxy(self, concept: str, variables: list[str], rationale: str) -> Result:
        """Record an indirect measure; estimate it separately when relevant."""
        self._text(concept, "concept")
        self._text(rationale, "rationale")
        if not isinstance(variables, list) or not variables:
            raise AnalysisError("WrongProxy: expected a nonempty list of catalogue IDs")
        for v in variables:
            self.var(v)
        return self.append(
            "note",
            concept=concept,
            variables=variables,
            reason=rationale,
            label="proxy_variable",
        )

    def caveat(self, text: str) -> Result:
        """Record a concise methodological qualification of at most 200 characters."""
        self._text(text, "caveat")
        if len(text) > 200:
            raise AnalysisError("CaveatTooLong: maximum 200 characters")
        return self.append("note", concept="scope", reason=text, label="caveat")

    def segment_sizes(self, g: Groups, among: Filter | None = None) -> Result:
        """Count segment members without applying any outcome's observed base."""
        g = self.group(g)
        if among is not None:
            self._filter(among)
        return self.append(
            "segment_sizes",
            segments={
                label: int((f.mask & (among.mask if among else True)).sum())
                for label, f in g.items()
            },
        )

    @staticmethod
    def _direction(r: Result) -> float | None:
        """Effect sign on the null-centred scale; None for undirected measures."""
        value = r.get("value")
        if value is None or r.get("measure") == "cramers_v":
            return None
        if r.get("value_scale") == "odds_ratio":
            return float(np.sign(np.log(value)))
        return float(np.sign(value))

    def pack(self) -> EvidencePack:
        """Snapshot evidence; Holm covers every executed, nonsuppressed test."""
        tests = [
            r for r in self.log if r["kind"] == "contrast" and r["p_value"] is not None
        ]
        for r, adjusted in zip(tests, st.holm([r["p_value"] for r in tests])):
            r["p_holm"] = adjusted
        for r in self.log:
            if r["kind"] != "contrast":
                continue
            if r["p_value"] is None:
                r["p_holm"] = None
            r["label_holm"] = st.holm_label(self._direction(r), r["p_holm"],
                                            EXPECT_SYMBOLS.get(r.get("expect"), r.get("expect")))
        return {
            "api_version": VERSION,
            "metadata_hash": self.metadata["hash"],
            "study": deepcopy(self.metadata["study"]),
            "results": deepcopy(self.log),
            "test_count": len(tests),
            "label_rule": "unadjusted_95pct_CI_vs_declared_expectation",
            "label_holm_rule": "holm_adjusted_p_lt_0.05_with_estimate_sign_vs_declared_expectation",
        }
