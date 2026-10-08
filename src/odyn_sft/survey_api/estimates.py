"""Estimators: observed shares, distributions, rankings, top-box and mean scores."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

import numpy as np
import pandas as pd

from . import statistics as st
from .constants import REFUSAL
from .errors import AnalysisError
from .filters import Filter
from .statistics import wilson
from .types import (
    Estimate,
    EstimateRow,
    Groups,
    Result,
    StoredValue,
)


class EstimatesMixin:
    """Estimators: observed shares, distributions, rankings, top-box and mean scores. Mixed into SurveyAPI."""

    def _register(self, result: Estimate, v: str, operation: str, groups: Groups,
                  among: Filter | None, bases: dict[str, pd.Series], outcome: pd.Series) -> None:
        """Keep the estimate's exact respondents and outcomes for later tests."""
        self._estimates[result["id"]] = deepcopy(result)
        self._samples[result["id"]] = {"variable": v, "operation": operation, "bases": bases, "outcome": outcome}
        self._trace_estimate(result, v, operation, groups, bases)

    def _trace_estimate(self, result: Estimate, v: str, operation: str, groups: Groups,
                        bases: dict[str, pd.Series]) -> None:
        """Evaluation tracing hook; ordinary sessions record nothing extra."""

    def _proportion(
        self,
        v: str,
        selected: pd.Series,
        by: Groups | None,
        among: Filter | None,
        operation: str,
    ) -> Estimate:
        groups = self._groups(by)
        if among is not None:
            self._filter(among)
        estimates: list[EstimateRow] = []
        bases: dict[str, pd.Series] = {}
        for label, f in groups.items():
            base = self._analysis_base(v, operation) & f.mask
            if among is not None:
                base &= among.mask
            bases[label] = base
            n, k = int(base.sum()), int((base & selected).sum())
            flags = list(self.cards[v]["flags"])
            denominator_rule = self.cards[v]["denominator_rule"]
            if operation.startswith("top_box_") and any(
                value not in (self.cards[v]["ordered_values"] or [])
                for value in self.cards[v]["values"]
            ):
                flags.append("unranked_responses_excluded")
                denominator_rule = "ordered_item_excluding_unranked_responses"
            if n < 30:
                flags.append("small_n")
            if n < 10:
                flags.append("suppressed" if n else "empty_segment")
            elif k < 10 or n - k < 10:
                flags.append("sparse_count")
            estimates.append(
                {
                    "group": label,
                    "n": n,
                    "numerator": k if n >= 10 else None,
                    "value": k / n if n >= 10 else None,
                    "ci": wilson(k, n) if n >= 10 else None,
                    "flags": flags,
                    "denominator_rule": denominator_rule,
                }
            )
        result = cast(
            Estimate,
            self.append(
                "estimate",
                variable=v,
                variable_label=self.cards[v]["label"],
                operation=operation,
                estimates=estimates,
            ),
        )
        self._register(result, v, operation, groups, among, bases, selected.astype(bool))
        return result

    def proportion(
        self,
        v: str,
        value: StoredValue | None = None,
        by: Groups | None = None,
        among: Filter | None = None,
    ) -> Estimate:
        """Observed unweighted response share with Wilson CI and small-base policy."""
        self.var(v)
        if self.cards[v]["answer_type"] == "select_all":
            if value is not None and value is not True:
                raise AnalysisError(
                    "WrongAnswerType: select-all only supports selection=True"
                )
            value = True
        elif value is None or value not in self.cards[v]["values"]:
            raise AnalysisError(
                "UnknownValue: single-choice proportion requires a stored value"
            )
        else:
            self._reject_refusal(value)
        return self._proportion(
            v,
            self.data[v].eq(value).fillna(False),
            by,
            among,
            "selection" if value is True else f"value={value}",
        )

    def distribution(self, v: str, by: Groups | None = None,
                     among: Filter | None = None) -> dict[StoredValue, Estimate]:
        """Recorded category Estimates, keyed by response; each is comparable.

        Single/scale/rank items only. Refusals remain valid stored responses but
        are excluded from the observed denominator and this distribution.
        """
        self.var(v)
        if self.cards[v]["answer_type"] == "select_all":
            raise AnalysisError("WrongAnswerType: use proportion or rank_options for checkbox items")
        return {value: self.proportion(v, value, by=by, among=among)
                for value in self.values(v) if value != REFUSAL}

    def rank_options(self, block: str, by: Groups | None = None,
                     among: Filter | None = None, top: int | None = None) -> dict[str, list[Result]]:
        """Checkbox shares sorted within groups; suppressed values have no rank."""
        if not isinstance(block, str) or block not in self.blocks:
            raise AnalysisError("UnknownBlock: use a select-all catalogue block ID")
        if top is not None and (type(top) is not int or top < 1):
            raise AnalysisError("WrongTop: top must be a positive integer")
        variables = self.blocks[block]
        if any(self.cards[v]["answer_type"] != "select_all" for v in variables):
            raise AnalysisError("WrongAnswerType: rank_options requires a select-all block")
        ranked: dict[str, list[Result]] = {}
        for v in variables:
            estimate = self.proportion(v, by=by, among=among)
            for estimate_row in estimate["estimates"]:
                ranked.setdefault(estimate_row["group"], []).append({**deepcopy(estimate_row),
                    "variable": v, "variable_label": self.cards[v]["label"], "estimate_id": estimate["id"]})
        for label, rows in ranked.items():
            rows.sort(key=lambda row: (row["value"] is None,
                                      -(row["value"] or 0)))
            observed = [row for row in rows if row["value"] is not None]
            ranks = st.rank_positions([row["value"] for row in observed])
            for row, rank in zip(observed, ranks):
                row["rank"] = int(rank)
            for row in rows[len(observed):]:
                row["rank"] = None
            ranked[label] = rows if top is None else rows[:top]
        self.append("ranking", block_id=block, top=top, rankings=deepcopy(ranked))
        return ranked

    def top_box(
        self, v: str, k: int = 2, by: Groups | None = None, among: Filter | None = None
    ) -> Estimate:
        """Share in the last k categories of the documented sequence.

        The caller must check whether that end of the sequence matches the claim;
        ordering alone does not verify a substantive high/low direction.
        """
        self.var(v)
        card = self.cards[v]
        if card["answer_type"] == "rank":
            raise AnalysisError(
                "ScaleDirectionUnknown: use not_measured; rank direction is undocumented"
            )
        if not card["ordered_values"]:
            raise AnalysisError(
                "WrongAnswerType: top_box requires a documented ordered scale"
            )
        if type(k) is not int or not 1 <= k <= len(card["ordered_values"]):
            raise AnalysisError("WrongTopBox: invalid k")
        return self._proportion(
            v, self.data[v].isin(card["ordered_values"][-k:]), by, among, f"top_box_{k}"
        )

    def mean_score(
        self, v: str, by: Groups | None = None, among: Filter | None = None
    ) -> Estimate:
        """Mean ordered-scale position (1..K in documented order), with t intervals.

        Treats category positions as equally spaced; the evidence flags this
        assumption. Responses outside the documented order are excluded.
        """
        self.var(v)
        card = self.cards[v]
        if card["answer_type"] == "rank":
            raise AnalysisError(
                "ScaleDirectionUnknown: use not_measured; rank direction is undocumented"
            )
        if card["answer_type"] != "ordered_scale" or not card["ordered_values"]:
            raise AnalysisError(
                "WrongAnswerType: mean_score requires a documented ordered scale"
            )
        order = list(card["ordered_values"])
        scores = self.data[v].map({value: i + 1 for i, value in enumerate(order)}).astype(float)
        groups = self._groups(by)
        if among is not None:
            self._filter(among)
        estimates: list[EstimateRow] = []
        bases: dict[str, pd.Series] = {}
        for label, f in groups.items():
            base = self._analysis_base(v, "top_box_") & f.mask
            if among is not None:
                base &= among.mask
            bases[label] = base
            values = scores[base].to_numpy()
            n = len(values)
            flags = list(card["flags"]) + ["interval_scale_assumption"]
            if any(value not in order for value in card["values"]):
                flags.append("unranked_responses_excluded")
            if n < 30:
                flags.append("small_n")
            if n < 10:
                flags.append("suppressed" if n else "empty_segment")
            row = {
                "group": label,
                "n": n,
                "numerator": None,
                "value": float(values.mean()) if n >= 10 else None,
                "ci": st.mean_interval(values) if n >= 10 else None,
                "sd": float(values.std(ddof=1)) if n >= 10 else None,
                "flags": flags,
                "denominator_rule": "ordered_item_excluding_unranked_responses",
            }
            estimates.append(cast(EstimateRow, row))
        result = cast(
            Estimate,
            self.append(
                "estimate",
                variable=v,
                variable_label=card["label"],
                operation="mean_score",
                scale=order,
                score_coding="1..K in documented order",
                estimates=estimates,
            ),
        )
        self._register(result, v, "mean_score", groups, among, bases, scores)
        return result

    def _session_estimate(self, est: Estimate, call: str) -> dict[str, Any]:
        # Literal dictionaries in a generated program must not fabricate results.
        if (
            not isinstance(est, dict)
            or not isinstance(est.get("id"), str)
            or est["id"] not in self._estimates
            or est != self._estimates[est["id"]]
        ):
            raise AnalysisError(
                f"UnknownEstimate: {call} requires an unmodified estimate from this session"
            )
        return {r["group"]: r for r in est["estimates"]}

    def _values(self, est: Estimate, label: str) -> np.ndarray:
        sample = self._samples[est["id"]]
        return sample["outcome"][sample["bases"][label]].astype(float).to_numpy()
