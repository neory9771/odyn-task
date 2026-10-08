"""Statistical comparisons: group/threshold contrasts, trends, associations, adjusted comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import statistics as st
from .errors import AnalysisError
from .filters import Filter
from .statistics import contrast_label, difference_ci
from .types import (
    Estimate,
    Expectation,
    Result,
)


class ContrastsMixin:
    """Statistical comparisons: group/threshold contrasts, trends, associations, adjusted comparisons. Mixed into SurveyAPI."""

    def _unavailable(self, reason: str | None = None, **fields: Any) -> Result:
        """Keep valid analyses usable when their inferential procedure is undefined."""
        if reason is not None:
            fields["reason"] = reason
        flag = reason.split(":", 1)[0] if reason else "suppressed_or_empty"
        return self.append("contrast", **fields, value=None, ci=None, p_value=None,
                           label="unavailable", flags=[flag])

    def compare(
        self, est: Estimate, a: str, b: str | float, expect: Expectation = None
    ) -> Result:
        """Compare independent groups or a fixed threshold, as a minus b."""
        rows = self._session_estimate(est, "compare")
        if not isinstance(a, str):
            raise AnalysisError("UnknownSegment: use a group label from the estimate")
        if expect not in (None, ">", "<", "!="):
            raise AnalysisError("WrongExpectation: choose >, <, != or None")
        if isinstance(b, dict):
            raise AnalysisError(
                "WrongContrast: compare takes two labels from ONE estimate; "
                "use by=group(...) instead of passing a second Estimate"
            )
        if a not in rows or isinstance(b, str) and b not in rows:
            raise AnalysisError("UnknownSegment: use a group from the estimate")
        ra = rows[a]
        rb = rows[b] if isinstance(b, str) else None
        if isinstance(b, str) and a == b:
            raise AnalysisError(
                "WrongContrast: independent comparison requires two distinct groups"
            )
        means = est["operation"] == "mean_score"
        if rb is None:
            low, high = (1, len(est["scale"])) if means else (0, 1)
            if isinstance(b, bool) or not isinstance(b, (int, float)) or not low < b < high:
                raise AnalysisError(
                    f"WrongThreshold: use a value strictly between {low} and {high}"
                    if means else "WrongThreshold: use a probability strictly between 0 and 1"
                )
        fields: Result = {"estimate_id": est["id"], "a": a, "b": b, "expect": expect,
                  "contrast_type": "threshold" if rb is None else "difference"}
        if ra["value"] is None or rb is not None and rb["value"] is None:
            return self._unavailable(**fields)
        if means:
            try:
                if rb is None:
                    tested = st.one_sample_t(self._values(est, a), float(b))
                    d = ra["value"] - b
                else:
                    assert isinstance(b, str)
                    tested = st.welch(self._values(est, a), self._values(est, b))
                    d = ra["value"] - rb["value"]
            except AnalysisError as error:
                if error.code != "NoVariation":
                    raise
                return self._unavailable(reason=str(error), **fields)
            ci = tested["ci"]
            return self.append("contrast", **fields, value=float(d), ci=ci,
                               p_value=tested["p_value"], label=contrast_label(ci, expect),
                               test=tested["test"], cohens_h=None,
                               standardized_mean_difference=tested.get("standardized_mean_difference"),
                               standardizer=tested.get("standardizer"),
                               flags=["unweighted", "interval_scale_assumption"])
        ka, na = ra["numerator"], ra["n"]
        assert ka is not None  # Nonsuppressed rows always have counts and intervals.
        if rb is None:
            assert isinstance(b, (int, float))
            d = ra["value"] - b
            ci, p, method = st.threshold_test(ka, na, b)
            h = None
        else:
            kb, nb = rb["numerator"], rb["n"]
            assert kb is not None
            d = ka / na - kb / nb
            ci = difference_ci(ka, na, kb, nb)
            p, method = st.two_proportion_test(ka, na, kb, nb)
            h = st.cohens_h(ka / na, kb / nb)
        return self.append(
            "contrast",
            **fields,
            value=float(d),
            ci=list(map(float, ci)),
            p_value=p,
            label=contrast_label(ci, expect),
            test=method,
            cohens_h=h,
            flags=["unweighted"],
        )

    def trend(
        self, est: Estimate, order: list[str] | None = None, expect: str | None = None
    ) -> Result:
        """Linear trend across three or more ordered groups of one estimate.

        The slope is the change in share (or mean score) per ordered step, from
        OLS with HC3 robust errors. The program declares the order; groups are
        coded 0, 1, 2, ... so adjacent groups are assumed equally spaced.
        """
        rows = self._session_estimate(est, "trend")
        order = list(rows) if order is None else order
        if not isinstance(order, list) or len(order) < 3 or len(set(order)) != len(order) \
                or any(label not in rows for label in order):
            raise AnalysisError("WrongTrend: order needs three or more distinct group labels from the estimate")
        directions: dict[str | None, Expectation] = {None: None, "increasing": ">", "decreasing": "<", "any": "!="}
        if expect not in directions:
            raise AnalysisError("WrongExpectation: trend expect is increasing, decreasing, any or None")
        fields: Result = {"estimate_id": est["id"], "a": None, "b": None, "order": order, "expect": expect,
                  "contrast_type": "trend"}
        if any(rows[label]["value"] is None for label in order):
            return self._unavailable(**fields)
        values = [self._values(est, label) for label in order]
        positions = np.concatenate([np.full(len(v), i) for i, v in enumerate(values)])
        try:
            tested = st.linear_trend(positions, np.concatenate(values))
        except AnalysisError as error:
            if error.code != "NoVariation":
                raise
            return self._unavailable(reason=str(error), **fields)
        flags = ["unweighted", "group_order_declared_by_program", "equal_group_step_assumption"]
        if est["operation"] == "mean_score":
            flags.append("interval_scale_assumption")
        return self.append("contrast", **fields, value=tested["value"], ci=tested["ci"],
                           p_value=tested["p_value"], label=contrast_label(tested["ci"], directions[expect]),
                           test=tested["test"], cohens_h=None,
                           value_unit="per ordered step", flags=flags)

    def _category_labels(self, v: str) -> pd.Series:
        """Readable stored categories; checkbox blanks inside the base are unselected."""
        if self.cards[v]["answer_type"] == "select_all":
            return self.data[v].eq(True).fillna(False).map({True: "selected", False: "not selected"}).rename(v)
        return self.data[v].astype(str).rename(v)

    def _association_scale(self, v: str) -> tuple[str, pd.Series, pd.Series]:
        """Return (kind, base, values) for association; nominal or ordinal."""
        card = self.cards[v]
        if card["answer_type"] == "rank":
            raise AnalysisError("ScaleDirectionUnknown: rank items cannot enter association")
        base = self.eligible(v).mask
        if card["answer_type"] == "select_all":
            return "ordinal", base, self.data[v].eq(True).fillna(False).astype(float)
        if card["ordered_values"]:
            order = list(card["ordered_values"])
            base = base & self.data[v].isin(order)
            return "ordinal", base, self.data[v].map({value: i + 1 for i, value in enumerate(order)}).astype(float)
        return "nominal", base, self.data[v].astype(object)

    def association(
        self, x: str, y: str, among: Filter | None = None, expect: str | None = None
    ) -> Result:
        """Association between two variables in their joint observed base.

        Ordinal or binary pairs: Spearman rho with a paired bootstrap interval.
        Any nominal variable: chi-square with Cramer's V (no direction).
        """
        self.var(x)
        self.var(y)
        if x == y:
            raise AnalysisError("WrongAssociation: choose two distinct variables")
        directions: dict[str | None, Expectation] = {None: None, "positive": ">", "negative": "<", "any": "!="}
        if expect not in directions:
            raise AnalysisError("WrongExpectation: association expect is positive, negative, any or None")
        kx, bx, vx = self._association_scale(x)
        ky, by_, vy = self._association_scale(y)
        base = bx & by_
        if among is not None:
            base &= self._filter(among).mask
        n = int(base.sum())
        fields: Result = {"estimate_id": None, "a": None, "b": None, "variables": [x, y],
                  "variable_labels": [self.cards[x]["label"], self.cards[y]["label"]],
                  "expect": expect, "contrast_type": "association", "n": n}
        if n < 10:
            return self._unavailable(**fields)
        flags = ["unweighted", "joint_observed_base"]
        if n < 30:
            flags.append("small_n")
        if any(self.cards[v]["answer_type"] == "select_all" for v in (x, y)):
            flags.append("checkbox_blank_as_unselected")
        if kx == "ordinal" and ky == "ordinal":
            try:
                tested = st.rank_association(vx[base].to_numpy(), vy[base].to_numpy())
            except AnalysisError as error:
                if error.code != "NoVariation":
                    raise
                return self._unavailable(reason=str(error), **fields)
            if tested["ci"] is None:
                flags.append("interval_unavailable")
            label = (contrast_label(tested["ci"], directions[expect]) if tested["ci"] is not None
                     else "not_tested" if expect is None else "unavailable")
            return self.append("contrast", **fields, value=tested["value"], ci=tested["ci"],
                               p_value=tested["p_value"], label=label, test=tested["test"],
                               measure=tested["measure"], cohens_h=None, flags=flags,
                               ci_method=tested["ci_method"], bootstrap_resamples=tested["bootstrap_resamples"],
                               resampling_seed=tested["resampling_seed"],
                               permutation_resamples=tested["permutation_resamples"])
        if expect in ("positive", "negative"):
            raise AnalysisError("WrongExpectation: nominal associations have no direction; use any or None")
        table = pd.crosstab(self._category_labels(x)[base], self._category_labels(y)[base])
        try:
            nominal_test = st.nominal_association(table.to_numpy())
        except AnalysisError as error:
            if error.code != "NoVariation":
                raise
            return self._unavailable(reason=str(error), **fields)
        if nominal_test["sparse_table"]:
            flags.append("sparse_table")
        if expect is None:
            label = "not_tested"
        else:
            label = "consistent" if nominal_test["p_value"] < st.ALPHA else "no_clear_difference"
        counts = table.where(table >= 10)
        if counts.isna().any().any():
            flags.append("cells_suppressed")
        return self.append("contrast", **fields, value=nominal_test["value"], ci=None,
                           p_value=nominal_test["p_value"], label=label, test=nominal_test["test"],
                           measure=nominal_test["measure"], chi_square=nominal_test["chi_square"],
                           degrees_of_freedom=nominal_test["degrees_of_freedom"], cohens_h=None,
                           table={"rows": list(table.index), "columns": list(table.columns),
                                  "counts": [[None if pd.isna(c) else int(c) for c in row]
                                             for row in counts.to_numpy()]},
                           label_basis="unadjusted p < 0.05" if expect else None, flags=flags,
                           monte_carlo_resamples=nominal_test["monte_carlo_resamples"],
                           resampling_seed=nominal_test["resampling_seed"])

    def adjusted_compare(
        self, est: Estimate, a: str, b: str, controls: str | list[str], expect: Expectation = None
    ) -> Result:
        """Mantel-Haenszel comparison of two groups within strata of observed controls.

        Odds-ratio scale; strata with fewer than 10 respondents or without both
        groups are dropped and reported. The pooled odds ratio then refers to the
        retained strata, not the full comparison. Adjustment is not causal identification.
        """
        rows = self._session_estimate(est, "adjusted_compare")
        if est["operation"] == "mean_score":
            raise AnalysisError("WrongAnswerType: adjusted_compare supports proportion and top_box estimates")
        if not isinstance(a, str) or not isinstance(b, str) or a == b or a not in rows or b not in rows:
            raise AnalysisError("UnknownSegment: use two distinct group labels from the estimate")
        if expect not in (None, ">", "<", "!="):
            raise AnalysisError("WrongExpectation: choose >, <, != or None")
        controls = [controls] if isinstance(controls, str) else controls
        if not isinstance(controls, list) or not controls:
            raise AnalysisError("WrongControls: expected a variable ID or nonempty list of IDs")
        for v in controls:
            self.var(v)
            if self.cards[v]["answer_type"] == "rank":
                raise AnalysisError("WrongControls: rank items cannot define strata")
        sample = self._samples[est["id"]]
        if sample["variable"] in controls:
            raise AnalysisError("WrongControls: the outcome cannot be a control")
        fields: Result = {"estimate_id": est["id"], "a": a, "b": b, "expect": expect,
                  "contrast_type": "adjusted_difference", "controls": controls}
        if rows[a]["value"] is None or rows[b]["value"] is None:
            return self._unavailable(**fields)
        covered = pd.Series(True, index=self.data.index)
        for v in controls:
            covered &= self.eligible(v).mask
        # Tuple keys prevent category labels containing delimiters from merging strata.
        strata = pd.Series(list(zip(*(self._category_labels(v) for v in controls))), index=self.data.index)
        outcome = sample["outcome"]
        tables, kept, dropped, used = [], [], [], {a: 0, b: 0}
        sparse = False
        for stratum in sorted(strata[covered & (sample["bases"][a] | sample["bases"][b])].unique()):
            in_stratum = covered & strata.map(lambda key: key == stratum)
            cells = []
            for label in (a, b):
                members = sample["bases"][label] & in_stratum
                k, n = int((members & outcome).sum()), int(members.sum())
                cells.append((k, n))
            (ka, na), (kb, nb) = cells
            if na == 0 or nb == 0 or na + nb < 10:
                dropped.append({"stratum": list(stratum), "n": na + nb})
                continue
            table = np.array([[ka, na - ka], [kb, nb - kb]])
            sparse |= bool((st.expected_counts(table) < 5).any())
            tables.append(table)
            kept.append({"stratum": list(stratum), "n": na + nb})
            used[a] += na
            used[b] += nb
        flags = ["unweighted", "adjusted_for_observed_controls_not_causal"]
        if dropped:
            flags += ["strata_dropped", "estimand_retained_strata_only"]
        if sparse:
            flags.append("sparse_strata")
        fields.update(strata_kept=kept, strata_dropped=dropped,
                      excluded_missing_controls=int(((sample["bases"][a] | sample["bases"][b]) & ~covered).sum()),
                      n={a: used[a], b: used[b]},
                      estimand="pooled_odds_ratio_over_retained_strata")
        if not tables:
            return self._unavailable(**fields)
        try:
            tested = st.mantel_haenszel(tables)
        except AnalysisError as error:
            if error.code != "UndefinedOddsRatio":
                raise
            return self._unavailable(reason=str(error), **fields)
        if sparse and tested["breslow_day_p"] is not None:
            flags.append("homogeneity_test_unreliable")
        if tested["breslow_day_p"] is not None and tested["breslow_day_p"] < st.ALPHA:
            flags.append("heterogeneous_strata")
        return self.append("contrast", **fields, value=tested["odds_ratio"], ci=tested["odds_ratio_ci"],
                           value_scale="odds_ratio", p_value=tested["p_value"],
                           label=contrast_label(tested["log_ci"], expect), test=tested["test"],
                           crude_odds_ratio=tested["crude_odds_ratio"],
                           breslow_day_p=tested["breslow_day_p"], cohens_h=None, flags=flags)
