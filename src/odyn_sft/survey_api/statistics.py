"""Unweighted statistical procedures delegated to scipy/statsmodels.

This module holds no survey metadata or execution state. Every test, interval
and adjustment is a library call; only result packaging lives here.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats.contingency import association as contingency_association
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from .errors import AnalysisError
from .types import AdjustedTest, Expectation, MeanTest, NominalTest, RankTest, TrendTest

ALPHA = 0.05
RESAMPLING_SEED = 0
BOOTSTRAP_RESAMPLES = 1999
PERMUTATION_RESAMPLES = 9999


def validate_counts(k: int, n: int) -> None:
    """Reject invalid bases before division, including bool-as-integer inputs."""
    if (
        isinstance(k, bool)
        or isinstance(n, bool)
        or not isinstance(k, Integral)
        or not isinstance(n, Integral)
        or n <= 0
        or not 0 <= k <= n
    ):
        raise AnalysisError("InvalidCounts: require integer 0 <= k <= n and n > 0")


def _floats(values: Iterable[Any]) -> list[float]:
    return [float(v) for v in values]


def _sample(values: np.ndarray, minimum: int = 2) -> np.ndarray:
    """Validate a finite one-dimensional sample before calling a procedure."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < minimum or not np.isfinite(array).all():
        raise AnalysisError("InvalidSample: require a finite one-dimensional sample with sufficient observations")
    return array


def _paired(x: np.ndarray, y: np.ndarray, minimum: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x, y = _sample(x, minimum), _sample(y, minimum)
    if len(x) != len(y):
        raise AnalysisError("InvalidSample: paired samples must have equal lengths")
    return x, y


def wilson(k: int, n: int) -> list[float]:
    """Wilson 95% interval for an observed binomial proportion."""
    validate_counts(k, n)
    return _floats(proportion_confint(k, n, alpha=ALPHA, method="wilson"))


def difference_ci(ka: int, na: int, kb: int, nb: int) -> list[float]:
    """Newcombe hybrid-score interval for a difference of independent proportions."""
    validate_counts(ka, na)
    validate_counts(kb, nb)
    return _floats(confint_proportions_2indep(ka, na, kb, nb, method="newcomb", compare="diff", alpha=ALPHA))


def expected_counts(table: np.ndarray) -> np.ndarray:
    return np.asarray(stats.contingency.expected_freq(table))


def two_proportion_test(ka: int, na: int, kb: int, nb: int) -> tuple[float, str]:
    """Pooled z-test, or Fisher's exact test when any expected cell is below 5.

    The expected-cell rule is a pragmatic sparse-table heuristic, not a sharp
    boundary at which the z approximation becomes invalid.
    """
    validate_counts(ka, na)
    validate_counts(kb, nb)
    table = np.array([[ka, na - ka], [kb, nb - kb]])
    if (expected_counts(table) < 5).any():
        return float(stats.fisher_exact(table).pvalue), "fisher_exact_sparse_fallback"
    return float(proportions_ztest([ka, kb], [na, nb])[1]), "two_proportion_z"


def threshold_test(k: int, n: int, threshold: float) -> tuple[list[float], float, str]:
    """Interval for p - threshold and a two-sided test from the same family.

    Each pair is an exact inversion, so the interval excludes zero exactly when
    p < 0.05. Default: Wilson interval with the score z-test (null-variance SE).
    When n*threshold or n*(1-threshold) is below 5 (a pragmatic sparse heuristic):
    Clopper-Pearson interval with the central exact binomial test.
    """
    validate_counts(k, n)
    if isinstance(threshold, bool) or not np.isfinite(threshold) or not 0 < threshold < 1:
        raise AnalysisError("WrongThreshold: require a finite probability strictly between 0 and 1")
    if min(n * threshold, n * (1 - threshold)) < 5:
        low, high = proportion_confint(k, n, alpha=ALPHA, method="beta")
        tail = min(stats.binom.cdf(k, n, threshold), stats.binom.sf(k - 1, n, threshold))
        p, method = min(1.0, 2 * float(tail)), "exact_binomial_central"
    else:
        low, high = proportion_confint(k, n, alpha=ALPHA, method="wilson")
        z = (k / n - threshold) / np.sqrt(threshold * (1 - threshold) / n)
        p, method = float(2 * stats.norm.sf(abs(z))), "wilson_score"
    return [float(low - threshold), float(high - threshold)], p, method


def cohens_h(pa: float, pb: float) -> float:
    if not all(np.isfinite(p) and 0 <= p <= 1 for p in (pa, pb)):
        raise AnalysisError("InvalidProbability: probabilities must lie between 0 and 1")
    return float(proportion_effectsize(pa, pb))


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment in input order."""
    if not p_values:
        return []
    if not all(np.isfinite(p) and 0 <= p <= 1 for p in p_values):
        raise AnalysisError("InvalidPValue: Holm requires finite p-values between 0 and 1")
    return _floats(multipletests(p_values, alpha=ALPHA, method="holm")[1])


def mean_interval(scores: np.ndarray) -> list[float]:
    """t interval for an unweighted mean; degenerate when all scores are equal."""
    scores = _sample(scores)
    if np.ptp(scores) == 0:
        return [float(scores[0]), float(scores[0])]
    return _floats(DescrStatsW(scores).tconfint_mean(alpha=ALPHA))


def welch(a: np.ndarray, b: np.ndarray) -> MeanTest:
    """Welch interval/test for a difference in means plus a standardised difference.

    The standardiser is the unweighted root mean of the two sample variances,
    sqrt((s_a^2 + s_b^2) / 2), not the df-weighted pooled SD of classical Cohen's d.
    """
    a, b = _sample(a), _sample(b)
    if np.ptp(a) == 0 and np.ptp(b) == 0:
        raise AnalysisError("NoVariation: both groups have constant scores")
    interval = CompareMeans.from_data(a, b).tconfint_diff(alpha=ALPHA, usevar="unequal")
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    spread = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"ci": _floats(interval), "p_value": float(p), "test": "welch_t",
            "standardized_mean_difference": float((a.mean() - b.mean()) / spread) if spread else None,
            "standardizer": "sqrt((sd_a^2 + sd_b^2) / 2)"}


def one_sample_t(scores: np.ndarray, value: float) -> MeanTest:
    scores = _sample(scores)
    if not np.isfinite(value):
        raise AnalysisError("WrongThreshold: require a finite reference value")
    if np.ptp(scores) == 0:
        raise AnalysisError("NoVariation: constant scores cannot be tested against a threshold")
    low, high = DescrStatsW(scores).tconfint_mean(alpha=ALPHA)
    return {"ci": [float(low - value), float(high - value)],
            "p_value": float(stats.ttest_1samp(scores, value).pvalue), "test": "one_sample_t"}


def linear_trend(positions: np.ndarray, outcomes: np.ndarray) -> TrendTest:
    """OLS slope per ordered step with HC3 robust errors; not a rank/trend score test.

    Positions are 0, 1, 2, ...: the slope assumes adjacent groups are equally spaced.
    """
    positions, outcomes = _paired(positions, outcomes)
    if np.ptp(positions) == 0:
        raise AnalysisError("NoVariation: trend positions must vary")
    if np.ptp(outcomes) == 0:
        raise AnalysisError("NoVariation: constant outcomes have no trend")
    fit = sm.OLS(outcomes, sm.add_constant(positions.astype(float))).fit(cov_type="HC3")
    low, high = fit.conf_int(alpha=ALPHA)[1]
    return {"value": float(fit.params[1]), "ci": [float(low), float(high)],
            "p_value": float(fit.pvalues[1]), "test": "ols_linear_trend_hc3"}


def _spearman_statistic(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
    """Library rank/correlation calls; bootstrap samples are ranked anew."""
    return np.asarray(stats.pearsonr(stats.rankdata(x, axis=axis),
                                   stats.rankdata(y, axis=axis), axis=axis).statistic)


def rank_association(x: np.ndarray, y: np.ndarray) -> RankTest:
    """Spearman rho, paired bootstrap CI and small-sample permutation p-value.

    A Pearson Fisher-z interval assumes a different sampling model even when
    applied to ranks. Resample respondent pairs and recompute their ranks.
    Undefined resample correlations make the CI unavailable, not fabricated.
    """
    x, y = _paired(x, y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise AnalysisError("NoVariation: an associated variable is constant in the joint base")
    rho = stats.spearmanr(x, y)
    p, p_method = float(rho.pvalue), "spearman_asymptotic"
    if len(x) <= 500:
        ranked_x, ranked_y = stats.rankdata(x), stats.rankdata(y)
        def statistic(sample: np.ndarray, axis: int = -1) -> np.ndarray:
            return np.asarray(stats.pearsonr(sample, ranked_y, axis=axis).statistic)
        permuted = stats.permutation_test((ranked_x,), statistic,
            permutation_type="pairings", vectorized=True, batch=128,
            n_resamples=PERMUTATION_RESAMPLES, alternative="two-sided",
            random_state=np.random.default_rng(RESAMPLING_SEED))
        p, p_method = float(permuted.pvalue), "spearman_permutation"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        warnings.simplefilter("ignore", stats.DegenerateDataWarning)
        boot = stats.bootstrap((x, y), _spearman_statistic, paired=True,
            vectorized=True, batch=128, n_resamples=BOOTSTRAP_RESAMPLES,
            confidence_level=1 - ALPHA, method="percentile",
            random_state=np.random.default_rng(RESAMPLING_SEED))
    interval = _floats(boot.confidence_interval)
    ci = interval if np.isfinite(interval).all() else None
    return {"value": float(rho.statistic), "ci": ci, "p_value": p,
            "test": p_method, "measure": "spearman_rho",
            "ci_method": "paired_percentile_bootstrap", "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "resampling_seed": RESAMPLING_SEED,
            "permutation_resamples": PERMUTATION_RESAMPLES if len(x) <= 500 else None}


def nominal_association(table: np.ndarray) -> NominalTest:
    """Cramér's V; Pearson, Fisher (sparse 2x2) or Monte Carlo (sparse RxC) test.

    V is the uncorrected statistic, biased upward in small samples and large tables.
    """
    table = np.asarray(table)
    if table.ndim != 2 or min(table.shape) < 2:
        raise AnalysisError("NoVariation: association needs at least two observed categories per variable")
    if not np.isfinite(table).all() or (table < 0).any() or (table != np.floor(table)).any():
        raise AnalysisError("InvalidCounts: contingency tables require finite nonnegative counts")
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        raise AnalysisError("NoVariation: contingency table contains an empty margin")
    chi2 = stats.chi2_contingency(table, correction=False)
    sparse = bool((chi2.expected_freq < 5).any())
    p, test = float(chi2.pvalue), "pearson_chi_square"
    if sparse and table.shape == (2, 2):
        p, test = float(stats.fisher_exact(table).pvalue), "fisher_exact_sparse_fallback"
    elif sparse:
        sampled = stats.chi2_contingency(table, correction=False,
            method=stats.MonteCarloMethod(n_resamples=PERMUTATION_RESAMPLES,
                                         rng=np.random.default_rng(RESAMPLING_SEED)))
        p, test = float(sampled.pvalue), "pearson_chi_square_monte_carlo"
    return {"value": float(contingency_association(table, method="cramer", correction=False)),
            "ci": None, "p_value": p, "test": test, "measure": "cramers_v",
            "chi_square": float(chi2.statistic), "degrees_of_freedom": int(chi2.dof), "sparse_table": sparse,
            "monte_carlo_resamples": PERMUTATION_RESAMPLES if test == "pearson_chi_square_monte_carlo" else None,
            "resampling_seed": RESAMPLING_SEED if test == "pearson_chi_square_monte_carlo" else None}


def mantel_haenszel(tables: list[np.ndarray]) -> AdjustedTest:
    """Cochran–Mantel–Haenszel test and pooled odds ratio over 2x2 strata.

    The pooled odds ratio refers to the strata supplied. Breslow-Day is
    asymptotic and unreliable when strata are sparse.
    """
    if not tables:
        raise AnalysisError("InvalidCounts: require at least one 2x2 stratum")
    if any(np.asarray(table).shape != (2, 2) for table in tables):
        raise AnalysisError("InvalidCounts: every stratum must be a 2x2 table")
    stack = np.stack(tables, axis=2).astype(float)
    if stack.shape[:2] != (2, 2) or not np.isfinite(stack).all() or (stack < 0).any() \
            or (stack != np.floor(stack)).any() or (stack.sum(axis=(0, 1)) == 0).any():
        raise AnalysisError("InvalidCounts: require finite nonnegative 2x2 stratum counts")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stratified = StratifiedTable(stack, shift_zeros=False)
        odds = float(stratified.oddsratio_pooled)
        low, high = stratified.oddsratio_pooled_confint(alpha=ALPHA)
        p = float(stratified.test_null_odds(correction=False).pvalue)
        homogeneity = float(stratified.test_equal_odds().pvalue) if len(tables) > 1 else None
        crude = Table2x2(stack.sum(axis=2), shift_zeros=False)
        crude_odds = float(crude.oddsratio)
    if not all(np.isfinite(v) and v > 0 for v in (odds, low, high)) or not np.isfinite(p):
        raise AnalysisError("UndefinedOddsRatio: pooled odds ratio is zero or infinite in the kept strata")
    return {"odds_ratio": odds, "odds_ratio_ci": [float(low), float(high)], "p_value": p,
            "log_ci": [float(np.log(low)), float(np.log(high))], "test": "cochran_mantel_haenszel",
            "breslow_day_p": homogeneity if homogeneity is not None and np.isfinite(homogeneity) else None,
            "crude_odds_ratio": crude_odds if np.isfinite(crude_odds) else None}


def rank_positions(shares: list[float]) -> list[int]:
    """Descending option ranks; ties receive the same minimum rank."""
    return [int(rank) for rank in stats.rankdata(-np.asarray(shares), method="min")]


def _check_expect(expect: Expectation) -> None:
    if expect not in (None, ">", "<", "!="):
        raise AnalysisError("WrongExpectation: choose >, <, != or None")


def holm_label(direction: float | None, p_holm: float | None, expect: Expectation) -> str:
    """Agreement with a predeclared direction after Holm adjustment.

    direction is the effect's sign on the null-centred scale (difference, slope,
    rho or log odds ratio); None for undirected tests such as chi-square.
    """
    _check_expect(expect)
    if p_holm is None:
        return "unavailable"
    if expect is None:
        return "not_tested"
    if p_holm >= ALPHA:
        return "no_clear_difference"
    if expect == "!=":
        return "consistent"
    if direction is None or direction == 0:
        return "no_clear_difference"
    return "consistent" if (direction > 0) == (expect == ">") else "inconsistent"


def contrast_label(ci: list[float], expect: Expectation) -> str:
    """Describe agreement with a predeclared direction using an unadjusted CI."""
    _check_expect(expect)
    if expect is None:
        return "not_tested"
    if ci[0] <= 0 <= ci[1]:
        return "no_clear_difference"
    positive = ci[0] > 0
    if expect == "!=":
        return "consistent"
    return "consistent" if positive == (expect == ">") else "inconsistent"
