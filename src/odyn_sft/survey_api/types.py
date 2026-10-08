"""Public Python contracts for metadata, filters and JSON evidence."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

from .filters import Filter

StoredValue: TypeAlias = str | bool | int | float
Expectation: TypeAlias = Literal[">", "<", "!="] | None
Groups: TypeAlias = dict[str, Filter]
Result: TypeAlias = dict[
    str, Any
]  # Heterogeneous note/contrast records at JSON boundary.


class MeanTest(TypedDict):
    ci: list[float]
    p_value: float
    test: str
    standardized_mean_difference: NotRequired[float | None]
    standardizer: NotRequired[str]


class TrendTest(MeanTest):
    value: float


class RankTest(TypedDict):
    value: float
    ci: list[float] | None
    p_value: float
    test: str
    measure: str
    ci_method: str
    bootstrap_resamples: int
    permutation_resamples: int | None
    resampling_seed: int


class NominalTest(TypedDict):
    value: float
    ci: None
    p_value: float
    test: str
    measure: str
    chi_square: float
    degrees_of_freedom: int
    sparse_table: bool
    monte_carlo_resamples: int | None
    resampling_seed: int | None


class AdjustedTest(TypedDict):
    odds_ratio: float
    odds_ratio_ci: list[float]
    p_value: float
    log_ci: list[float]
    test: str
    breslow_day_p: float | None
    crude_odds_ratio: float | None


class VariableCard(TypedDict):
    id: str
    label: str
    block_id: str
    answer_type: Literal["select_all", "single", "ordered_scale", "rank"]
    values: list[StoredValue]
    ordered_values: list[StoredValue] | None
    flags: list[str]
    denominator_rule: str
    section: NotRequired[str]
    order_source: NotRequired[str]
    ordering_basis: NotRequired[str | None]
    unranked_values: NotRequired[list[StoredValue]]


class Metadata(TypedDict):
    hash: str
    variables: list[VariableCard]
    study: dict[str, Any]


class EstimateRow(TypedDict):
    group: str
    n: int
    numerator: int | None
    value: float | None
    ci: list[float] | None
    flags: list[str]
    denominator_rule: str
    sd: NotRequired[float | None]


class Estimate(TypedDict):
    id: str
    subclaim: str
    kind: Literal["estimate"]
    variable: str
    variable_label: str
    operation: str
    estimates: list[EstimateRow]
    scale: NotRequired[list[StoredValue]]


class EvidencePack(TypedDict):
    api_version: str
    metadata_hash: str
    study: dict[str, Any]
    results: list[Result]
    test_count: int
    label_rule: str
    label_holm_rule: str
    analysis_trace: NotRequired[dict[str, Any]]
