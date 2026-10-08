"""Domain types for the benchmark's methodological annotations.

Availability describes evidence required for the requested inference, including
identification/design evidence; observed associations alone are only partial
evidence for a causal claim.
Statistical evidence direction describes a measured proposition, never the truth
of the whole claim. Multiple inference domains may apply to one component.
"""

from pathlib import Path
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict

PathLike: TypeAlias = str | Path
DataObject: TypeAlias = dict[str, Any]  # External metadata/evidence JSON boundary.
Groups: TypeAlias = dict[str, list[str]]
Counts: TypeAlias = list[tuple[int, int]]  # Each pair is (observed base, numerator).
EvidenceDirection: TypeAlias = Literal[
    "support", "contradict", "inconclusive", "unavailable", "not_applicable"
]
Availability: TypeAlias = Literal["available", "partially_available", "unavailable"]
Composition: TypeAlias = Literal["single", "multiple", "linked"]
InferenceDomain: TypeAlias = Literal[
    "statistical_inference",
    "measurement_construct_validity",
    "survey_population_inference",
    "causal_identification",
    "transportability",
    "temporal_comparability",
]


class StabilityAudit(TypedDict):
    rule: str
    stable: bool
    original: EvidenceDirection
    perturbed_states: list[EvidenceDirection]
    limitation: str


class AnalysisRecord(TypedDict):
    """One fixed-direction estimand with an independently computed reference."""

    core: str
    row: DataObject
    threshold: bool
    direction: int
    direction_commitment: str
    counts: Counts
    ci: list[float] | None
    E: EvidenceDirection
    S: list[str]
    diagnostic_flags: list[str]
    stability: StabilityAudit


class Candidate(TypedDict):
    """Unworded claim specification; absence cases use scenario keys, not fake cores."""

    scenario: str
    analyses: list[AnalysisRecord]
    family: str
    scenario_keys: list[str]
    target: NotRequired[str]
    context: NotRequired[AnalysisRecord]
    context_keys: NotRequired[list[str]]
    dimensions: DataObject


class EvaluationConfig(TypedDict):
    workspace: str
    source: str
    metadata: str
    overlap_corpus: list[str]
    heldout_outcome_blocks: list[str]
    size: int
    calibration_size: int
    seed: int
    coverage_minima: dict[str, dict[str, int]]
    interaction_minima: dict[str, int]
    study_bindings: NotRequired[DataObject]
    scenario_rules: DataObject
