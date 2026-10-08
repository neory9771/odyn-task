"""Typed local survey analysis and native Python execution."""

from .data import normalize_data, validate_inputs
from .errors import AnalysisError
from .filters import Filter
from .provenance import implementation_digest, implementation_hashes
from .runtime import (
    CALLS,
    DEFAULT_LIMITS,
    ProcessLimits,
    execute_program,
    run_program,
    run_reference,
    static_check,
)
from .service import SurveyAPI
from .statistics import contrast_label, difference_ci, wilson
from .types import (
    Estimate,
    EstimateRow,
    EvidencePack,
    Expectation,
    Groups,
    Metadata,
    VariableCard,
)
from .version import IMPLEMENTATION_VERSION, VERSION

__all__ = [
    "VERSION",
    "IMPLEMENTATION_VERSION",
    "AnalysisError",
    "Filter",
    "normalize_data",
    "validate_inputs",
    "wilson",
    "difference_ci",
    "contrast_label",
    "SurveyAPI",
    "CALLS",
    "ProcessLimits",
    "DEFAULT_LIMITS",
    "static_check",
    "run_program",
    "execute_program",
    "run_reference",
    "implementation_hashes",
    "implementation_digest",
    "Metadata",
    "VariableCard",
    "Groups",
    "Estimate",
    "EstimateRow",
    "EvidencePack",
    "Expectation",
]
