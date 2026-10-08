"""Respondent filters preserve their observed universe under Boolean composition."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype

from .errors import AnalysisError


@dataclass(frozen=True)
class Filter:
    mask: pd.Series
    universe: pd.Series

    def __post_init__(self) -> None:
        if not isinstance(self.mask, pd.Series) or not isinstance(
            self.universe, pd.Series
        ):
            raise AnalysisError("WrongFilter: mask and universe must be pandas Series")
        if (
            not self.mask.index.equals(self.universe.index)
            or not self.mask.index.is_unique
        ):
            raise AnalysisError(
                "FilterIndexMismatch: masks need the same unique respondent index"
            )
        if (
            not is_bool_dtype(self.mask.dtype)
            or not is_bool_dtype(self.universe.dtype)
            or self.mask.isna().any()
            or self.universe.isna().any()
        ):
            raise AnalysisError("WrongFilter: masks must contain nonmissing booleans")
        if (self.mask & ~self.universe).any():
            raise AnalysisError(
                "WrongFilter: selected respondents must belong to the observed universe"
            )

    def _check_other(self, other: Filter) -> None:
        if not isinstance(other, Filter):
            raise AnalysisError("WrongFilter: combine only respondent Filters")
        if not self.mask.index.equals(other.mask.index):
            raise AnalysisError(
                "FilterIndexMismatch: filters refer to different respondent indices"
            )

    def __invert__(self) -> Filter:
        return Filter(self.universe & ~self.mask, self.universe)

    def __and__(self, other: Filter) -> Filter:
        self._check_other(other)
        universe = self.universe & other.universe
        return Filter(self.mask & other.mask & universe, universe)

    def __or__(self, other: Filter) -> Filter:
        """Complete-case OR: the universe is the INTERSECTION of both universes.

        A respondent who satisfies one side but is outside the other side's
        observed base is excluded, so A | B can shrink the analysis base.
        """
        self._check_other(other)
        universe = self.universe & other.universe
        return Filter((self.mask | other.mask) & universe, universe)
