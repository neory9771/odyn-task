"""SurveyAPI: one analysis session over one study, composed from the call groups below."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .contrasts import ContrastsMixin
from .data import normalize_data
from .errors import AnalysisError
from .estimates import EstimatesMixin
from .evidence import EvidenceMixin
from .filters import Filter
from .types import Estimate, Metadata, Result
from .vocabulary import VocabularyMixin


class SurveyAPI(VocabularyMixin, EstimatesMixin, ContrastsMixin, EvidenceMixin):
    """One analysis session with shared metadata, filters and evidence provenance.

    Create one session per program. Respondent data are read-only by convention;
    sessions are not shared across concurrent callers. No estimator implies
    survey representativeness or a causal interpretation.
    """

    def __init__(self, wide: pd.DataFrame, metadata: Metadata) -> None:
        self.data = normalize_data(wide, metadata)
        self.cards = {v["id"]: v for v in metadata["variables"]}
        self.metadata = metadata
        self.blocks: dict[str, list[str]] = {}
        for card in self.cards.values():
            self.blocks.setdefault(card["block_id"], []).append(card["id"])
        self.log: list[Result] = []
        self.current: str | None = None
        self.counts: dict[str, int] = {}
        self._estimates: dict[str, Estimate] = {}
        # Per-respondent outcomes behind each estimate, for tests that need raw data.
        self._samples: dict[str, dict[str, Any]] = {}

    def _filter(self, value: Filter) -> Filter:
        """Require filters from the session's exact respondent index."""
        if not isinstance(value, Filter):
            raise AnalysisError("WrongFilter: expected a respondent Filter")
        if not value.mask.index.equals(self.data.index):
            raise AnalysisError(
                "FilterIndexMismatch: filter does not match this survey"
            )
        return value

    @staticmethod
    def _text(value: str, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise AnalysisError("WrongText: " + name + " must be a nonempty string")
