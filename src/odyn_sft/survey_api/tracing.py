"""Optional respondent-population traces produced by ordinary library calls."""

from __future__ import annotations

import pandas as pd

from ..common.storage import digest
from .service import SurveyAPI
from .types import Estimate, Groups, Metadata


def mask_hash(mask: pd.Series) -> str:
    return digest([str(i) for i in mask.index[mask.fillna(False)]])


class TracedSurveyAPI(SurveyAPI):
    """The same statistical API with extra analysis semantics for evaluation."""

    def __init__(self, wide: pd.DataFrame, metadata: Metadata) -> None:
        super().__init__(wide, metadata)
        self.trace: dict[str, dict] = {}

    def _trace_estimate(self, result: Estimate, v: str, operation: str, groups: Groups,
                        bases: dict[str, pd.Series]) -> None:
        self.trace[result["id"]] = {
            "variable": v,
            "operation": operation,
            "groups": [
                {"group": label, "population_hash": mask_hash(f.mask), "base_hash": mask_hash(bases[label])}
                for label, f in groups.items()
            ],
        }
