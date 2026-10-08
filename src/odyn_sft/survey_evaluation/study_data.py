"""Load already-produced study data for execution and scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.storage import read_json
from ..survey_api import normalize_data
from .types import DataObject, PathLike


def load_inputs(source: PathLike, metadata_path: PathLike) -> tuple[pd.DataFrame, DataObject]:
    """Load local study artifacts and normalise stored response labels once."""
    source, metadata_path = Path(source), Path(metadata_path)
    metadata = read_json(metadata_path)
    data = normalize_data(pd.read_parquet(source), metadata)
    return data, metadata
