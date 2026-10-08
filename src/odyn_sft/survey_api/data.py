"""Validate survey boundaries and normalise stored response labels."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype

from ..survey_metadata.cleaning import clean
from .errors import AnalysisError
from .types import Metadata


def validate_inputs(wide: pd.DataFrame, metadata: Metadata) -> None:
    """Reject inconsistent metadata/indices rather than silently align respondents."""
    if (
        not isinstance(wide, pd.DataFrame)
        or not wide.index.is_unique
        or not wide.columns.is_unique
    ):
        raise AnalysisError(
            "InvalidData: require a DataFrame with unique index and columns"
        )
    if (
        not isinstance(metadata, dict)
        or not isinstance(metadata.get("hash"), str)
        or not isinstance(metadata.get("study"), dict)
        or not isinstance(metadata.get("variables"), list)
    ):
        raise AnalysisError("InvalidMetadata: require hash, study and variables")
    seen = set()
    for card in metadata["variables"]:
        required = {
            "id",
            "label",
            "block_id",
            "answer_type",
            "values",
            "ordered_values",
            "flags",
            "denominator_rule",
        }
        if not isinstance(card, dict) or not required <= card.keys():
            raise AnalysisError("InvalidMetadata: incomplete variable card")
        if any(
            not isinstance(card[k], str) or not card[k]
            for k in ("id", "label", "block_id", "denominator_rule")
        ):
            raise AnalysisError(
                "InvalidMetadata: variable identifiers and descriptions must be nonempty strings"
            )
        if card["id"] in seen:
            raise AnalysisError("InvalidMetadata: duplicate variable ID " + card["id"])
        seen.add(card["id"])
        if card["id"] not in wide:
            raise AnalysisError("MissingColumn: " + card["id"])
        if (
            card["answer_type"] not in {"select_all", "single", "ordered_scale", "rank"}
            or not isinstance(card["values"], list)
            or not isinstance(card["flags"], list)
            or any(not isinstance(f, str) for f in card["flags"])
        ):
            raise AnalysisError("InvalidMetadata: invalid variable response definition")
        ordered = card["ordered_values"]
        if ordered is not None and (
            not isinstance(ordered, list)
            or any(v not in card["values"] for v in ordered)
            or len({(type(v).__name__, str(v)) for v in ordered}) != len(ordered)
        ):
            raise AnalysisError("InvalidMetadata: inconsistent ordered categories")
        if card["answer_type"] == "ordered_scale" and not ordered:
            raise AnalysisError(
                "InvalidMetadata: ordered scale requires documented ordering"
            )
        unranked = card.get("unranked_values", [])
        if (
            not isinstance(unranked, list)
            or any(value not in card["values"] or value in (ordered or []) for value in unranked)
            or len({(type(v).__name__, str(v)) for v in unranked}) != len(unranked)
        ):
            raise AnalysisError("InvalidMetadata: inconsistent unranked categories")
        if card["answer_type"] == "select_all":
            column = wide[card["id"]]
            if not is_bool_dtype(column.dtype) and any(
                not isinstance(v, bool) for v in column.dropna().unique()
            ):
                raise AnalysisError(
                    "InvalidData: select-all responses must be booleans or missing"
                )


def normalize_data(wide: pd.DataFrame, metadata: Metadata) -> pd.DataFrame:
    """Return a cached normalised frame or a copy; never mutate caller data."""
    validate_inputs(wide, metadata)
    # Only label normalisation is cached; boundary invariants are always checked.
    if (
        isinstance(wide, pd.DataFrame)
        and isinstance(metadata, dict)
        and wide.attrs.get("survey_metadata_hash") == metadata.get("hash")
        and wide.attrs.get("survey_api_validated") == 1
    ):
        return wide
    result = wide.copy()
    for card in metadata["variables"]:
        if card["answer_type"] != "select_all":
            result[card["id"]] = (
                result[card["id"]]
                .astype("object")
                .map(lambda v: clean(v) if isinstance(v, str) else v if pd.notna(v) else None)
            )
    result.attrs["survey_metadata_hash"] = metadata["hash"]
    result.attrs["survey_api_validated"] = 1
    return result
