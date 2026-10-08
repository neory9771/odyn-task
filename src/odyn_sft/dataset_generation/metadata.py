"""Conservative metadata from the actual pinned release, with explicit curation."""

from pathlib import Path

import pandas as pd

from ..common.storage import digest, file_hash, read_json
from ..survey_metadata.cleaning import clean
from ..survey_metadata.interpretation import (
    DENOMINATOR_NOTE,
    RESPONSE_METADATA_NOTE,
    study_caveats,
)
from .ordering import category_ordering

AGREEMENT = "To what extent would you agree or disagree with the following statements about multi-attraction passes?"
RANK = "MC_WW_MC_RK_S_v1_16072020"
AGE = "dem_ww_age_dm_l_v1_14072020_tgt_none"
GENDER = "dem_ww_gender_rb_l_v3_14072020_tgt_none"
FREQUENCY = "sbeh_ww_trfreq_rb_07062021_none"
STATE = "dem_usa_state_dm_l_v1_27052021_tgt_none"
SETTLEMENT = "dem_ww_settlement_rb_s_v1_cint_24072020_none"
LABELS = {
    AGE: "Age band",
    GENDER: "Gender",
    FREQUENCY: "Travel frequency",
    STATE: "Declared US state",
    SETTLEMENT: "Settlement type",
}

# Shared interpretation, rather than a repeated ordering rationale per variable.


def build_metadata(source):
    source = Path(source)
    manifest = read_json(source / "download_manifest.json")
    for entry in manifest["files"]:
        if file_hash(source / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Source checksum mismatch: {entry['path']}")
    wide = pd.read_parquet(source / "data/respondents.parquet")
    cb = pd.read_csv(source / "docs/codebook.csv").fillna("")
    if wide.respondent_id.duplicated().any() or set(cb.col_id) != set(wide.columns) - {
        "respondent_id"
    }:
        raise ValueError("Respondent/codebook integrity failed")
    cards, excluded = [], []
    for row in cb.itertuples():
        col = row.col_id
        values = wide[col].dropna()
        reason = None
        if col in {"something_else", "total_time_none", "page05_tie_none"}:
            reason = "write_in_or_telemetry"
        elif values.nunique() == 1 and len(values) == len(wide):
            reason = "fully_observed_constant"
        if reason:
            excluded.append({"id": col, "reason": reason})
            continue
        block = col if row.block_id == "unknown" else row.block_id
        flags = ["unweighted"]
        if row.type == "grid_item":
            answer_type = "select_all"
            flags.append("eligibility_unknown")
            allowed = [True]
        elif row.block_id == AGREEMENT:
            answer_type = "ordered_scale"
            flags.append("acquiescence_risk")
            allowed = row.allowed_values.split("|")
        elif row.block_id == RANK:
            answer_type = "rank"
            flags.append("scale_direction_unknown")
            allowed = row.allowed_values.split("|")
        else:
            answer_type = "single"
            allowed = row.allowed_values.split("|")
        if row.block_id.startswith("LIFE_WW_LS_CB_L_v1_04062021"):
            flags.append("duplicate_block")
        cleaned = [clean(v) if isinstance(v, str) else v for v in allowed]
        if cleaned != allowed:
            flags.append("merged_labels")
        # Response format and category order are independent. Keep refusals and
        # other unranked answers in values; ordering must never recode responses.
        original_values = list(dict.fromkeys(cleaned))
        source_order = str(getattr(row, "order_source", "none") or "none")
        ordering = category_ordering(original_values, source_order, answer_type=answer_type)
        flags.extend(ordering.flags)
        cards.append(
            {
                "id": col,
                "source_column": row.original_column,
                "label": LABELS.get(col, row.item_label),
                "label_origin": "curated" if col in LABELS else "source",
                "block_id": block,
                "section": row.section,
                "answer_type": answer_type,
                "values": original_values,
                "ordered_values": ordering.ordered_values,
                "order_source": source_order,
                "ordering_basis": ordering.basis,
                "unranked_values": ordering.unranked_values,
                "denominator_rule": "observed_block_respondents"
                if answer_type == "select_all"
                else "answered_item_excluding_refusals",
                "flags": flags,
                "answered_n": int(len(values)),
                "review_status": "pending_human_review",
            }
        )
    metadata = {
        "version": "0.1.0-core",
        "source_revision": manifest["revision"],
        "source_manifest_hash": digest(manifest),
        "study_id": "quant_us_2022",
        "study": {
            "label": "US consumer travel survey",
            "period": "2022-03-22..2022-03-28",
            "n": len(wide),
            "weights": None,
            "scope": "single US panel sample; older-skewed; cross-sectional self-report",
            # The release provides no per-cell missingness reasons.
            # Explain this once, rather than repeating it on each card.
            "missingness_note": "Reasons for missing cells are not recorded. Checkbox blanks cannot distinguish not selected from not shown.",
            "response_metadata_note": RESPONSE_METADATA_NOTE,
            "denominator_note": DENOMINATOR_NOTE,
            "limitations": [
                "unweighted",
                "sample_only",
                "no_causal_identification",
                "true_routing_unknown",
            ],
        },
        "variables": cards,
        "excluded": excluded,
        "supported_generation": [
            "select_all_proportion",
            "agreement_top_two",
            "group_difference",
            "majority_threshold",
            "scope_notes",
        ],
        "unsupported_generation": [
            "adjusted_compare",
            "mean_score",
            "trend",
            "association",
            "cross_study",
            "weighted_estimates",
        ],
        "review_status": "pending_human_review",
    }
    caveats = study_caveats(cards)
    if caveats:
        metadata["study"]["caveats_note"] = caveats
    metadata["hash"] = digest(metadata)
    return metadata
