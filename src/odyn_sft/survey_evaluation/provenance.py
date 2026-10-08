"""Fingerprint execution/scoring and frozen-package validation dependencies."""

from pathlib import Path

from ..common.storage import file_hash
from ..survey_api.provenance import implementation_hashes as api_hashes


def implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    files = [
        p
        for name in ("survey_evaluation", "survey_metadata", "prompts")
        for p in sorted((package / name).rglob("*.py"))
    ]
    return {**api_hashes(), **{str(p.relative_to(package)): file_hash(p) for p in files}}
