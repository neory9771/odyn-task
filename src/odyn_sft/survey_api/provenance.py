"""Fingerprint the public facade and every package implementation dependency."""

from __future__ import annotations

from pathlib import Path

from ..common.storage import digest, file_hash


def implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    files = [
        package / "survey_api/__init__.py",
        package / "survey_metadata/catalogue.py",
        package / "survey_metadata/cleaning.py",
        package / "common/storage.py",
        *sorted((package / "survey_api").rglob("*.py")),
    ]
    return {str(p.relative_to(package)): file_hash(p) for p in files}


def implementation_digest() -> str:
    return digest(implementation_hashes())
