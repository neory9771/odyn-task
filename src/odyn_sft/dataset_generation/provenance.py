"""Fingerprint offline generators plus their runtime dependencies and presets."""

from pathlib import Path

from ..common.storage import file_hash
from ..survey_evaluation.provenance import implementation_hashes as runtime_hashes


def implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    files = sorted(
        p
        for p in (package / "dataset_generation").rglob("*")
        if p.is_file() and p.suffix in (".py", ".json")
    )
    return {**runtime_hashes(), **{str(p.relative_to(package)): file_hash(p) for p in files}}
