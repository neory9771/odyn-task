"""Stable relative-path fingerprints for the entire shipped SFT implementation."""

from __future__ import annotations

import os
from pathlib import Path

from ..common.storage import file_hash


def implementation_hashes() -> dict[str, str]:
    """Hash nested core files plus launchers, including a supervisor override."""
    package = Path(__file__).resolve().parent
    hashes = {str(p.relative_to(package)): file_hash(p) for p in sorted(package.rglob("*.py"))}
    for name in ("cli.py", "supervisor.py", "probe.py"):
        hashes["run_management/" + name] = file_hash(package.parent / "run_management" / name)
    supervisor = Path(
        os.environ.get("ODYN_SUPERVISOR_SOURCE", str(package.parent / "run_management" / "supervisor.py"))
    )
    hashes["supervisor"] = file_hash(supervisor)
    return hashes
