"""Contracts between task adapters and task-independent training machinery."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ReportLayout:
    directory: str
    tag: str
    filename: str


class GenerationScorer(Protocol):
    """Task owns reference lookup, output interpretation and quality metrics."""
    def case(self, source_index: int) -> dict[str, Any]: ...
    def score(self, completion: str, case: dict[str, Any]) -> dict[str, Any]: ...
    def outcome(self, record: dict[str, Any], case: dict[str, Any]) -> str: ...
    def accumulate(self, record: dict[str, Any], case: dict[str, Any]) -> None: ...
    def metrics(self, examples: int) -> dict[str, Any]: ...


class TaskAdapter(Protocol):
    name: str
    view: str
    scope: str
    diagnostic_scope: str
    comparison_metrics: tuple[str, ...]

    def validate_suite(self, suite: Path) -> dict[str, Any]: ...
    def record_path(self, suite: Path, split: str, *, inputs: bool = False) -> Path: ...
    def source_files(self, suite: Path, split: str) -> dict[str, str]: ...
    def implementation_hashes(self) -> dict[str, str]: ...
    def scorer(self, suite: Path, split: str) -> GenerationScorer: ...
    def report_layout(self, use_adapter: bool) -> ReportLayout: ...
