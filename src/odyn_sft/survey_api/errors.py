"""Stable, machine-readable failures with optional source locations."""

from __future__ import annotations

from typing import Any


class AnalysisError(ValueError):
    """Retain legacy error text while exposing a code and program location."""

    def __init__(
        self, message: str, *, line: int | None = None, column: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = message.split(":", 1)[0]
        self.line = line
        self.column = column

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "line": self.line,
            "column": self.column,
        }

    def located(self, line: int | None, column: int | None = None) -> AnalysisError:
        return type(self)(
            str(self),
            line=self.line if self.line is not None else line,
            column=self.column if self.column is not None else column,
        )
