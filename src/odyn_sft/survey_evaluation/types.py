"""Shared JSON boundary and path types for execution/scoring."""

from pathlib import Path
from typing import Any, TypeAlias

DataObject: TypeAlias = dict[str, Any]
PathLike: TypeAlias = str | Path
Groups: TypeAlias = dict[str, list[str]]
