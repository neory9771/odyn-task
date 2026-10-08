"""Fixed vocabulary shared by every analysis call."""

from __future__ import annotations

from typing import Any

from .types import Expectation

# Refusals sit outside every observed base and can never be selected or grouped.
REFUSAL = "Prefer not to say"

# Word forms accepted for expect=, mapped to the comparison symbol.
EXPECT_SYMBOLS: dict[Any, Expectation] = {"increasing": ">", "decreasing": "<", "positive": ">",
                                          "negative": "<", "any": "!="}
