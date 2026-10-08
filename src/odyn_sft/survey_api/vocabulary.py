"""Program vocabulary: resolve variables, respondent filters, groups and subclaims."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from difflib import SequenceMatcher

import pandas as pd

from .constants import REFUSAL
from .errors import AnalysisError
from .filters import Filter
from .types import (
    Groups,
    StoredValue,
)


class VocabularyMixin:
    """Program vocabulary: resolve variables, respondent filters, groups and subclaims. Mixed into SurveyAPI."""

    def var(self, identifier: str) -> str:
        """Resolve an exact stored variable ID without guessing labels."""
        if not isinstance(identifier, str):
            raise AnalysisError("UnknownVariable: expected a catalogue ID string")
        if identifier not in self.cards:
            raise AnalysisError(
                f"UnknownVariable: {identifier!r}; use an ID from the supplied variable cards"
            )
        return identifier

    def find_variables(self, query: str, k: int = 10, section: str | None = None) -> list[str]:
        """Local text discovery, returning exact handles; no embedding service."""
        self._text(query, "query")
        if type(k) is not int or k < 1:
            raise AnalysisError("WrongSearch: k must be a positive integer")
        if section is not None:
            self._text(section, "section")
        query = query.casefold()
        matches: list[tuple[float, str]] = []
        for v, card in self.cards.items():
            if section is not None and card.get("section") != section:
                continue
            texts = [v, card["label"], card["block_id"], *map(str, card["values"])]
            scores = [1.0 if query in text.casefold() else
                      SequenceMatcher(None, query, text.casefold()).ratio() for text in texts]
            score = max(scores)
            if score >= 0.3:
                matches.append((score, v))
        # Stable catalogue order breaks ties; discovery does not resolve ambiguities.
        return [v for _, v in sorted(matches, key=lambda pair: -pair[0])[:k]]

    def values(self, v: str) -> list[StoredValue]:
        """Return stored categories in documented order, then unranked categories."""
        self.var(v)
        card = self.cards[v]
        ordered = list(card["ordered_values"] or [])
        return deepcopy(ordered + [value for value in card["values"] if value not in ordered])

    def describe(self, v: str) -> str:
        """A metadata card and observed counts, without interpreting missingness."""
        self.var(v)
        return json.dumps({**self.cards[v], "n_recorded": int(self.data[v].notna().sum()),
                           "n_observed_base": int(self.eligible(v).mask.sum())}, ensure_ascii=False)

    def eligible(self, v: str) -> Filter:
        """Observed item/block base; it does not establish true question routing."""
        self.var(v)
        card = self.cards[v]
        if card["answer_type"] == "select_all":
            mask = self.data[self.blocks[card["block_id"]]].notna().any(axis=1)
        else:
            mask = self.data[v].notna() & ~self.data[v].isin([REFUSAL])
        return Filter(mask, mask)

    def where(self, v: str, values: StoredValue | list[StoredValue]) -> Filter:
        """Match stored responses inside the variable's observed universe."""
        self.var(v)
        values = values if isinstance(values, list) else [values]
        for value in values:
            if self.cards[v]["answer_type"] == "select_all" and type(value) is not bool:
                raise AnalysisError(
                    "UnknownValue: select-all response values must be booleans"
                )
            if value not in self.cards[v]["values"]:
                raise AnalysisError(
                    f"UnknownValue: {value!r} for {v}; allowed: {self.cards[v]['values']}"
                )
            self._reject_refusal(value)
        base = self.eligible(v).mask
        return Filter(self.data[v].isin(values) & base, base)

    @staticmethod
    def _reject_refusal(value: StoredValue) -> None:
        """Refusals are outside every observed base, so they cannot be selected."""
        if value == REFUSAL:
            raise AnalysisError(
                f"RefusalExcluded: {REFUSAL!r} is excluded from the observed base; "
                "its share would always be 0"
            )

    def _analysis_base(self, v: str, operation: str) -> pd.Series:
        """An ordered statistic needs a ranked response in its denominator."""
        base = self.eligible(v).mask
        if operation.startswith("top_box_"):
            base = base & self.data[v].isin(self.cards[v]["ordered_values"] or [])
        return base

    def group(
        self,
        v: str | Groups,
        mapping: dict[str, StoredValue | list[StoredValue]] | None = None,
    ) -> Groups:
        """Construct disjoint groups from stored categories or named Filters."""
        if isinstance(v, dict):
            if mapping is not None:
                raise AnalysisError(
                    "WrongGroup: filter mapping does not take a second argument"
                )
            groups = v
        else:
            self.var(v)
            mapping = (
                {str(value): [value] for value in self.cards[v]["values"] if value != REFUSAL}
                if mapping is None
                else mapping
            )
            if not isinstance(mapping, dict):
                raise AnalysisError("WrongGroup: expected a stored-value mapping")
            groups = {label: self.where(v, values) for label, values in mapping.items()}
        if not groups or not all(isinstance(f, Filter) for f in groups.values()):
            raise AnalysisError("WrongGroup: expected named filters")
        masks = list(groups.values())
        for label, f in groups.items():
            self._text(label, "group label")
            self._filter(f)
        for i, a in enumerate(masks):
            for b in masks[i + 1 :]:
                if (a.mask & b.mask).any():
                    raise AnalysisError(
                        "OverlappingSegments: make groups mutually exclusive"
                    )
        return groups

    @contextmanager
    def subclaim(self, identifier: str, text: str) -> Iterator[None]:
        """Associate evidence with a proposition, restoring context on failure."""
        self._text(identifier, "subclaim identifier")
        self._text(text, "subclaim text")
        previous = self.current
        self.current = identifier
        self.append("subclaim", text=text)
        try:
            yield
        finally:
            self.current = previous

    def _groups(self, by: Groups | None) -> Groups:
        if by is not None:
            return self.group(by)
        everyone = pd.Series(True, index=self.data.index)
        return {"All observed respondents": Filter(everyone, everyone.copy())}
