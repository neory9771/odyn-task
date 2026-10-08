"""Paired train/validation/test packages, built family-first.

Families are assigned to splits from metadata alone, before any reference program
is executed, so no outcome can influence which split a family lands in.
Everything here is local; no LLM calls.
"""

from .export import generate_splits
from .planning import audit_splits, validate_split_plan
from .settings import load_config

__all__ = ['audit_splits', 'generate_splits', 'load_config', 'validate_split_plan']
