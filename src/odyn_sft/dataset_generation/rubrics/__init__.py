"""Frozen, case-specific judge rubrics for the pilot, methodology and use-case suites.

Rubrics are agent-authored specifications, not human calibration labels. Building
a bundle makes no provider calls and never rescores historical results.
"""

from .bundle import prepare, validate_bundle
from .criteria import VERSION, build_rubric

__all__ = ['VERSION', 'build_rubric', 'prepare', 'validate_bundle']
