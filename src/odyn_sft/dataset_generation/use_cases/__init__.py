"""Executable use-case coverage suite, separate from the frozen client benchmark.

The proposed vocabulary and product workflows are not all implemented. Their cases
exercise correct capability boundaries, not nonexistent numerical functionality.
"""

from .build import prepare
from .catalogue import USE_CASES
from .fixture import fixture

__all__ = ['USE_CASES', 'fixture', 'prepare']
