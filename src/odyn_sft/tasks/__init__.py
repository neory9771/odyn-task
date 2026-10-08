"""Lazy task registry: the training engine never imports survey scoring directly."""
from __future__ import annotations

from .base import TaskAdapter


def get_task(name: str) -> TaskAdapter:
    if name == 'sft1':
        from .sft1 import AnalysisTask
        return AnalysisTask()
    if name == 'sft2':
        from .sft2 import SynthesisTask
        return SynthesisTask()
    raise ValueError('Unknown SFT task: ' + name)
