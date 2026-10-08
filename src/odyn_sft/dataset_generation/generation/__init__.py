"""Client-evaluation benchmark generation. Local only: no provider clients or LLM calls.

Audit -> review a capacity-derived plan -> generate reference drafts.
This package deliberately does not evaluate a model or certify a release.
"""

from .candidates import annotate_candidate, candidates
from .cases import materialize, reference_perspective, validate_reference
from .evidence import evidence_direction, fixed_direction, interval, stability
from .inventory import blocked_cores, enumerate_analyses
from .pipeline import audit, generate, lexical_audit
from .scenarios import STYLES, validate_scenario_bindings
from .version import VERSION

__all__ = ['STYLES', 'VERSION', 'annotate_candidate', 'audit', 'blocked_cores', 'candidates', 'enumerate_analyses', 'evidence_direction', 'fixed_direction', 'generate', 'interval', 'lexical_audit', 'materialize', 'reference_perspective', 'stability', 'validate_reference', 'validate_scenario_bindings']
