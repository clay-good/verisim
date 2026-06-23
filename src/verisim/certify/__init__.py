"""SPEC-28 -- the Coverage Certifier: prove a deployed agent guardrail's completeness, don't assert.

Point it at a guardrail you shipped (a Claude Code PreToolUse hook, a denylist) and it returns the
concrete commands that realize harm off the checked surface, oracle-proven, plus a coverage
certificate. See :mod:`verisim.certify.core` and ``python -m verisim.certify audit --hook <path>``.
"""

from __future__ import annotations

from .core import CertifyResult, certify_denylist, certify_hook, certify_monitor

__all__ = ["CertifyResult", "certify_denylist", "certify_hook", "certify_monitor"]
