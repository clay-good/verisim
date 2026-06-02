"""The host oracle (SPEC-6 §5): protocol, step result, and the Tier-A reference host oracle
that composes the v0 FS sub-oracle with the process/fd/credential glue (HC0)."""

from __future__ import annotations

from .base import EXIT_ERR, EXIT_OK, HostOracle, HostStepResult
from .reference import ReferenceHostOracle

__all__ = ["EXIT_ERR", "EXIT_OK", "HostOracle", "HostStepResult", "ReferenceHostOracle"]
