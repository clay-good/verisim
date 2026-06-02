"""Host metrics (SPEC-6 §9, HC3): composed + per-subsystem divergence, bits-to-correct, the
composition-faithfulness diagnostic (H13), privilege-faithfulness, and the run-record schema.

The deterministic metric core for the host world -- no runtime deps, no GPU, like v0's
:mod:`verisim.metrics` and the network's :mod:`verisim.netmetrics`. The faithful horizon ``H_ε``
(SPEC-6 §9.3) is reused verbatim from v0 (:func:`~verisim.metrics.horizon.faithful_horizon`), since
its definition is unchanged -- it operates on a per-step divergence trajectory regardless of world.
"""

from __future__ import annotations

from verisim.metrics.horizon import faithful_horizon

from .bits import (
    bits_to_correct,
    bits_to_correct_by_subsystem,
    correction_symbols,
    correction_symbols_by_subsystem,
    delta_exact,
    edit_symbols,
)
from .composition import CompositionLaw, composition_law
from .divergence import (
    SUBSYSTEMS,
    composed_faithful,
    divergence,
    divergence_by_subsystem,
    facts_by_subsystem,
    host_facts,
    step_faithful_by_subsystem,
)
from .privilege import privilege_faithfulness
from .record import HostRunRecord, read_host_records, write_host_records

__all__ = [
    "SUBSYSTEMS",
    "CompositionLaw",
    "HostRunRecord",
    "bits_to_correct",
    "bits_to_correct_by_subsystem",
    "composed_faithful",
    "composition_law",
    "correction_symbols",
    "correction_symbols_by_subsystem",
    "delta_exact",
    "divergence",
    "divergence_by_subsystem",
    "edit_symbols",
    "facts_by_subsystem",
    "faithful_horizon",
    "host_facts",
    "privilege_faithfulness",
    "read_host_records",
    "step_faithful_by_subsystem",
    "write_host_records",
]
