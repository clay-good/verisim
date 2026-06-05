"""Distributed metric core (SPEC-7 §9; DS3): divergence, consistency-faithfulness, bits-to-correct.

Pure and dependency-free. ``divergence`` feeds the generic
:func:`verisim.metrics.horizon.faithful_horizon`, so the distributed ``H_ε(ρ)`` is defined as
for every prior world; ``consistency_faithfulness`` is the headline-new §9.1 metric.
"""

from verisim.distmetrics.bits import (
    bits_to_correct,
    correction_symbols,
    delta_exact,
    delta_exact_rate,
)
from verisim.distmetrics.divergence import (
    consistency_faithfulness,
    dist_facts,
    divergence,
    object_consistency_view,
)

__all__ = [
    "bits_to_correct",
    "consistency_faithfulness",
    "correction_symbols",
    "delta_exact",
    "delta_exact_rate",
    "dist_facts",
    "divergence",
    "object_consistency_view",
]
