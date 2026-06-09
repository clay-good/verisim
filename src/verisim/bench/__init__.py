"""The Verisim product layer: a frozen faithfulness benchmark + ACD environment (SPEC-18).

Hardens the program's accumulated asset -- an oracle-grounded world-model faithfulness benchmark
with
**ground-truth labels** -- into a versioned, reproducible, community-usable artifact:

  - :mod:`~verisim.bench.manifest` -- the frozen battery + Croissant / datasheet / model-card;
  - :mod:`~verisim.bench.leaderboard` -- the `H_ε` leaderboard + Kendall-τ rank stability (H65);
  - :mod:`~verisim.bench.conformance` -- Gymnasium / verifiers / Inspect contract checks (PB-pack).
"""

from .conformance import ConformanceResult, all_passed, run_conformance
from .leaderboard import (
    LeaderRow,
    RankStability,
    build_leaderboard,
    kendall_tau,
    score_proposer,
)
from .manifest import (
    BENCH_VERSION,
    REFERENCE_PROPOSERS,
    BatteryManifest,
    Proposer,
    croissant_metadata,
    datasheet,
    model_card,
)

__all__ = [
    "BENCH_VERSION",
    "REFERENCE_PROPOSERS",
    "BatteryManifest",
    "ConformanceResult",
    "LeaderRow",
    "Proposer",
    "RankStability",
    "all_passed",
    "build_leaderboard",
    "croissant_metadata",
    "datasheet",
    "kendall_tau",
    "model_card",
    "run_conformance",
    "score_proposer",
]
