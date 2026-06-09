"""Smoke + structural-invariant test for CX4, the CoDA contrast (SPEC-17 §6, H63).

CX4 trains a learned local model (the CoDA stand-in) plus three downstream `M_θ` (torch), so this is
``skipif``-guarded and runs a tiny instance: it asserts structural invariants (the three arms +
causal-validity rows appear; rates in range; the oracle augmenter's validity is exactly 1.0 by
construction; the run is deterministic), not the H63 verdict's magnitude — that is the committed
figure on the primary host (the SPEC-10 scale discipline).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # CX4 trains real distributed M_θ; skip where torch is absent

from verisim.experiments.cx4 import ARMS, CX4Config, run_cx4  # noqa: E402
from verisim.experiments.ed6 import ED6Config  # noqa: E402


def _tiny() -> CX4Config:
    return CX4Config(
        ed6=ED6Config(
            train_seeds=(0, 1), train_steps_per_traj=12, n_layer=1, n_head=2, n_embd=32,
            block_size=256, train_iters=20, batch_size=32, eval_seeds=(100,), eval_steps=6,
            m_interventions=3,
        ),
        k_augment=3, model_seeds=(0,),
    )


def test_cx4_smoke_structural() -> None:
    stats = run_cx4(_tiny())
    cells = {(s.arm, s.metric) for s in stats}
    for arm in ARMS:
        assert (arm, "intervention_exact") in cells
        assert (arm, "medium_recall") in cells
    for s in stats:
        assert 0.0 <= s.mean <= 1.0
        assert s.ci_lo <= s.mean <= s.ci_hi
    # the oracle augmenter is valid by construction; the learned one is a measured rate in [0,1].
    by = {(s.arm, s.metric): s for s in stats}
    assert by[("+oracle-aug", "causal_validity")].mean == 1.0
    assert 0.0 <= by[("+learned-aug", "causal_validity")].mean <= 1.0


def test_cx4_deterministic() -> None:
    def run() -> list[tuple[str, str, float]]:
        return [(s.arm, s.metric, round(s.mean, 4)) for s in run_cx4(_tiny())]
    assert run() == run()
