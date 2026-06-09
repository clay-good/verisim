"""Smoke + structural-invariant test for RS6, the per-compute Pareto (SPEC-16 §5, H58).

RS6 trains the *real* structured graph arm with all four rollout-aware trainers (torch), so this is
``skipif``-guarded and runs a tiny instance: it asserts *structural* invariants (every trainer ×
compute-budget cell appears; compute is monotone in steps; horizons/rates are in range; the run is
deterministic), not the shape of the frontier — that is the committed figure on the primary host
(the SPEC-10 scale discipline). The H58 charge accounting — teacher forcing / noise pay zero model
forwards for data, self-forcing / unrolled pay a positive, refresh-cadence-scaled charge — is pinned
directly on ``_datagen_forwards`` since that asymmetry is what makes the per-compute compare fair.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # RS6 trains a real graph arm; skip where torch is absent

from verisim.experiments.rs6_pareto import (  # noqa: E402
    TRAINERS,
    RS6Config,
    _datagen_forwards,
    run_rs6,
)


def _tiny() -> RS6Config:
    return RS6Config(
        n_hosts=4, n_ports=2, train_seeds=(0, 1), train_steps_per_traj=16,
        graph_d_model=32, graph_mp_rounds=2, compute_steps=(40, 80),
        sf_refresh_every=30, unroll_refresh_every=30, unroll_k=2, model_seeds=(0,),
        eval_seeds=(100, 101), eval_steps=12, one_step_seeds=(200,), one_step_steps=12,
        headline_epsilon=0.3,
    )


def test_rs6_smoke_structural() -> None:
    points = run_rs6(_tiny())
    cells = {(p.trainer, p.steps) for p in points}
    assert cells == {(t, s) for t in TRAINERS for s in (40, 80)}
    for p in points:
        assert 0.0 <= p.p_one_step <= 1.0
        assert 0.0 <= p.h_free <= 12.0
        assert p.h_lo <= p.h_free <= p.h_hi
        assert p.compute > 0 and p.forwards > 0
        assert p.n == 1
    # Compute is strictly monotone in the gradient-step budget within every trainer.
    for trainer in TRAINERS:
        per = sorted((p for p in points if p.trainer == trainer), key=lambda q: q.steps)
        assert per[0].compute < per[1].compute


def test_rs6_datagen_charge_asymmetry() -> None:
    """The H58 charge: teacher forcing / noise pay zero model forwards for data; self-forcing and
    the unrolled pushforward pay a positive, refresh-cadence-scaled charge — the per-compute gap."""
    cfg = _tiny()
    assert _datagen_forwards("teacher-forced", 80, cfg) == 0
    assert _datagen_forwards("noise-injected", 80, cfg) == 0
    assert _datagen_forwards("self-forced", 80, cfg) > 0
    assert _datagen_forwards("unrolled", 80, cfg) > 0
    # More gradient steps -> more dataset refreshes -> at least as large a data-gen charge.
    assert _datagen_forwards("self-forced", 80, cfg) >= _datagen_forwards("self-forced", 40, cfg)


def test_rs6_deterministic() -> None:
    def run() -> list[tuple[str, int, float]]:
        return [(p.trainer, p.steps, round(p.h_free, 4)) for p in run_rs6(_tiny())]
    assert run() == run()
