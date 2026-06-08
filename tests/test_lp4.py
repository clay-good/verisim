"""Smoke + invariant test for LP4 edge-metric ablation (SPEC-12 §6, H34).

Tiny, undertrained config so it is fast: checks the ablation emits well-formed stats for both edge
projections with bounded goal-reach and CIs, and the structural invariant the ablation rests on --
the reachability horizon is never *shorter* than the exact-state horizon (the model sustains the
reachability projection at least as long as the bit-for-bit state, EN10 over HS3). Not the
(small-scale) magnitudes -- the headline gap comes from the committed full-config run.
"""

from __future__ import annotations

from verisim.experiments.lp4 import LP4Config, run_lp4


def _tiny() -> LP4Config:
    return LP4Config(
        n_hosts=4,
        n_ports=2,
        train_seeds=(0,),
        train_steps_per_traj=14,
        graph_d_model=24,
        graph_iters=50,
        eval_difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        hop_length=2,
        goal_distances=(2, 4, 6),
    )


def test_run_lp4_smoke() -> None:
    stats = run_lp4(_tiny())
    assert {s.edge_metric for s in stats} == {"reachability", "exact"}
    for s in stats:
        assert s.n > 0
        assert 0.0 <= s.goal_reach <= 1.0
        assert s.gr_lo <= s.goal_reach <= s.gr_hi
        assert s.horizon >= 0.0
    # The reachability projection is sustained at least as long as the exact-state projection
    # (EN10 over HS3): per goal distance, reach horizon >= exact horizon.
    reach = {s.goal_distance: s.horizon for s in stats if s.edge_metric == "reachability"}
    exact = {s.goal_distance: s.horizon for s in stats if s.edge_metric == "exact"}
    for g in reach:
        assert reach[g] >= exact[g]


def test_run_lp4_is_deterministic() -> None:
    a = {(s.edge_metric, s.goal_distance): s.goal_reach for s in run_lp4(_tiny())}
    b = {(s.edge_metric, s.goal_distance): s.goal_reach for s in run_lp4(_tiny())}
    assert a == b
