"""Smoke + invariant test for LP3 goal reach (SPEC-12 §6, H33).

Tiny, undertrained config so it is fast: checks both sweeps emit well-formed stats with bounded
goal-reach and CIs, that the flat baseline spends zero budget while landmark spends ``ρ = 1/L > 0``
once there is an intermediate hop, and that the run is deterministic. Not the (small-scale)
magnitudes -- the headline number comes from the committed full-config run.
"""

from __future__ import annotations

from verisim.experiments.lp3 import LP3Config, run_lp3


def _tiny() -> LP3Config:
    return LP3Config(
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
        budget_hop_lengths=(2, 3),
        budget_goal_distance=6,
    )


def test_run_lp3_smoke() -> None:
    stats = run_lp3(_tiny())
    assert {s.sweep for s in stats} == {"distance", "budget"}
    assert {s.arm for s in stats} == {"flat", "landmark"}
    for s in stats:
        assert s.n > 0
        assert 0.0 <= s.goal_reach <= 1.0
        assert s.gr_lo <= s.goal_reach <= s.gr_hi
        assert s.rho >= 0.0
    # The flat arm never consults; landmark spends budget once a hop has an interior boundary.
    flat = [s for s in stats if s.arm == "flat"]
    assert all(s.rho == 0.0 for s in flat)
    far_landmark = max(
        (s for s in stats if s.sweep == "distance" and s.arm == "landmark"),
        key=lambda s: s.x_value,
    )
    assert far_landmark.rho > 0.0


def test_run_lp3_is_deterministic() -> None:
    a = {(s.sweep, s.arm, s.x_value): s.goal_reach for s in run_lp3(_tiny())}
    b = {(s.sweep, s.arm, s.x_value): s.goal_reach for s in run_lp3(_tiny())}
    assert a == b
