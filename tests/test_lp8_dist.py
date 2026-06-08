"""Smoke + invariant test for LP8-dist cross-world goal reach (SPEC-12 §6, H38).

Tiny, undertrained config so it is fast: checks both sweeps emit well-formed stats with bounded
goal-reach and CIs, the flat baseline spends zero budget while landmark spends ρ > 0 once there is
an
interior boundary, the consistency-graph soundness fields are well-formed (verified residual exactly
0.0 — zero false paths by construction), and that the run is deterministic. Not the (small-scale)
magnitudes — the headline transfer number comes from the committed full-config run.
"""

from __future__ import annotations

from verisim.experiments.lp8_dist import LP8DistConfig, run_lp8_dist


def _tiny() -> LP8DistConfig:
    return LP8DistConfig(
        train_seeds=(0,),
        train_steps_per_traj=14,
        train_iters=40,
        n_embd=32,
        eval_difficulties={"low": "contention"},
        eval_seeds=(100, 101),
        hop_length=2,
        goal_distances=(2, 4, 6),
        budget_hop_lengths=(2, 3),
        budget_goal_distance=6,
    )


def test_run_lp8_dist_smoke() -> None:
    stats, soundness = run_lp8_dist(_tiny())
    assert {s.sweep for s in stats} == {"distance", "budget"}
    assert {s.arm for s in stats} == {"flat", "landmark"}
    for s in stats:
        assert s.n > 0
        assert 0.0 <= s.goal_reach <= 1.0
        assert s.gr_lo <= s.goal_reach <= s.gr_hi
        assert s.rho >= 0.0
    flat = [s for s in stats if s.arm == "flat"]
    assert all(s.rho == 0.0 for s in flat)
    far_landmark = max(
        (s for s in stats if s.sweep == "distance" and s.arm == "landmark"),
        key=lambda s: s.x_value,
    )
    assert far_landmark.rho > 0.0
    # Zero-false-paths guarantee in consistency space (the LP2 analogue), by construction.
    assert soundness["verified_residual_false_rate"] == 0.0
    assert 0.0 <= soundness["false_edge_rate"] <= 1.0
    assert soundness["consult_bits_ratio"] >= 0.0


def test_run_lp8_dist_is_deterministic() -> None:
    a = {(s.sweep, s.arm, s.x_value): s.goal_reach for s in run_lp8_dist(_tiny())[0]}
    b = {(s.sweep, s.arm, s.x_value): s.goal_reach for s in run_lp8_dist(_tiny())[0]}
    assert a == b
