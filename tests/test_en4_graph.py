"""Smoke test for the EN4 graph-vs-flat comparison harness (SPEC-5 §12, H11).

Runs a tiny instance end-to-end (both arms trained on the same data, scored with the same eval
primitives) and checks the result shape. Not a science run -- just proof the comparison apparatus
is wired correctly and the graph arm drops into EN1's exact eval path.
"""

from __future__ import annotations

from verisim.experiments.en4_graph import EN4Config, run_en4_graph


def test_en4_comparison_runs_and_returns_all_arms() -> None:
    cfg = EN4Config(
        train_seeds=(0,),
        train_steps_per_traj=6,
        eval_seeds=(100,),
        eval_steps=6,
        difficulties=(("low", "weighted"),),
        epsilons=(0.0, 0.1),
        d_model=16,
        mp_rounds=1,
        graph_iters=10,
        graph_noise_prob=0.3,
    )
    results = run_en4_graph(cfg)
    assert set(results) == {"flat", "graph", "graph+noise", "graph+selfforce"}
    for arm in results:
        row = results[arm]
        assert 0.0 <= row["onestep_acc"] <= 1.0
        assert 0.0 <= row["delta_exact"] <= 1.0  # the per-step free-decode exact-match rate
        assert "h@0.0" in row and "h@0.1" in row
        assert row["h@0.1"] >= row["h@0.0"]  # looser tolerance never shortens the horizon
