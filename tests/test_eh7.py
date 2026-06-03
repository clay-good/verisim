"""EH7 model-invariance tests (SPEC-6, H22; the host analogue of the network EN7).

Exercises the four-proposer sweep on a tiny config and pins the structural invariants H22 rests on
(the *outcome* — is the shape shared? — is a datum read off the committed figure, not asserted):

  - all four proposers (null / flat / factored / oracle) are swept across every ρ;
  - the bracketing baselines bound the plot: at ρ=0 the oracle-backed proposer is perfectly faithful
    (H_ε = T) while the null proposer drifts (H_ε < T); at ρ=1 every proposer is fully corrected;
  - determinism.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config
from verisim.experiments.eh7 import PROPOSERS, EH7Config, run_eh7


def _tiny_config() -> EH7Config:
    base = EH1Config(
        train_seeds=(0, 1), train_steps_per_traj=16, train_iters=80,
        n_layer=1, n_embd=32, block_size=160, difficulties={"low": "forky"},
        eval_seeds=(100, 101), eval_steps=12, epsilons=(0.05,),
    )
    return EH7Config(
        base=base, max_pid=32, graph_d_model=24, graph_mp_rounds=2, graph_iters=80,
        graph_batch=16, rhos=(0.0, 0.5, 1.0), epsilon=0.05,
    )


def test_run_eh7_sweeps_all_proposers_and_rhos():
    config = _tiny_config()
    points = run_eh7(config)
    assert {p.proposer for p in points} == set(PROPOSERS)
    assert len(points) == len(PROPOSERS) * len(config.rhos)
    for p in points:
        assert p.ci_lo <= p.mean <= p.ci_hi or p.n <= 1
        assert p.n == 1 * 2  # difficulties(1) x eval_seeds(2)


def test_baselines_bracket_the_plot():
    config = _tiny_config()
    points = {(p.proposer, p.rho): p for p in run_eh7(config)}
    t = config.base.eval_steps
    # ρ=0: the oracle-backed proposer never drifts (H_ε = T); the null proposer does (H_ε < T)
    assert points[("oracle", 0.0)].mean == t
    assert points[("null", 0.0)].mean < t
    # ρ=1: full consultation every step -> every proposer is corrected to truth (H_ε = T)
    for name in PROPOSERS:
        assert points[(name, 1.0)].mean == t


def test_run_eh7_is_deterministic():
    a = run_eh7(_tiny_config())
    b = run_eh7(_tiny_config())
    assert [p.csv_row() for p in a] == [p.csv_row() for p in b]
