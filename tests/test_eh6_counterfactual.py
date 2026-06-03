"""EH6 counterfactual-grounding harness tests (SPEC-6 §2.8, H16). Torch-gated.

Tiny config; the finding's *outcome* (does counterfactual replay lift intervention fidelity?) is
read off the committed figure, not asserted -- the test pins the apparatus: every (arm, metric) cell
present, the example-count budget matched across arms, metrics bounded, deterministic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh6_counterfactual import (
    ARMS,
    METRICS,
    EH6CFConfig,
    run_eh6_counterfactual,
)


def _tiny_config() -> EH6CFConfig:
    return EH6CFConfig(
        train_seeds=(0, 1), train_steps_per_traj=12, k_counterfactual=3,
        max_pid=32, graph_d_model=24, graph_mp_rounds=2, graph_iters=60, graph_batch=16,
        eval_seeds=(100, 101), eval_steps=10, m_interventions=4,
    )


def test_run_covers_arms_and_metrics():
    stats = run_eh6_counterfactual(_tiny_config())
    cells = {(s.arm, s.metric) for s in stats}
    assert cells == {(arm, metric) for arm in ARMS for metric in METRICS}
    for s in stats:
        assert 0.0 <= s.mean <= 1.0
        assert s.ci_lo <= s.mean <= s.ci_hi
        assert s.n == 2  # one value per eval seed


def test_run_is_deterministic():
    a = run_eh6_counterfactual(_tiny_config())
    b = run_eh6_counterfactual(_tiny_config())
    assert [s.csv_row() for s in a] == [s.csv_row() for s in b]
