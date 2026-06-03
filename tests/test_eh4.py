"""EH4 factored-vs-flat comparison tests (SPEC-6 §6.1, DD-H1; the H13 follow-up).

Exercises the full comparison on a tiny config: train both arms on identical data, score each with
the same eval primitives (delta-exact + the composition law), and check the result is well-formed
and deterministic. Whatever the verdict is at this scale is a datum, not asserted -- the test pins
the *apparatus*, not the (smoke-scale) outcome.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config
from verisim.experiments.eh4 import EH4Config, run_eh4


def _tiny_config() -> EH4Config:
    base = EH1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=80,
        n_layer=1,
        n_embd=32,
        block_size=160,
        difficulties={"low": "forky"},
        eval_seeds=(100, 101),
        eval_steps=10,
        epsilons=(0.0, 0.1),
    )
    return EH4Config(base=base, max_pid=32, graph_iters=80, graph_d_model=24, graph_batch=16)


def test_run_eh4_scores_both_arms():
    results = run_eh4(_tiny_config())
    assert set(results) == {"flat", "factored"}
    for arm in ("flat", "factored"):
        r = results[arm]
        assert 0.0 <= r["delta_exact"] <= 1.0
        # composed acceptance <= the weakest link (one bad subsystem fails the whole step)
        assert 0.0 <= r["composed"] <= r["weakest_link"] + 1e-9
        assert r["verdict"] in {"multiplicative", "weakest_link", "coupled"}
        assert set(r["subsystem_acceptance"]) == {"proc", "fd", "fs", "global"}


def test_run_eh4_is_deterministic():
    a = run_eh4(_tiny_config())
    b = run_eh4(_tiny_config())
    for arm in ("flat", "factored"):
        assert a[arm]["delta_exact"] == b[arm]["delta_exact"]
        assert a[arm]["composed"] == b[arm]["composed"]
