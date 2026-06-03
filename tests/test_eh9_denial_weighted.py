"""EH9 denial-weighted-objective harness tests (SPEC-6 §9.4). Torch-gated.

Tiny config; the finding's *outcome* (does recall lift?) is read off the committed figure, not
asserted -- the test pins the apparatus: every (arm, oversample) cell present, metrics bounded,
deterministic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config
from verisim.experiments.eh9_denial_weighted import EH9Config, run_eh9


def _tiny_config() -> EH9Config:
    base = EH1Config(
        train_seeds=(0, 1), train_steps_per_traj=16, train_iters=60,
        n_layer=1, n_embd=32, block_size=160, difficulties={"high": "adversarial"},
        eval_seeds=(100, 101), eval_steps=12, epsilons=(0.0, 0.1),
    )
    return EH9Config(
        base=base, train_drivers=("adversarial",), denial_oversamples=(1, 8),
        max_pid=32, graph_d_model=24, graph_mp_rounds=2, graph_iters=60, graph_batch=16,
    )


def test_run_eh9_covers_arms_and_oversamples():
    config = _tiny_config()
    results = run_eh9(config)
    assert set(results) == {"flat@1", "flat@8", "factored@1", "factored@8"}
    for r in results.values():
        for k in ("denied_recall", "allowed_specificity", "privilege_faithfulness"):
            assert 0.0 <= r[k] <= 1.0
        assert r["n_denied"] > 0  # the adversarial eval set actually exercises denials


def test_run_eh9_is_deterministic():
    a = run_eh9(_tiny_config())
    b = run_eh9(_tiny_config())
    assert a == b
