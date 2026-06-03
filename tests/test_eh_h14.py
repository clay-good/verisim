"""EH-H14 concurrency-dial harness tests (SPEC-6 §3.4, H14, HC7).

Exercises the full H14 pipeline on a tiny config: train on recorded schedules, sweep the chaos dial,
and produce ``H_ε`` vs interleaving-entropy points. The H14 *outcome* (does horizon fall?) is a
datum, not asserted -- the test pins the apparatus: monotone realized entropy across the knob, every
(interleave, ε) cell present, and determinism.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh_h14 import EHH14Config, run_eh_h14


def _tiny_config() -> EHH14Config:
    return EHH14Config(
        n_layer=1, n_embd=24, mp_rounds=2, max_pid=32, graph_iters=80, graph_batch=16,
        n_threads=4, train_workload_seeds=(0, 1, 2),
        eval_workload_seeds=(10,), eval_interleaves=(0.0, 0.5, 1.0), eval_chaos_seeds=(0, 1),
        epsilons=(0.0, 0.1),
    )


def test_run_eh_h14_produces_points_per_cell():
    points = run_eh_h14(_tiny_config())
    # interleaves(3) x epsilons(2)
    assert len(points) == 3 * 2
    for p in points:
        assert 0.0 <= p.mean_entropy <= 1.0
        assert p.mean_h >= 0.0
        assert p.ci_low <= p.mean_h <= p.ci_high or p.n <= 1
        assert p.n == 1 * 2  # eval_workload_seeds(1) x eval_chaos_seeds(2)


def test_realized_entropy_is_monotone_in_the_knob():
    points = run_eh_h14(_tiny_config())
    # the mean realized interleaving entropy rises with the chaos knob (per ε slice, same entropy)
    by_il = {}
    for p in points:
        if p.epsilon == 0.0:
            by_il[p.interleave] = p.mean_entropy
    assert by_il[0.0] <= by_il[0.5] <= by_il[1.0]
    assert by_il[0.0] < by_il[1.0]  # the dial genuinely moves the realized entropy


def test_run_eh_h14_is_deterministic():
    a = run_eh_h14(_tiny_config())
    b = run_eh_h14(_tiny_config())
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]
