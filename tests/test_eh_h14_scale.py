"""EH-H14-scale tests (concurrency cost vs thread count, SPEC-6 §3.4).

Exercises the thread-count sweep on a tiny config: every (thread-count, interleave) cell is produced
at the configured ε, realized entropy is monotone in the knob within each width, and the run is
deterministic. The scaling *outcome* (does the collapse steepen with width?) is a datum read off the
committed figure, not asserted.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh_h14 import EHH14Config
from verisim.experiments.eh_h14_scale import EHH14ScaleConfig, run_eh_h14_scale


def _tiny_config() -> EHH14ScaleConfig:
    base = EHH14Config(
        n_layer=1, n_embd=24, mp_rounds=2, max_pid=32, graph_iters=60, graph_batch=16,
        train_workload_seeds=(0, 1), eval_workload_seeds=(10,),
        eval_interleaves=(0.0, 1.0), eval_chaos_seeds=(0, 1), epsilons=(0.0,),
    )
    return EHH14ScaleConfig(thread_counts=(2, 4), epsilon=0.0, base=base)


def test_run_eh_h14_scale_covers_every_cell():
    config = _tiny_config()
    points = run_eh_h14_scale(config)
    # thread_counts(2) x eval_interleaves(2)
    assert len(points) == 2 * 2
    assert {p.n_threads for p in points} == {2, 4}
    for p in points:
        assert 0.0 <= p.mean_entropy <= 1.0
        assert p.mean_h >= 0.0


def test_entropy_monotone_in_the_knob_per_width():
    points = run_eh_h14_scale(_tiny_config())
    for n in (2, 4):
        cells = sorted((p for p in points if p.n_threads == n), key=lambda p: p.interleave)
        assert cells[0].mean_entropy <= cells[-1].mean_entropy  # chaos >= sequential


def test_run_eh_h14_scale_is_deterministic():
    a = run_eh_h14_scale(_tiny_config())
    b = run_eh_h14_scale(_tiny_config())
    assert [p.csv_row() for p in a] == [p.csv_row() for p in b]
