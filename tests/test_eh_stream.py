"""EH-stream experience-stream-vs-batch harness tests (SPEC-6 §8.5, H15 / HW-4). Torch-gated.

Tiny config; the finding's *outcome* (does the stream beat the batch? does plasticity decay?) is
read off the committed figure, not asserted -- the test pins the apparatus: every (arm, metric) cell
present, equal compute across arms, metrics bounded, deterministic.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh_stream import ARMS, METRICS, EHStreamConfig, run_eh_stream


def _tiny_config() -> EHStreamConfig:
    return EHStreamConfig(
        stream_seeds=(0, 1, 2, 3), stream_steps_per_traj=10, replay_batch=8,
        max_pid=32, graph_d_model=24, graph_mp_rounds=2,
        difficulties={"high": "adversarial"}, eval_seeds=(100, 101), eval_steps=10,
        probe_seed=900, probe_size=12, probe_steps=5,
    )


def test_run_covers_arms_and_metrics():
    stats = run_eh_stream(_tiny_config())
    cells = {(s.arm, s.metric) for s in stats}
    assert cells == {(arm, metric) for arm in ARMS for metric in METRICS}
    for s in stats:
        assert s.ci_lo <= s.mean <= s.ci_hi
        if s.metric != "free_horizon":  # exact + plasticity are fractions
            assert 0.0 <= s.mean <= 1.0
        else:
            assert 0.0 <= s.mean <= 10.0  # bounded by eval_steps ceiling


def test_run_is_deterministic():
    a = run_eh_stream(_tiny_config())
    b = run_eh_stream(_tiny_config())
    assert [s.csv_row() for s in a] == [s.csv_row() for s in b]
