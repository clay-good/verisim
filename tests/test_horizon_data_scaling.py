"""HS1.2 data-cross-axis tests (SPEC-10 §5).

The harness fixes capacity and sweeps the coverage-set size; these cover that the sweep is
well-formed (every metric present, the data x-axis strictly increasing, CIs bracket the mean) and
deterministic. The smoke instance is deliberately tiny (one capacity, two data points, two model
seeds, few iters) so CI stays fast; the committed figure comes from the local sweep.
"""

from __future__ import annotations

import pytest

from verisim.experiments.horizon_scaling import METRICS

torch = pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config  # noqa: E402
from verisim.experiments.horizon_data_scaling import run_data_scaling  # noqa: E402
from verisim.experiments.horizon_scaling import (  # noqa: E402
    HorizonScalingConfig,
    ModelScale,
)


def _tiny_config() -> HorizonScalingConfig:
    base = EN1Config(train_steps_per_traj=12)
    return HorizonScalingConfig(
        base=base,
        scales=(ModelScale("xs", n_embd=32, n_layer=1, train_steps=60),),
        seeds=(0, 1),
        block_size=64,
        batch_size=16,
        num_threads=1,  # bit-deterministic, so the determinism test holds
        eval_seeds=(100, 101),
        eval_steps=12,
        one_step_seeds=(200,),
        one_step_steps=16,
    )


def test_run_data_scaling_is_well_formed():
    stats = run_data_scaling(_tiny_config(), data_seeds=(2, 4))
    # the data x-axis (params field carries n_transitions) strictly increases
    n_train = sorted({(s.params, s.scale) for s in stats})
    assert n_train[0][0] < n_train[-1][0]
    assert {s.scale for s in stats} == {"2", "4"}
    assert {s.metric for s in stats} == set(METRICS)
    for s in stats:
        assert s.n == 2
        assert s.ci_lo <= s.mean <= s.ci_hi or s.n == 1
        if s.metric.startswith("one_step_acc"):
            assert 0.0 <= s.mean <= 1.0
        else:
            assert s.mean >= 0.0


def test_run_data_scaling_rejects_multiple_capacities():
    cfg = _tiny_config()
    bad = HorizonScalingConfig(
        base=cfg.base,
        scales=(*cfg.scales, ModelScale("s", n_embd=64, n_layer=2, train_steps=60)),
        seeds=cfg.seeds,
    )
    with pytest.raises(ValueError, match="exactly one scale"):
        run_data_scaling(bad, data_seeds=(2,))


def test_run_data_scaling_is_deterministic():
    a = run_data_scaling(_tiny_config(), data_seeds=(2, 4))
    b = run_data_scaling(_tiny_config(), data_seeds=(2, 4))
    assert [(s.scale, s.metric, s.mean) for s in a] == [(s.scale, s.metric, s.mean) for s in b]
