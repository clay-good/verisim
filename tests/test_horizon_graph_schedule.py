"""HS3-T trainer-diagnostic tests: flat-LR vs warmup+cosine for the graph arm (SPEC-10 §4.11).

Covers that the diagnostic trains the fixed-capacity graph arm under each ``warmup_frac`` and gives
well-formed per-arm stats (every metric, CIs), and that the ``warmup_frac=0`` arm reproduces
the default flat-LR HS3 cell (the backward-compatibility guarantee of the opt-in schedule).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config  # noqa: E402
from verisim.experiments.horizon_graph_scaling import (  # noqa: E402
    GraphHorizonScalingConfig,
    GraphScale,
    run_graph_horizon_scaling,
)
from verisim.experiments.horizon_graph_schedule import (  # noqa: E402
    run_graph_schedule_diag,
)
from verisim.experiments.horizon_scaling import METRICS  # noqa: E402


def _tiny_config() -> GraphHorizonScalingConfig:
    base = EN1Config(train_seeds=(0, 1), train_steps_per_traj=8)
    return GraphHorizonScalingConfig(
        base=base,
        scales=(GraphScale("m", d_model=16, n_layer=1, mp_rounds=1, train_steps=20),),
        seeds=(0, 1),
        eval_seeds=(100,),
        eval_steps=6,
        one_step_seeds=(200,),
        one_step_steps=6,
    )


def test_run_graph_schedule_diag_is_well_formed():
    stats = run_graph_schedule_diag(_tiny_config(), warmup_fracs=(0.0, 0.1))
    assert {s.metric for s in stats} == set(METRICS)
    assert {s.scale for s in stats} == {"flat-LR", "scheduled@0.1"}
    for s in stats:
        assert s.ci_lo <= s.mean <= s.ci_hi or s.n == 1
        assert s.n == 2
        if s.metric.startswith("one_step_acc"):
            assert 0.0 <= s.mean <= 1.0
        else:
            assert s.mean >= 0.0


def test_flat_lr_arm_matches_default_hs3_cell():
    """warmup_frac=0 must reproduce the plain HS3 sweep exactly (the opt-in is default-off)."""
    cfg = _tiny_config()
    diag = run_graph_schedule_diag(cfg, warmup_fracs=(0.0,))
    plain = run_graph_horizon_scaling(cfg)
    diag_by = {s.metric: s.mean for s in diag if s.scale == "flat-LR"}
    plain_by = {s.metric: s.mean for s in plain}
    for metric in METRICS:
        assert diag_by[metric] == pytest.approx(plain_by[metric])


def test_run_graph_schedule_diag_rejects_multi_capacity():
    cfg = _tiny_config()
    multi = GraphHorizonScalingConfig(
        base=cfg.base,
        scales=(*cfg.scales, GraphScale("s", d_model=24, n_layer=1, mp_rounds=1, train_steps=20)),
    )
    with pytest.raises(ValueError):
        run_graph_schedule_diag(multi, warmup_fracs=(0.0,))
