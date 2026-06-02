"""Smoke + determinism test for the EN9 scale runner (SPEC-8 §7.1, OG5).

Tiny grid so it is fast: checks the sweep emits the interventional-lift gap metrics with well-formed
CIs and is reproducible from seeds on CPU — not the (small-scale) numbers themselves.
"""

from __future__ import annotations

from verisim.experiments.en9_scale import GAP_METRICS, EN9ScaleConfig, run_en9_scale
from verisim.experiments.scale_common import ModelSize


def _tiny() -> EN9ScaleConfig:
    return EN9ScaleConfig(
        world_sizes=(5,),
        model_sizes=(ModelSize("d16-mp1", 16, 1),),
        seeds=(0, 1),
        train_seeds=(0,),
        eval_seeds=(100,),
        train_steps_per_traj=12,
        eval_steps=8,
        k_negatives=4,
        contrastive_iters=40,
    )


def test_run_en9_scale_smoke() -> None:
    stats = run_en9_scale(_tiny())
    by_metric = {s.metric for s in stats}
    for m in GAP_METRICS:
        assert m in by_metric
    for s in stats:
        assert s.n == 2
        assert s.ci_lo <= s.mean <= s.ci_hi


def test_run_en9_scale_is_deterministic() -> None:
    a = {(s.world_size, s.model_label, s.metric): s.mean for s in run_en9_scale(_tiny())}
    b = {(s.world_size, s.model_label, s.metric): s.mean for s in run_en9_scale(_tiny())}
    assert a == b
