"""Smoke + determinism test for the EN8 scale runner (SPEC-8 §7.1, OG5).

Tiny grid so it is fast: it checks the sweep wires together, emits the pre-registered gap metrics
with well-formed CIs, and is reproducible from seeds on CPU (the deterministic gate, §7.3) -- not
the (small-scale) numbers themselves.
"""

from __future__ import annotations

import dataclasses

from verisim.experiments.en8_scale import (
    COLLAPSE_GAP,
    GAP_METRICS,
    RESIDUAL_GAP,
    EN8ScaleConfig,
    run_en8_scale,
)
from verisim.experiments.scale_common import ModelSize


def _tiny() -> EN8ScaleConfig:
    return EN8ScaleConfig(
        world_sizes=(5,),
        model_sizes=(ModelSize("d16-mp1", 16, 1),),
        seeds=(0, 1),
        train_seeds=(0,),
        eval_seeds=(100,),
        train_steps_per_traj=12,
        eval_steps=8,
        decoder_iters=40,
        jepa_iters=40,
    )


def test_run_en8_scale_smoke() -> None:
    stats = run_en8_scale(_tiny())
    by_metric = {s.metric for s in stats}
    for m in GAP_METRICS:
        assert m in by_metric
    for s in stats:
        assert s.n == 2  # two model seeds aggregated
        assert s.ci_lo <= s.mean <= s.ci_hi


def test_run_en8_scale_is_deterministic() -> None:
    a = {(s.world_size, s.model_label, s.metric): s.mean for s in run_en8_scale(_tiny())}
    b = {(s.world_size, s.model_label, s.metric): s.mean for s in run_en8_scale(_tiny())}
    assert a == b


def test_collapse_only_skips_residual_axis() -> None:
    """``collapse_only`` (the hero/large-N path) emits the collapse gap, skips the residual axis."""
    stats = run_en8_scale(dataclasses.replace(_tiny(), collapse_only=True))
    by_metric = {s.metric for s in stats}
    for m in COLLAPSE_GAP:
        assert m in by_metric
    for m in RESIDUAL_GAP:
        assert m not in by_metric
