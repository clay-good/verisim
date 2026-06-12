"""SPEC-21 CP0/CP4/CP5 scale-law harness tests.

The contract: the frontier reducer + forecast + verdicts are correct (CS1/CS2/H87-H89 logic), the
GPU-readiness gate validates good configs and rejects bad ones with a cost estimate (CP5), the
committed GPU config parses, and the smoke ladder runs the whole pipeline end-to-end (CP0).
"""

from __future__ import annotations

import pytest

from verisim.cue.tasks import TaskGap
from verisim.experiments.horizon_scaling import ModelScale
from verisim.experiments.scale_law import (
    GPU_LADDER,
    ScaleLawConfig,
    ScaleLawResult,
    ScaleRung,
    config_from_json,
    dry_run,
    estimate_cost,
    forecast_check,
    frontier_verdict,
    knee_trajectory,
    knee_verdict,
    load_bearing_frontier,
)


def _rung(label: str, params: int, gaps: list[float], drifts: list[float]) -> ScaleRung:
    names = ["process-control", "fd-control", "file-integrity", "content-value"]
    dims = ["procs", "fds", "fs", "fs"]
    task_gaps = [
        TaskGap(n, d, o, 1.0, 1.0 - g, g, 16)
        for n, d, o, g in zip(names, dims, range(4), gaps, strict=True)
    ]
    return ScaleRung(
        label=label, params=params,
        dimension_drift={"proc_drift": 0.0, "fd_drift": 0.1, "fs_drift": 0.3},
        gaps=task_gaps, keyed_drift=dict(zip(names, drifts, strict=True)), knees={},
    )


def _receding_result() -> ScaleLawResult:
    # the frontier recedes structural-first: fd-control loses its gap as scale grows; content stays
    return ScaleLawResult(rungs=[
        _rung("xs", 1024, [0.00, 0.20, 0.40, 0.55], [0.01, 0.22, 0.42, 0.58]),
        _rung("s", 4096, [0.00, 0.08, 0.30, 0.52], [0.01, 0.10, 0.33, 0.55]),
        _rung("l", 110592, [0.00, 0.02, 0.18, 0.50], [0.00, 0.03, 0.20, 0.52]),
    ])


def test_load_bearing_frontier_recedes():
    frontier = load_bearing_frontier(_receding_result(), threshold=0.05)
    # at xs all of fd/file/content load-bearing (order 3); by l, fd-control drops below threshold
    assert frontier[0]["frontier_order"] == 3
    assert "fd-control" in frontier[0]["load_bearing"]
    assert "fd-control" not in frontier[-1]["load_bearing"]
    assert frontier[-1]["frontier_order"] == 3  # content-value still load-bearing


def test_frontier_verdict_h87_h88():
    v = frontier_verdict(_receding_result(), threshold=0.05)
    assert v["frontier_recedes_or_flat"]  # H87: non-increasing frontier order
    assert v["deepest_task"] == "content-value"
    assert v["irreducible_residue"]  # H88: deepest gap > threshold at every rung
    assert all(g > 0.05 for g in v["deepest_gap_by_scale"].values())


def test_forecast_check_h89():
    fc = forecast_check(_receding_result())
    assert fc["n_cells"] == 12
    assert fc["forecastable"]  # cheap keyed drift strongly orders the gaps
    assert fc["spearman"] > 0.6


def _knee_result() -> ScaleLawResult:
    # content-value (order 3) load-bearing with a RISING knee; file-integrity flat; process not LB
    rungs = []
    for label, params, knees in (
        ("xs", 1024, {"process-control": 0.3, "file-integrity": 0.2, "content-value": 0.3}),
        ("l", 110592, {"file-integrity": 0.2, "content-value": 0.5}),
    ):
        r = _receding_result().rungs[0] if label == "xs" else _receding_result().rungs[-1]
        rungs.append(ScaleRung(label, params, r.dimension_drift, r.gaps, r.keyed_drift, knees))
    return ScaleLawResult(rungs=rungs)


def test_knee_trajectory_and_verdict():
    result = _knee_result()
    traj = knee_trajectory(result, threshold=0.05)
    # only tasks that are load-bearing (gap > threshold) AND have a knee appear
    assert "content-value" in traj and "file-integrity" in traj
    assert traj["content-value"] == [(1024, 0.3), (110592, 0.5)]  # sorted by capacity
    v = knee_verdict(result, threshold=0.05)
    # the deepest load-bearing task's knee rises with scale -> the residue is doubly hard
    assert v["deepest_load_bearing"] == "content-value"
    assert v["knee_trend"] == "rising"
    assert v["knee_delta"] == pytest.approx(0.2)


def test_estimate_cost_monotone_in_ladder():
    cost = estimate_cost(ScaleLawConfig(scales=GPU_LADDER))
    assert cost["n_rungs"] == len(GPU_LADDER)
    assert cost["estimated_gpu_hours"] > 0.0
    # the largest rung dominates the param-step budget
    per = cost["per_rung"]
    assert per["xxxl"]["param_steps"] > per["xs"]["param_steps"]


def test_dry_run_accepts_gpu_config_and_rejects_bad():
    ok = dry_run(ScaleLawConfig(scales=GPU_LADDER, device="cuda"))
    assert ok["ok"] and not ok["issues"]
    # a non-monotone ladder + bad device + indivisible head is caught
    bad = ScaleLawConfig(
        scales=(ModelScale("big", 128, 4, train_steps=10),
                ModelScale("small", 32, 1, train_steps=10),
                ModelScale("odd", 30, 1, n_head=4, train_steps=10)),
        device="quantum",
    )
    report = dry_run(bad)
    assert not report["ok"]
    assert any("monotone" in i for i in report["issues"])
    assert any("device" in i for i in report["issues"])
    assert any("divisible" in i for i in report["issues"])


def test_committed_gpu_config_parses():
    config = config_from_json("configs/scale_law_gpu.json")
    assert config.device == "cuda"
    assert [s.label for s in config.scales] == ["xs", "s", "m", "l", "xl", "xxl", "xxxl"]
    assert dry_run(config)["ok"]


# --- torch-gated: the smoke ladder end-to-end (CP0) ----------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.scale_law import run_scale_law  # noqa: E402


def test_smoke_ladder_runs_end_to_end():
    result = run_scale_law(ScaleLawConfig.smoke())
    assert len(result.rungs) == 2
    for rung in result.rungs:
        assert len(rung.gaps) == 4
        # faithful predictor exact on every task; the structural task is drift-robust
        for g in rung.gaps:
            assert g.faithful == pytest.approx(1.0)
        proc = next(g for g in rung.gaps if g.task == "process-control")
        assert proc.gap == pytest.approx(0.0, abs=0.05)
    # the apparatus computes all three verdicts without error
    assert "frontier_recedes_or_flat" in frontier_verdict(result)
    assert "forecastable" in forecast_check(result)
