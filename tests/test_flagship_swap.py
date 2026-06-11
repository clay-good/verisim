"""FL4 proposer-swap tests (SPEC-19 §4, H72).

The contract: both proposers run the FL1 curve on the same world, the shape signature is
magnitude-free, and the H72 verdict compares shapes. The smoke arms are trivial; the committed swap
comes from the local frontier run -- CI guarantees the apparatus, not the verdict.
"""

from __future__ import annotations

import pytest

from verisim.experiments.flagship_curve import CurvePoint
from verisim.experiments.flagship_swap import h72_verdict, shape_signature


def _curve(floor: float, ceiling: float, interior: float) -> list[CurvePoint]:
    return [
        CurvePoint("floor", 0.0, floor, floor, floor, 2),
        CurvePoint("ceiling", 1.0, ceiling, ceiling, ceiling, 2),
        CurvePoint("composed", 0.2, interior, interior, interior, 2),
        CurvePoint("fixed", 0.2, interior, interior, interior, 2),
    ]


def test_shape_signature_detects_knee():
    # interior reaches ≥80% of ceiling at ρ=0.2 -> knee
    knee = shape_signature(_curve(floor=1, ceiling=10, interior=9))
    assert knee["has_knee"]
    # interior hugs the floor -> no knee (floor+cliff)
    cliff = shape_signature(_curve(floor=1, ceiling=10, interior=1))
    assert not cliff["has_knee"]
    assert cliff["span"] == pytest.approx(9.0)


def test_h72_verdict_same_and_different_shapes():
    # both floor+cliff (no knee) -> same shape -> supported (the loop governs)
    both_cliff = {"flat": _curve(1, 10, 1), "graph": _curve(0, 5, 0)}
    assert h72_verdict(both_cliff)["h72_supported"]
    # one knee, one cliff -> different shape -> refuted
    split = {"flat": _curve(1, 10, 9), "graph": _curve(0, 5, 0)}
    assert not h72_verdict(split)["h72_supported"]


# --- torch-gated: the real two-proposer swap ------------------------------------------------------

torch = pytest.importorskip("torch")

from verisim.experiments.flagship import FlagshipConfig, train_flagship  # noqa: E402
from verisim.experiments.flagship_swap import SwapConfig, run_swap  # noqa: E402
from verisim.netoracle import ReferenceNetworkOracle  # noqa: E402


def test_run_swap_produces_both_curves():
    flat, _ = train_flagship(FlagshipConfig.smoke())  # pass the flat arm directly (no checkpoint)
    curves = run_swap(SwapConfig.smoke(), flat_model=flat, oracle=ReferenceNetworkOracle())
    assert set(curves) == {"flat", "graph"}
    for pts in curves.values():
        arms = {p.arm for p in pts}
        assert {"floor", "ceiling", "fixed", "composed"} <= arms
    verdict = h72_verdict(curves)
    assert isinstance(verdict["h72_supported"], bool)
    assert set(verdict["knee_per_arm"]) == {"flat", "graph"}
