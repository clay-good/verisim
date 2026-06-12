"""CS3 -- the scale law survives the system oracle (SPEC-21 §6, H90).

The smoke instance of the apparatus: a tiny seeded α-ladder asserting (a) the apparatus is correct
and deterministic against the reference oracle (no shell needed -- the structure->content gradient,
the content-gap recession, the cheap-drift forecast), and (b) the platform-independent headline --
the load-bearing gap is **anchor-invariant** when a real ``/bin/sh`` replaces the reference oracle
(``gap_sys == gap_ref`` bit-for-bit on the validated content grammar).
"""

from __future__ import annotations

import random

import pytest

from verisim.data.drivers import Driver
from verisim.env.config import DEFAULT_CONFIG
from verisim.env.state import State
from verisim.experiments.cs3_system_anchor import (
    ANCHOR_TASKS,
    ContentDriftProposer,
    CS3Config,
    content_set,
    cs3_verdict,
    run_cs3,
    which_file_set,
    write_csv,
)
from verisim.oracle.reference import ReferenceOracle
from verisim.oracle.sandbox import SandboxOracle, SystemOracleUnavailable

try:
    SandboxOracle()
    _HAVE_SHELL = True
except SystemOracleUnavailable:  # pragma: no cover
    _HAVE_SHELL = False

requires_shell = pytest.mark.skipif(not _HAVE_SHELL, reason="no real shell")


def _tiny() -> CS3Config:
    return CS3Config(alphas=(0.3, 0.5, 0.7, 0.9), horizon=10, seeds=tuple(range(800, 806)))


# --- apparatus correctness (reference-only; no shell required) ------------------------------------


def test_suite_is_ordered_structure_to_content():
    orders = [t.order for t in ANCHOR_TASKS]
    assert orders == sorted(orders)
    assert ANCHOR_TASKS[0].name == "file-integrity"  # structure
    assert ANCHOR_TASKS[-1].name == "content-value"  # deep content


def test_content_drift_proposer_keeps_structure_corrupts_content():
    """The capacity-proxy stand-in is faithful on *which* files but drifts on *content* (α=0)."""
    ref = ReferenceOracle()
    drv = Driver("structural", DEFAULT_CONFIG, random.Random(1))
    start = State.empty()
    actions = []
    s = start
    for _ in range(14):
        a = drv.sample(s)
        actions.append(a)
        s = ref.step(s, a).state
    true_files = which_file_set(s)

    proposer = ContentDriftProposer(alpha=0.0, seed=1)  # always drifts
    ds = start
    for a in actions:
        ds = proposer.step(ds, a)
    # structure faithful: every true file path is present in the drifted state
    assert which_file_set(ds) == true_files
    # content drifted: not a single (path, content) pair matches the truth
    assert content_set(ds).isdisjoint(content_set(s))


def test_perfect_capacity_with_no_residue_has_zero_gap():
    """α=1.0 with no irreducible floor leaves no gap -- the proposer reduces to the oracle."""
    cfg = CS3Config(alphas=(1.0,), horizon=12, seeds=tuple(range(800, 806)), irreducible=0.0)
    ref = ReferenceOracle()
    result = run_cs3(cfg, sys_oracle=ref)  # inject reference as the "anchor" -> no shell needed
    for c in result.cells:
        assert c.gap_ref == pytest.approx(0.0, abs=1e-9)


def test_irreducible_residue_survives_perfect_capacity():
    """With an irreducible floor, the content gap stays load-bearing even at α=1.0 (H88)."""
    cfg = CS3Config(alphas=(1.0,), horizon=14, seeds=tuple(range(800, 812)), irreducible=0.2)
    ref = ReferenceOracle()
    by = {c.task: c for c in run_cs3(cfg, sys_oracle=ref).cells}
    assert by["file-integrity"].gap_ref == pytest.approx(0.0, abs=1e-9)  # structure always faithful
    assert by["content-value"].gap_ref > cfg.threshold  # the residue persists at infinite capacity


def test_gradient_and_recession_against_reference():
    """Content gap exceeds structure gap and recedes with capacity (the frontier motion)."""
    cfg = _tiny()
    ref = ReferenceOracle()
    result = run_cs3(cfg, sys_oracle=ref)
    by = {(c.alpha, c.task): c for c in result.cells}
    alphas = sorted({c.alpha for c in result.cells})
    # the structure task is never load-bearing; the content task is, at low capacity
    for a in alphas:
        assert by[(a, "file-integrity")].gap_ref == pytest.approx(0.0, abs=1e-9)
        assert by[(a, "content-value")].gap_ref >= by[(a, "file-integrity")].gap_ref
    content = [by[(a, "content-value")].gap_ref for a in alphas]
    assert all(content[i + 1] <= content[i] + 1e-9 for i in range(len(content) - 1))
    assert content[0] > content[-1]  # it genuinely recedes


def test_determinism():
    cfg = _tiny()
    ref = ReferenceOracle()
    a = run_cs3(cfg, sys_oracle=ref)
    b = run_cs3(cfg, sys_oracle=ref)
    assert [c.gap_ref for c in a.cells] == [c.gap_ref for c in b.cells]


def test_write_csv(tmp_path):
    cfg = CS3Config(alphas=(0.5,), horizon=10, seeds=(800, 801))
    ref = ReferenceOracle()
    result = run_cs3(cfg, sys_oracle=ref)
    path = write_csv(result, tmp_path / "cs3.csv")
    text = path.read_text()
    assert text.startswith("alpha,task,order,gap_ref,gap_sys,anchor_delta")
    assert "content-value" in text


def test_config_round_trips():
    d = {"driver": "structural", "alphas": [0.4, 0.8], "horizon": 9, "seeds": [1, 2, 3]}
    cfg = CS3Config.from_dict(d)
    assert cfg.alphas == (0.4, 0.8)
    assert cfg.horizon == 9
    assert cfg.seeds == (1, 2, 3)


def test_unavailable_result_is_disclosed_not_counted():
    """An unavailable result (no shell) gates the verdict to available=False (§2.5: not counted)."""
    from verisim.experiments.cs3_system_anchor import CS3Result

    unavailable = CS3Result(available=False, platform="nowhere", cells=[])
    assert cs3_verdict(unavailable) == {"available": False}


# --- the H90 headline: anchor-invariance against a real /bin/sh -----------------------------------


@requires_shell
def test_h90_load_bearing_gap_is_anchor_invariant():
    """The load-bearing gap does not move when the real kernel replaces the reference oracle."""
    result = run_cs3(_tiny())
    assert result.available
    verdict = cs3_verdict(result, _tiny())
    # bit-exact on the validated content grammar (SY1/H27): the anchors produce identical keyed sets
    assert verdict["max_anchor_delta"] == pytest.approx(0.0, abs=1e-9)
    assert verdict["anchor_invariant"]
    for c in result.cells:
        assert c.gap_sys == pytest.approx(c.gap_ref, abs=1e-9)


@requires_shell
def test_h90_frontier_properties_under_real_kernel():
    """Gradient, recession, residue, and the H89 forecast all hold against the real shell."""
    cfg = _tiny()
    verdict = cs3_verdict(run_cs3(cfg), cfg)
    assert verdict["gradient_holds"]  # content gap >= structure gap, real kernel
    assert verdict["content_gap_recedes"]  # the frontier motion survives the real kernel
    assert verdict["residue_under_real_kernel"]  # the deep-content residue is load-bearing (H88)
    assert verdict["forecastable"]  # cheap drift forecasts the gap under the real shell (H89)
    assert verdict["forecast_spearman"] > 0.6
