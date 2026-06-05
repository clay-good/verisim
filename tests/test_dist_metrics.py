"""DS3 — the distributed metric core: divergence, consistency-faithfulness, bits (SPEC-7 §9).

Tests the dependency-free, GPU-free metric layer:

  - ``divergence`` is ``0`` iff identical, symmetric, in ``[0, 1]``, and feeds ``faithful_horizon``;
  - ``consistency_faithfulness`` (the headline-new §9.1 metric) is ``1.0`` iff the per-object
    converged/split structure matches — and correctly catches a model that mispredicts a partition
    split as converged;
  - ``bits_to_correct`` / ``delta_exact`` gate the predicted ``DistDelta`` against truth.
"""

from __future__ import annotations

from verisim.dist import DistConfig, DistributedState, parse_dist_action
from verisim.distmetrics import (
    bits_to_correct,
    consistency_faithfulness,
    delta_exact,
    delta_exact_rate,
    divergence,
    object_consistency_view,
)
from verisim.distoracle import ReferenceDistOracle
from verisim.metrics.horizon import faithful_horizon

CFG = DistConfig()
ORACLE = ReferenceDistOracle(CFG)


def _state(cmds: list[str]) -> DistributedState:
    state = DistributedState.initial(CFG)
    for cmd in cmds:
        state = ORACLE.step(state, parse_dist_action(cmd)).state
    return state


# --- divergence ----------------------------------------------------------------------------------

def test_divergence_zero_iff_identical_and_symmetric():
    a = _state(["put n0 x b", "advance 2"])
    assert divergence(a, a) == 0.0
    b = _state(["put n0 x c", "advance 2"])
    assert 0.0 < divergence(a, b) <= 1.0
    assert divergence(a, b) == divergence(b, a)  # symmetric


def test_divergence_excludes_the_audit_log():
    # two states with identical live cluster but different log lengths (an extra get logs an event)
    a = _state(["put n0 x b", "advance 2"])
    b = _state(["put n0 x b", "get n0 x", "advance 2"])
    # a `get` changes only the log (+ last_result), not replicas/inflight/partition/clock
    assert divergence(a, b) == 0.0  # the log is the audit trail, excluded from divergence


def test_divergence_feeds_faithful_horizon():
    truth = DistributedState.initial(CFG)
    drifted = _state(["put n0 x b"])  # a state that diverges from boot
    divs = [divergence(truth, truth), divergence(truth, truth), divergence(truth, drifted)]
    assert faithful_horizon(divs, 0.0) == 2  # first two faithful, third diverges


# --- consistency-faithfulness (the headline-new metric) ------------------------------------------

def test_consistency_faithfulness_identical_is_one():
    a = _state(["put n0 x b", "advance 2"])
    assert consistency_faithfulness(a, a) == 1.0


def test_consistency_view_captures_split_then_convergence():
    # before advance, x is split: n0 has (1,b), n1/n2 still (0,nil)
    split = _state(["put n0 x b"])
    assert object_consistency_view(split, "x") == frozenset({(1, "b"), (0, "nil")})
    # after advance, x converges to a single (version, value)
    converged = _state(["put n0 x b", "advance 2"])
    assert object_consistency_view(converged, "x") == frozenset({(1, "b")})


def test_consistency_faithfulness_catches_mispredicted_convergence():
    # truth: under partition, x is split (n2 stale); a model that predicts full convergence is wrong
    truth = _state(["put n0 x b", "advance 2", "partition n0 n1 | n2", "put n0 x c", "advance 2"])
    converged_guess = _state(["put n0 x c", "advance 2"])  # all replicas at (1,c) — no split
    assert object_consistency_view(truth, "x") != object_consistency_view(converged_guess, "x")
    # x is mispredicted, y is fine -> 1/2 objects consistency-faithful
    assert consistency_faithfulness(truth, converged_guess) == 0.5


# --- bits-to-correct -----------------------------------------------------------------------------

def test_bits_to_correct_gates_the_delta():
    result = ORACLE.step(DistributedState.initial(CFG), parse_dist_action("put n0 x b"))
    assert bits_to_correct(result.delta, result.delta) == 0.0
    assert delta_exact(result.delta, result.delta)
    assert bits_to_correct(result.delta[:-1], result.delta) > 0.0  # a missing edit costs bits
    assert not delta_exact(result.delta[:-1], result.delta)


def test_delta_exact_rate():
    r = ORACLE.step(DistributedState.initial(CFG), parse_dist_action("put n0 x b"))
    pairs = [(r.delta, r.delta), (r.delta[:-1], r.delta), (r.delta, r.delta)]
    assert delta_exact_rate(pairs) == 2 / 3
    assert delta_exact_rate([]) == 1.0
