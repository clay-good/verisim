"""HC3 metric tests (SPEC-6 §9): composed + per-subsystem divergence, bits-to-correct, the
composition-faithfulness diagnostic (H13), privilege-faithfulness, and the host run-record schema.

Pins the deterministic metric core before any model claim (the M3/NW3 discipline): the metrics
are pure functions of the bundle state/delta, ``0`` iff faithful, decompose by subsystem, and
round-trip through the run-record schema.
"""

from __future__ import annotations

import random

from verisim.delta.edits import Modify
from verisim.host import (
    FdOpen,
    FsDelta,
    HostDelta,
    HostState,
    ProcSpawn,
    SetExit,
    parse_host_action,
)
from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.hostdata.drivers import HostDriver
from verisim.hostmetrics import (
    SUBSYSTEMS,
    HostRunRecord,
    bits_to_correct,
    bits_to_correct_by_subsystem,
    composed_faithful,
    composition_law,
    delta_exact,
    divergence,
    divergence_by_subsystem,
    faithful_horizon,
    privilege_faithfulness,
    read_host_records,
    step_faithful_by_subsystem,
    write_host_records,
)
from verisim.hostoracle import EXIT_ERR, EXIT_OK, ReferenceHostOracle

ORACLE = ReferenceHostOracle()


def _run(cmds: list[str]) -> HostState:
    state = HostState.initial()
    for cmd in cmds:
        state = ORACLE.step(state, parse_host_action(cmd)).state
    return state


# -- divergence: composed + per-subsystem -------------------------------------------------


def test_divergence_zero_iff_identical() -> None:
    a = _run(["fork 1", "open 2 /log", "write 2 0 alpha"])
    b = _run(["fork 1", "open 2 /log", "write 2 0 alpha"])
    assert divergence(a, b) == 0.0
    assert all(d == 0.0 for d in divergence_by_subsystem(a, b).values())
    c = _run(["fork 1", "open 2 /log", "write 2 0 beta"])  # different file content
    assert 0.0 < divergence(a, c) <= 1.0


def test_divergence_localizes_to_the_right_subsystem() -> None:
    base = _run(["fork 1"])
    # A divergence purely in the filesystem subsystem: same procs/fds, different file content.
    fs_only = _run(["fork 1", "open 2 /log", "write 2 0 alpha", "close 2 0"])
    by_sub = divergence_by_subsystem(base, fs_only)
    assert by_sub["fs"] > 0.0
    assert by_sub["proc"] == 0.0  # the process table is identical (fork in both)
    # A divergence purely in the process table: an extra fork, no file activity.
    proc_only = _run(["fork 1", "fork 1"])
    by_sub2 = divergence_by_subsystem(base, proc_only)
    assert by_sub2["proc"] > 0.0
    assert by_sub2["fs"] == 0.0


def test_step_faithful_by_subsystem_and_composed() -> None:
    a = _run(["fork 1", "open 2 /log", "write 2 0 alpha"])
    b = _run(["fork 1", "open 2 /log", "write 2 0 beta"])  # fs differs
    faithful = step_faithful_by_subsystem(a, b, epsilon=0.0)
    assert faithful["proc"] is True and faithful["fd"] is True
    assert faithful["fs"] is False
    assert composed_faithful(faithful) is False
    assert composed_faithful(step_faithful_by_subsystem(a, a, epsilon=0.0)) is True


# -- bits-to-correct: composed + per-subsystem --------------------------------------------


def test_bits_to_correct_zero_iff_equal_and_localizes() -> None:
    fs_edit = Modify("/log", "alpha")
    true: HostDelta = [ProcSpawn(2, 1, 0), FdOpen(2, 0, "/log"), FsDelta([fs_edit]), SetExit(0)]
    assert bits_to_correct(true, list(true)) == 0.0
    assert delta_exact(true, list(true)) is True
    # Drop the fd edit: the residual is charged entirely to the fd subsystem.
    near: HostDelta = [ProcSpawn(2, 1, 0), FsDelta([fs_edit]), SetExit(0)]
    by_sub = bits_to_correct_by_subsystem(near, true)
    assert by_sub["fd"] > 0.0
    assert by_sub["proc"] == 0.0 and by_sub["fs"] == 0.0 and by_sub["global"] == 0.0
    assert bits_to_correct(near, true) > 0.0
    assert delta_exact(near, true) is False


def test_bits_to_correct_monotone_and_fs_delegates() -> None:
    true: HostDelta = [FsDelta([Modify("/log", "alpha")])]
    # A wrong-content fs prediction costs more than a missing fd, less than dropping everything.
    one_off: HostDelta = [FsDelta([Modify("/log", "beta")])]
    empty: HostDelta = []
    assert bits_to_correct(one_off, true) > 0.0
    assert bits_to_correct(empty, true) > 0.0
    # The fs residual is charged to the fs subsystem only.
    by_sub = bits_to_correct_by_subsystem(one_off, true)
    assert by_sub["fs"] > 0.0
    assert sum(v for k, v in by_sub.items() if k != "fs") == 0.0


# -- composition-faithfulness diagnostic (H13) --------------------------------------------


def test_composition_law_multiplicative_when_failures_independent() -> None:
    # proc fails on odd steps, fs fails on steps divisible by 3 -- (near-)independent.
    steps = [
        {"proc": (t % 2 == 0), "fs": (t % 3 != 0)} for t in range(60)
    ]
    law = composition_law(steps)
    # composed should sit near the product of the two acceptance rates, not the min.
    assert law.multiplicative_residual <= law.weakest_link_residual
    assert law.verdict in {"multiplicative", "coupled"}
    assert 0.0 <= law.composed_acceptance <= min(law.subsystem_acceptance.values()) + 1e-9


def test_composition_law_weakest_link_when_failures_coincide() -> None:
    # Whenever fs fails, proc fails on the same step (perfectly correlated): composed == min.
    steps = []
    for t in range(40):
        fs_ok = t % 4 != 0
        steps.append({"proc": fs_ok, "fs": fs_ok})
    law = composition_law(steps)
    assert law.verdict == "weakest_link"
    assert abs(law.composed_acceptance - law.weakest_link_prediction) < 1e-9


def test_composition_law_all_faithful_is_degenerate_multiplicative() -> None:
    law = composition_law([{"proc": True, "fs": True}] * 10)
    assert law.composed_acceptance == 1.0
    assert law.verdict == "multiplicative"
    assert composition_law([]).composed_acceptance == 1.0


# -- privilege-faithfulness (§9.4) --------------------------------------------------------


def test_privilege_faithfulness_grades_denials() -> None:
    # A non-root process that tries setuid is denied (EPERM); the oracle pins this.
    state = _run(["fork 1", "setuid 2 1000"])  # 2 is root -> ok, now uid 1000
    denied = ORACLE.step(state, parse_host_action("setuid 2 0"))  # non-root setuid -> EPERM
    assert denied.exit_code == EXIT_ERR
    # A model that predicts the denial correctly scores 1.0; one that predicts success scores 0.0.
    assert privilege_faithfulness([EXIT_ERR], [denied.exit_code]) == 1.0
    assert privilege_faithfulness([EXIT_OK], [denied.exit_code]) == 0.0
    # Mixed: agree on the allowed, disagree on the denied -> 0.5.
    assert privilege_faithfulness([EXIT_OK, EXIT_OK], [EXIT_OK, EXIT_ERR]) == 0.5
    assert privilege_faithfulness([], []) == 1.0


# -- the host run-record schema -----------------------------------------------------------


def test_host_run_record_horizons_and_roundtrip(tmp_path) -> None:
    rec = HostRunRecord(
        config={"driver": "uniform"},
        seed=7,
        epsilon=0.1,
        divergences=[0.0, 0.05, 0.2, 0.0],  # composed: first exceedance at index 2
        subsystem_divergences={"proc": [0.0, 0.0, 0.0, 0.0], "fs": [0.0, 0.05, 0.2, 0.0]},
        consultation_schedule=[True, False, True, False],
    )
    assert rec.faithful_horizon == 2  # composed H_eps
    assert rec.faithful_horizon == faithful_horizon(rec.divergences, rec.epsilon)
    horizons = rec.subsystem_horizons
    assert horizons["proc"] == 4  # proc never exceeds
    assert horizons["fs"] == 2
    assert rec.oracle_calls == 2

    path = write_host_records([rec], tmp_path / "host_runs.jsonl")
    (back,) = read_host_records(path)
    assert back.divergences == rec.divergences
    assert back.subsystem_divergences == rec.subsystem_divergences
    assert back.faithful_horizon == rec.faithful_horizon


# -- end-to-end over a generated trajectory (the metrics compose with the data factory) ---


def test_metrics_over_a_driver_trajectory_are_self_consistent() -> None:
    driver = HostDriver(name="forky", config=DEFAULT_HOST_CONFIG, rng=random.Random(3))
    state = HostState.initial()
    for _ in range(40):
        action = driver.sample(state)
        result = ORACLE.step(state, action)
        # A perfect predictor: the true next state diverges 0 from itself, true delta needs 0 bits.
        assert divergence(result.state, result.state) == 0.0
        assert bits_to_correct(result.delta, result.delta) == 0.0
        assert delta_exact(result.delta, result.delta) is True
        # Every per-subsystem divergence is in [0, 1] and the subsystem set is the canonical one.
        by_sub = divergence_by_subsystem(state, result.state)
        assert set(by_sub) == set(SUBSYSTEMS)
        assert all(0.0 <= d <= 1.0 for d in by_sub.values())
        state = result.state
