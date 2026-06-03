"""Harness tests for the three new findings (EH8 privilege, EH6 two-oracle, EH-H13-scale).

Tiny configs; the findings' *outcomes* are data read off the committed figures, not asserted -- the
tests pin the apparatus (well-formed metrics, bounds, determinism). Torch-gated.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config
from verisim.experiments.eh6_two_oracle import EH6Config, run_eh6
from verisim.experiments.eh8_privilege import EH8Config, predicted_exit, run_eh8
from verisim.experiments.eh_h13_scale import EHH13ScaleConfig, run_eh_h13_scale
from verisim.host.delta import ProcSpawn, SetExit


def _base() -> EH1Config:
    return EH1Config(
        train_seeds=(0, 1), train_steps_per_traj=16, train_iters=60,
        n_layer=1, n_embd=32, block_size=160, difficulties={"low": "adversarial"},
        eval_seeds=(100, 101), eval_steps=10, epsilons=(0.0, 0.1),
    )


def test_predicted_exit_reads_the_last_setexit():
    assert predicted_exit([ProcSpawn(2, 1, 0), SetExit(1)]) == 1
    assert predicted_exit([ProcSpawn(2, 1, 0)]) == 0  # no SetExit -> EXIT_OK default


def test_eh8_privilege_metrics_are_well_formed():
    cfg = EH8Config(base=_base(), max_pid=32, graph_d_model=24, graph_mp_rounds=2, graph_iters=60,
                    graph_batch=16)
    results = run_eh8(cfg)
    assert set(results) == {"flat", "factored"}
    for r in results.values():
        for k in ("privilege_faithfulness", "setuid_faithfulness", "denied_recall"):
            assert 0.0 <= r[k] <= 1.0
        assert r["n_transitions"] > 0


def test_eh6_two_oracle_h12_structure():
    cfg = EH6Config(base=_base(), arm="factored", max_pid=32, graph_d_model=24, graph_mp_rounds=2,
                    graph_iters=60, graph_batch=16)
    r = run_eh6(cfg)
    # non-redundant is 0 by construction (the invariant is a pure function of the state)
    assert r["non_redundant_rate"] == 0.0
    # the security consult is strictly cheaper than the full state
    assert 0.0 < r["consult_bits_ratio"] < 1.0
    assert 0.0 <= r["invariant_sufficient_rate"] <= 1.0


def test_eh_h13_scale_covers_widths_and_is_bounded():
    cfg = EHH13ScaleConfig(
        base=_base(), thread_counts=(2, 4), eval_workload_seeds=(10,), eval_chaos_seeds=(0, 1),
        max_pid=32, graph_d_model=24, graph_mp_rounds=2, graph_iters=60, graph_batch=16,
    )
    points = run_eh_h13_scale(cfg)
    assert {p.n_threads for p in points} == {2, 4}
    for p in points:
        # composed acceptance <= the weakest link (one bad subsystem fails the whole step)
        assert 0.0 <= p.composed <= p.weakest_link + 1e-9
        assert p.verdict in {"multiplicative", "weakest_link", "coupled"}


def test_findings_are_deterministic():
    cfg = EH6Config(base=_base(), arm="factored", max_pid=32, graph_d_model=24, graph_mp_rounds=2,
                    graph_iters=60, graph_batch=16)
    assert run_eh6(cfg) == run_eh6(cfg)
