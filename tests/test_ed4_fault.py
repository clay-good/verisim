"""ED4-fault — H21: fault-injected vs fault-free training under fault (SPEC-7 §10.2, DS7).

The smoke instance of the H21 apparatus (torch extra): train a tiny fault-free and a tiny
fault-injected `M_θ` of equal volume, sweep eval fault-intensity free-running, and check the
deliverables are well-formed and structurally correct — both regimes get a free-run-`H_ε`-vs-
fault-intensity curve over the swept fault probabilities, and each gets a clean teacher-forced
accuracy in [0, 1]. The H21 *direction* (does fault-injection help under fault?) is a quantitative
finding the committed run reports, not a brittle unit assertion.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.ed4_fault import (
    REGIMES,
    ED4FaultConfig,
    ED4FaultResult,
    run_ed4_fault,
    write_csv,
)


def _tiny() -> ED4FaultConfig:
    return ED4FaultConfig(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=40,
        n_layer=1,
        n_embd=32,
        block_size=384,
        eval_fault_probs=(0.0, 0.3),
        eval_seeds=(100, 101),
        n_steps=12,
        clean_eval_seeds=(200,),
    )


def test_run_is_well_formed():
    result = run_ed4_fault(_tiny())
    assert isinstance(result, ED4FaultResult)
    assert set(result.curves) == set(REGIMES)
    for regime in REGIMES:
        assert [p["fault_prob"] for p in result.curves[regime]] == [0.0, 0.3]
        for p in result.curves[regime]:
            assert p["ci_lo"] <= p["h_eps"] <= p["ci_hi"] or p["ci_lo"] == p["ci_hi"]
            assert 0.0 <= p["h_eps"] <= 12.0  # bounded by n_steps
        assert 0.0 <= result.clean_accuracy[regime] <= 1.0


def test_is_deterministic():
    a = run_ed4_fault(_tiny())
    b = run_ed4_fault(_tiny())
    for regime in REGIMES:
        assert [p["h_eps"] for p in a.curves[regime]] == [p["h_eps"] for p in b.curves[regime]]
    assert a.clean_accuracy == b.clean_accuracy


def test_write_csv(tmp_path):
    result = run_ed4_fault(_tiny())
    out = write_csv(result, tmp_path / "ed4_fault.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("regime,")
    assert any(line.startswith("fault_free,") for line in lines)
    assert any(line.startswith("fault_injected,") for line in lines)


def test_config_round_trips():
    cfg = ED4FaultConfig.from_dict(
        {"train_fault_prob": 0.3, "eval_fault_probs": [0.0, 0.5], "rho": 0.0}
    )
    assert cfg.train_fault_prob == 0.3
    assert cfg.eval_fault_probs == (0.0, 0.5)
    assert cfg.rho == 0.0
