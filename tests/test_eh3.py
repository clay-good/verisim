"""EH3 host operator-comparison tests (SPEC-6 §8.3, HC7).

Mirrors v0's ``test_e3`` and the network ``test_en3``: the full-consult operators coincide on
``H_ε`` (the full-truth identity), a per-subsystem filter corrects strictly less (lower ``H_ε`` at
equal consultation count -- the §8.3 no-identity-collapse result), and the per-subsystem consult
spends strictly fewer oracle-bits (the §9.4 cost lens that makes horizon-per-bit the real question).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config
from verisim.experiments.eh3 import EH3Config, efficiency_by_operator, run_eh3


def _tiny_config() -> EH3Config:
    base = EH1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=80,
        n_layer=1,
        n_embd=32,
        block_size=160,
        difficulties={"low": "forky"},
        eval_seeds=(100, 101),
        eval_steps=12,
        epsilons=(0.0, 0.1),
    )
    return EH3Config(base=base, rho=0.5)


def test_run_eh3_produces_records_for_every_operator():
    records = run_eh3(_tiny_config())
    operators = {str(r.config["operator"]) for r in records}
    assert operators == {
        "hard_reset", "residual", "projection", "subsystem_rr", "subsystem_proc", "subsystem_fd"
    }
    for r in records:
        assert r.oracle_calls <= r.config["rho"] * r.config["n_steps"] + 1e-9
        assert r.config["oracle_bits"] >= 0


def test_full_operators_coincide_on_horizon():
    """``hard_reset`` / ``residual`` / ``projection`` all snap to full truth -> identical H_ε."""
    records = run_eh3(_tiny_config())
    eps0 = [r for r in records if r.epsilon == 0.0]
    horizons = {
        op: sorted(r.faithful_horizon for r in eps0 if r.config["operator"] == op)
        for op in ("hard_reset", "residual", "projection")
    }
    assert horizons["hard_reset"] == horizons["residual"] == horizons["projection"]


def test_per_subsystem_corrects_strictly_less_and_costs_fewer_bits():
    """A per-subsystem consult corrects less (H_ε no greater) at strictly fewer oracle-bits."""
    records = run_eh3(_tiny_config())
    eff = efficiency_by_operator(records)
    full = eff["hard_reset"]
    for op in ("subsystem_rr", "subsystem_proc", "subsystem_fd"):
        # strictly cheaper per consult (one subsystem's facts vs the whole bundle's)
        assert eff[op]["mean_bits"] < full["mean_bits"]
        # corrects less, so faithful horizon is no greater than the full consult's
        assert eff[op]["mean_h"] <= full["mean_h"] + 1e-9


def test_run_eh3_is_deterministic():
    a = run_eh3(_tiny_config())
    b = run_eh3(_tiny_config())
    assert [r.divergences for r in a] == [r.divergences for r in b]
    assert [r.config["oracle_bits"] for r in a] == [r.config["oracle_bits"] for r in b]
