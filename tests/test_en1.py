"""EN1 network sweep-harness tests (SPEC-5 §12, NW6).

Exercises the full network pipeline on a tiny config: train -> sweep -> records, and the
loop invariants surfaced through the curve (ρ=1 ⇒ H_ε = T; determinism). Mirrors v0's
``test_e1`` for the network world.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.en1 import EN1Config, run_en1
from verisim.metrics import read_records, write_records


def _tiny_config() -> EN1Config:
    return EN1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=60,
        n_layer=1,
        n_embd=32,
        block_size=128,
        difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        eval_steps=8,
        rhos=(0.0, 0.5, 1.0),
        epsilons=(0.0, 0.1),
    )


def test_run_en1_produces_expected_records():
    records = run_en1(_tiny_config())
    # difficulties(1) x eval_seeds(2) x rhos(3) x epsilons(2)
    assert len(records) == 1 * 2 * 3 * 2
    for rec in records:
        assert set(rec.config) >= {"experiment", "model", "difficulty", "rho", "n_steps"}
        assert len(rec.divergences) == rec.config["n_steps"] == 8
        assert rec.oracle_calls <= rec.config["rho"] * 8 + 1e-9
        assert rec.config["oracle_bits"] >= 0  # the §9.4 probe-efficiency denominator


def test_rho1_gives_full_faithful_horizon():
    """Consult every step -> the coupled rollout matches ground truth (H_ε = T)."""
    records = run_en1(_tiny_config())
    rho1 = [r for r in records if r.config["rho"] == 1.0]
    assert rho1
    for rec in rho1:
        assert rec.faithful_horizon == rec.config["n_steps"]


def test_run_en1_is_deterministic():
    a = run_en1(_tiny_config())
    b = run_en1(_tiny_config())
    assert [r.divergences for r in a] == [r.divergences for r in b]


def test_records_roundtrip_jsonl(tmp_path):
    records = run_en1(_tiny_config())
    path = write_records(records, tmp_path / "en1.jsonl")
    restored = read_records(path)
    assert [r.to_json() for r in restored] == [r.to_json() for r in records]
