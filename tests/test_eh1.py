"""EH1 composed-host sweep-harness tests (SPEC-6 §0, §9.2, HC6 -- the prime directive).

Exercises the full host pipeline on a tiny config: train -> sweep -> host records, the loop
invariants surfaced through the curve (ρ=1 ⇒ composed H_ε = T; determinism), and the composition-law
diagnostic (H13). Mirrors v0's ``test_e1`` and the network ``test_en1`` for the host world.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.eh1 import EH1Config, run_eh1
from verisim.hostmetrics.composition import CompositionLaw
from verisim.hostmetrics.divergence import SUBSYSTEMS
from verisim.hostmetrics.record import read_host_records, write_host_records


def _tiny_config() -> EH1Config:
    return EH1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=60,
        n_layer=1,
        n_embd=32,
        block_size=160,
        difficulties={"low": "forky"},
        eval_seeds=(100, 101),
        eval_steps=8,
        rhos=(0.0, 0.5, 1.0),
        epsilons=(0.0, 0.1),
    )


def test_run_eh1_produces_expected_records():
    records = run_eh1(_tiny_config()).records
    # difficulties(1) x eval_seeds(2) x rhos(3) x epsilons(2)
    assert len(records) == 1 * 2 * 3 * 2
    for rec in records:
        assert set(rec.config) >= {"experiment", "model", "difficulty", "rho", "n_steps"}
        assert len(rec.divergences) == rec.config["n_steps"] == 8
        # every record carries every per-subsystem trajectory (the H13 diagnostic input)
        assert set(rec.subsystem_divergences) == set(SUBSYSTEMS)
        for sub in SUBSYSTEMS:
            assert len(rec.subsystem_divergences[sub]) == 8
        assert rec.oracle_calls <= rec.config["rho"] * 8 + 1e-9
        assert rec.config["oracle_bits"] >= 0  # the §9.4 per-subsystem-efficiency denominator


def test_rho1_gives_full_composed_faithful_horizon():
    """Consult every step (full mode) -> the coupled rollout matches truth (composed H_ε = T)."""
    records = run_eh1(_tiny_config()).records
    rho1 = [r for r in records if r.config["rho"] == 1.0]
    assert rho1
    for rec in rho1:
        assert rec.faithful_horizon == rec.config["n_steps"]
        assert all(d == 0.0 for d in rec.divergences)


def test_composition_law_is_well_formed():
    """H13: the composition-law verdict is computed per difficulty and is internally consistent."""
    law = run_eh1(_tiny_config()).composition
    assert set(law) == {"low"}
    result = law["low"]
    assert isinstance(result, CompositionLaw)
    assert result.verdict in {"multiplicative", "weakest_link", "coupled"}
    # A step is composed-faithful iff *every* subsystem is, so composed <= min_i a_i always holds.
    # The *lower* bound prod(a_i) holds only under independent-or-positively-correlated failures;
    # anti-correlated failures (the model failing different subsystems on different steps) push
    # composed *below* the multiplicative floor -- exactly the "coupled" honest negative (§9.2).
    assert 0.0 <= result.composed_acceptance <= result.weakest_link_prediction + 1e-9
    assert set(result.subsystem_acceptance) == set(SUBSYSTEMS)


def test_run_eh1_is_deterministic():
    a = run_eh1(_tiny_config())
    b = run_eh1(_tiny_config())
    assert [r.divergences for r in a.records] == [r.divergences for r in b.records]
    assert [r.subsystem_divergences for r in a.records] == [
        r.subsystem_divergences for r in b.records
    ]
    assert {d: law.to_dict() for d, law in a.composition.items()} == {
        d: law.to_dict() for d, law in b.composition.items()
    }


def test_host_records_roundtrip_jsonl(tmp_path):
    records = run_eh1(_tiny_config()).records
    path = write_host_records(records, tmp_path / "host_records.jsonl")
    loaded = read_host_records(path)
    assert len(loaded) == len(records)
    assert [r.divergences for r in loaded] == [r.divergences for r in records]
    assert [r.subsystem_divergences for r in loaded] == [r.subsystem_divergences for r in records]
