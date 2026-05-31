"""EN3 network correction/belief-operator comparison tests (SPEC-5 §12, §8.3, NW7).

The marquee partial-observation invariant (no v0 identity collapse): the three
full-consultation operators coincide on faithful horizon, while the probe-mode belief
filter — which corrects only one host's subgraph per consult — earns no more horizon and
spends strictly fewer oracle-bits.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from statistics import fmean

from verisim.experiments.en1 import EN1Config
from verisim.experiments.en3 import EN3Config, run_en3
from verisim.metrics.record import RunRecord


def _tiny() -> EN3Config:
    base = EN1Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=60,
        n_layer=1,
        n_embd=32,
        block_size=128,
        difficulties={"low": "weighted"},
        eval_seeds=(100, 101),
        eval_steps=8,
        rhos=(0.5,),
        epsilons=(0.0,),
    )
    return EN3Config(base=base, rho=0.5, policy="fixed")


def _mean_h(records: list[RunRecord], operator: str) -> float:
    return fmean(r.faithful_horizon for r in records if r.config["operator"] == operator)


def _mean_bits(records: list[RunRecord], operator: str) -> float:
    return fmean(r.config["oracle_bits"] for r in records if r.config["operator"] == operator)


def test_full_operators_coincide_and_probe_differs():
    records = run_en3(_tiny())
    # operators(4) x difficulties(1) x eval_seeds(2) x epsilons(1)
    assert len(records) == 4 * 1 * 2 * 1

    # The three full-consult operators snap to the same truth -> identical faithful horizon.
    hr = _mean_h(records, "hard_reset")
    assert _mean_h(records, "residual") == hr
    assert _mean_h(records, "projection") == hr

    # The probe belief-filter corrects strictly less -> no more horizon (identity broken)...
    assert _mean_h(records, "belief_filter") <= hr
    # ... but it is strictly cheaper: a one-host probe reveals fewer facts than a full consult.
    assert _mean_bits(records, "belief_filter") < _mean_bits(records, "hard_reset")


def test_run_en3_is_deterministic():
    a = run_en3(_tiny())
    b = run_en3(_tiny())
    assert [r.divergences for r in a] == [r.divergences for r in b]
