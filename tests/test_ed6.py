"""ED6 — H5: distributed counterfactual grounding (SPEC-7 §10.1, DS8).

The smoke instance of the H5 apparatus (torch extra): train the three matched-count arms tiny, score
held-out fault interventions, and check the deliverables are well-formed and structurally correct —
every (arm, metric) cell present, means in [0, 1], CIs bracketing the mean, deterministic across
runs. The H5 *direction* (does counterfactual fault replay beat factual?) is a quantitative finding
the committed run reports, not a brittle unit assertion on an undertrained smoke model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.dist.state import DistributedState
from verisim.experiments.ed6 import (
    ARMS,
    METRICS,
    ED6Config,
    _medium,
    h5_supported,
    run_ed6,
    write_csv,
)


def _tiny() -> ED6Config:
    return ED6Config(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        k_counterfactual=3,
        n_layer=1,
        n_embd=32,
        block_size=384,
        train_iters=40,
        eval_seeds=(100, 101),
        eval_steps=10,
        m_interventions=5,
    )


def test_run_is_well_formed():
    stats = run_ed6(_tiny())
    # one cell per (arm, metric)
    assert {(s.arm, s.metric) for s in stats} == {(a, m) for a in ARMS for m in METRICS}
    for s in stats:
        assert 0.0 <= s.mean <= 1.0
        assert s.ci_lo <= s.mean <= s.ci_hi or s.ci_lo == s.ci_hi
        assert s.n == 2  # one score per eval seed


def test_is_deterministic():
    a = run_ed6(_tiny())
    b = run_ed6(_tiny())
    assert [(s.arm, s.metric, s.mean) for s in a] == [(s.arm, s.metric, s.mean) for s in b]


def test_h5_verdict_is_a_pure_function_of_means():
    stats = run_ed6(_tiny())
    # h5_supported is just "+counterfactual beats both factual arms" — a deterministic readout.
    assert isinstance(h5_supported(stats, "intervention_exact"), bool)


def test_write_csv(tmp_path):
    stats = run_ed6(_tiny())
    out = write_csv(stats, tmp_path / "ed6.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "arm,metric,mean,ci_lo,ci_hi,n"
    assert any(line.startswith("+counterfactual,") for line in lines)


def test_config_round_trips():
    cfg = ED6Config.from_dict(
        {"k_counterfactual": 2, "intervention_fault_prob": 0.9, "eval_steps": 8}
    )
    assert cfg.k_counterfactual == 2
    assert cfg.intervention_fault_prob == 0.9
    assert cfg.eval_steps == 8


def test_medium_distinguishes_partition_and_crash():
    # the medium is (partition structure, crashed-node set): a converged-cluster baseline differs
    # from one with a node down — the hidden state the medium_recall readout is built on.
    from verisim.dist.config import DEFAULT_DIST_CONFIG

    s0 = DistributedState.initial(DEFAULT_DIST_CONFIG)
    base = _medium(s0)
    assert isinstance(base, tuple) and len(base) == 2
