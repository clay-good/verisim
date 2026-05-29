"""The baseline sweep runs end to end and brackets the loop's behavior (SPEC-2 §8)."""

from __future__ import annotations

from verisim.experiments.baselines import DEFAULT_RHOS, run_baseline_sweep
from verisim.metrics.record import RunRecord


def test_baseline_sweep_runs_and_brackets():
    records = run_baseline_sweep(n_steps=40, seed=3, epsilon=0.0)
    assert len(records) == 2 * len(DEFAULT_RHOS)

    by_model: dict[str, list[RunRecord]] = {"null": [], "oracle": []}
    for rec in records:
        by_model[rec.config["model"]].append(rec)

    # b2 (oracle-backed) is perfect: H_eps == T at every budget, including rho=0.
    for rec in by_model["oracle"]:
        assert rec.faithful_horizon == 40

    # b3 (null) is held exact only when consulted every step (rho=1), and drifts
    # otherwise -> it never beats the perfect model.
    null_at_rho1 = next(r for r in by_model["null"] if r.config["rho"] == 1.0)
    null_at_rho0 = next(r for r in by_model["null"] if r.config["rho"] == 0.0)
    assert null_at_rho1.faithful_horizon == 40
    assert null_at_rho0.faithful_horizon < 40

    # Every cell respects its budget.
    for rec in records:
        assert rec.oracle_calls <= (rec.config["rho"] * 40) + 1e-9
