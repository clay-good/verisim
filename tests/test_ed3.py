"""ED3 — distributed correction-operator comparison (SPEC-7 §8.3, §10 ED3; DS7).

The smoke instance of the operator apparatus (dependency-free, GPU-free): a tiny seeded sweep that
checks the deliverables are well-formed and that the central §8.3 verdict has the right *structural*
shape — the three full-correction operators (`hard_reset`/`residual`/`projection`) are identical on
faithful horizon (the v0 identity), and the partial `replicas_only` operator breaks that identity
*only* for the in-flight (`subtle`) error class, never for the replica (`gross`) one. The committed
figure comes from the local run.
"""

from __future__ import annotations

from verisim.distloop import HardReset, ReplicasOnlyCorrection
from verisim.experiments.ed3 import OPERATORS, ED3Config, ED3Result, run_ed3, write_csv


def _tiny() -> ED3Config:
    return ED3Config(eval_seeds=(100, 101, 102, 103), n_steps=20, rho=0.5)


def test_run_ed3_is_well_formed():
    result = run_ed3(_tiny())
    assert isinstance(result, ED3Result)
    keys = {(c["mode"], c["operator"]) for c in result.cells}
    expected = {(m, op) for m in ("gross", "subtle") for op in OPERATORS}
    assert keys == expected
    for c in result.cells:
        assert c["h_eps"] >= 0.0
        assert c["ci_lo"] <= c["h_eps"] <= c["ci_hi"] or c["ci_lo"] == c["ci_hi"]
        assert c["repaired_fraction"] >= 0.0
    assert {v["mode"] for v in result.verdict} == {"gross", "subtle"}


def test_full_correction_operators_satisfy_the_v0_identity():
    """`hard_reset` / `residual` / `projection` snap to truth → identical horizon in both modes."""
    result = run_ed3(_tiny())
    by = {(c["mode"], c["operator"]): c["h_eps"] for c in result.cells}
    for mode in ("gross", "subtle"):
        assert by[(mode, "hard_reset")] == by[(mode, "residual")] == by[(mode, "projection")]
    for v in result.verdict:
        assert v["identity_holds"] is True
        assert v["full_correction_spread"] == 0.0


def test_partial_operator_breaks_the_identity_only_for_subtle():
    """`replicas_only` recovers full horizon for replica (gross) errors and *less* for in-flight
    (subtle) errors — the distributed identity-break the weak-consistency regime predicts."""
    result = run_ed3(_tiny())
    by_mode = {v["mode"]: v for v in result.verdict}
    # gross: a corrupted replica write is fixed by replicas_only → no horizon cost
    assert by_mode["gross"]["partial_costs_horizon"] is False
    assert by_mode["gross"]["horizon_gap"] == 0.0
    # subtle: a corrupted in-flight message is trusted by replicas_only → strictly less horizon
    assert by_mode["subtle"]["partial_costs_horizon"] is True
    assert by_mode["subtle"]["horizon_gap"] > 0.0


def test_hard_reset_is_the_default_operator_in_the_runner():
    """The runner with no operator must match an explicit HardReset, and differ from the partial."""
    import random

    from verisim.dist.config import DEFAULT_DIST_CONFIG
    from verisim.dist.state import DistributedState
    from verisim.distdata import DistDriver
    from verisim.distloop import DistNoisyModel, FixedTierPolicy, budget_for_rho, run_dist_rollout
    from verisim.distoracle import ReferenceDistOracle
    from verisim.loop.policy import fixed_interval_for_rho

    oracle = ReferenceDistOracle(DEFAULT_DIST_CONFIG)
    drv = DistDriver("contention", DEFAULT_DIST_CONFIG, random.Random(1))
    state = DistributedState.initial(DEFAULT_DIST_CONFIG)
    actions = []
    for _ in range(20):
        a = drv.sample(state)
        actions.append(a)
        state = oracle.step(state, a).state

    def horizon(operator):
        model = DistNoisyModel(oracle, noise=0.4, mode="subtle", rng=random.Random(8))
        rec = run_dist_rollout(
            model, oracle, DistributedState.initial(DEFAULT_DIST_CONFIG), actions,
            fixed_interval_for_rho(0.5), epsilon=0.0, tier_policy=FixedTierPolicy("bit_exact"),
            operator=operator, budget=budget_for_rho(0.5, len(actions)), seed=8,
        )
        return rec.faithful_horizon

    assert horizon(None) == horizon(HardReset())            # default is HardReset
    assert horizon(ReplicasOnlyCorrection()) <= horizon(HardReset())  # partial corrects ≤ full


def test_write_csv(tmp_path):
    result = run_ed3(_tiny())
    out = write_csv(result, tmp_path / "ed3.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("cell,") for line in lines)
    assert any(line.startswith("verdict,") for line in lines)


def test_config_round_trips():
    cfg = ED3Config.from_dict({"noise": 0.3, "driver": "uniform", "rho": 0.25})
    assert cfg.noise == 0.3
    assert cfg.driver == "uniform"
    assert cfg.rho == 0.25
