"""ED6 two-oracle — H12: the consistency oracle is redundant but decision-sufficient (SPEC-7 §10.1).

Dependency-free, GPU-free (the synthetic ``DistNoisyModel``), so CI runs a real instance. The two
load-bearing H12 facts are asserted because they are structural, not noisy: the consistency oracle
is **non-redundant-rate 0** by construction (a bit-exact-correct prediction is always consistency-
correct), and the decision-sufficiency is **mode-dependent** — a ``subtle`` (in-flight) error is
consistency-invisible (sufficiency high) while a ``gross`` (durable-replica) error is consistency-
visible (sufficiency low). The exact magnitudes are a quantitative finding the committed run gives.
"""

from __future__ import annotations

from verisim.experiments.ed6_two_oracle import (
    ED6TwoOracleConfig,
    ED6TwoOracleResult,
    consistency_consult_facts,
    run_ed6_two_oracle,
    write_csv,
)


def _tiny() -> ED6TwoOracleConfig:
    return ED6TwoOracleConfig(eval_seeds=(100, 101, 102, 103), n_steps=24, noise=0.6)


def test_run_is_well_formed():
    result = run_ed6_two_oracle(_tiny())
    assert isinstance(result, ED6TwoOracleResult)
    assert {c["mode"] for c in result.per_mode} == {"gross", "subtle"}
    for c in result.per_mode:
        for key in ("non_redundant_rate", "consistency_sufficient_rate", "full_wrong_rate",
                    "consult_fact_ratio"):
            assert 0.0 <= c[key] <= 1.0
            assert c[f"{key}_lo"] <= c[key] <= c[f"{key}_hi"] or c[f"{key}_lo"] == c[f"{key}_hi"]


def test_non_redundant_is_zero_by_construction():
    # the consistency view is a pure function of the replica state, so a bit-exact-correct
    # prediction is always consistency-correct: the cheap oracle catches nothing the full one drops.
    result = run_ed6_two_oracle(_tiny())
    for c in result.per_mode:
        assert c["non_redundant_rate"] == 0.0
        assert c["redundant_for_verification"] is True


def test_decision_sufficiency_is_mode_dependent():
    # subtle (in-flight) errors are consistency-invisible -> sufficiency high; gross (durable) ones
    # are consistency-visible -> low. The in-flight medium is the distributed world's hidden state.
    result = run_ed6_two_oracle(_tiny())
    by_mode = {c["mode"]: c for c in result.per_mode}
    assert by_mode["subtle"]["consistency_sufficient_rate"] > \
        by_mode["gross"]["consistency_sufficient_rate"]
    # the cheap consult is materially cheaper than the full state
    assert by_mode["subtle"]["consult_fact_ratio"] < 1.0


def test_is_deterministic():
    a = run_ed6_two_oracle(_tiny())
    b = run_ed6_two_oracle(_tiny())
    assert [(c["mode"], c["consistency_sufficient_rate"]) for c in a.per_mode] == \
        [(c["mode"], c["consistency_sufficient_rate"]) for c in b.per_mode]


def test_consult_facts_counts_consistency_view():
    from verisim.dist.config import DEFAULT_DIST_CONFIG
    from verisim.dist.state import DistributedState

    s0 = DistributedState.initial(DEFAULT_DIST_CONFIG)
    # a converged initial cluster: each object's consistency view is a singleton, so the consult
    # cost equals the object count.
    assert consistency_consult_facts(s0, DEFAULT_DIST_CONFIG) == len(DEFAULT_DIST_CONFIG.objects)


def test_write_csv(tmp_path):
    result = run_ed6_two_oracle(_tiny())
    out = write_csv(result, tmp_path / "ed6_two_oracle.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("mode,non_redundant_rate,")
    assert any(line.startswith("subtle,") for line in lines)


def test_config_round_trips():
    cfg = ED6TwoOracleConfig.from_dict({"noise": 0.4, "fallback": True, "n_steps": 12})
    assert cfg.noise == 0.4
    assert cfg.fallback is True
    assert cfg.n_steps == 12
