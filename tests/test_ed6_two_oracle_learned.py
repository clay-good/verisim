"""ED6 two-oracle (learned arm) — H12 on the real `M_θ` (SPEC-7 §10.1, DS8).

The smoke instance of the learned-model two-oracle apparatus (torch extra): train a tiny flat
`M_θ`, run the teacher-forced H12 measurement on it, and check the deliverables are well-formed and
structurally correct. The load-bearing fact is **structural, not noisy** and so is asserted: the
consistency oracle is **non-redundant-rate 0 by construction** — the consistency view is a pure
function of the replica state, so a bit-exact-correct prediction is always consistency-correct,
exactly as in the synthetic arm. The decision-sufficiency magnitude is a quantitative finding the
committed run gives; here we only check it is a well-formed conditional rate and that the consult is
genuinely cheaper than the full state.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.ed6_two_oracle_learned import (
    ED6TwoOracleLearnedConfig,
    ED6TwoOracleLearnedResult,
    run_ed6_two_oracle_learned,
    write_csv,
)


def _tiny() -> ED6TwoOracleLearnedConfig:
    return ED6TwoOracleLearnedConfig(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=40,
        n_layer=1,
        n_embd=32,
        block_size=384,
        eval_seeds=(100, 101, 102),
        n_steps=16,
    )


def test_run_is_well_formed():
    result = run_ed6_two_oracle_learned(_tiny())
    assert isinstance(result, ED6TwoOracleLearnedResult)
    c = result.cell
    for key in ("non_redundant_rate", "consistency_sufficient_rate", "full_wrong_rate",
                "consult_fact_ratio"):
        assert 0.0 <= c[key] <= 1.0
        assert c[f"{key}_lo"] <= c[key] <= c[f"{key}_hi"] or c[f"{key}_lo"] == c[f"{key}_hi"]


def test_non_redundant_is_zero_by_construction():
    # the consistency view is a pure function of the replica state, so a bit-exact-correct
    # prediction is always consistency-correct: the cheap oracle catches nothing the full one drops.
    result = run_ed6_two_oracle_learned(_tiny())
    assert result.cell["non_redundant_rate"] == 0.0
    assert result.cell["redundant_for_verification"] is True
    assert result.verdict["redundant_for_verification"] is True


def test_consult_is_cheaper_than_full_state():
    # the consistency answer is a strict subset of the full-state facts under fault, so the consult
    # is materially cheaper (the in-flight medium + partition structure never enter it).
    result = run_ed6_two_oracle_learned(_tiny())
    assert 0.0 < result.cell["consult_fact_ratio"] < 1.0
    assert result.verdict["cheaper_factor"] > 1.0


def test_is_deterministic():
    a = run_ed6_two_oracle_learned(_tiny())
    b = run_ed6_two_oracle_learned(_tiny())
    assert a.cell["consistency_sufficient_rate"] == b.cell["consistency_sufficient_rate"]
    assert a.cell["full_wrong_rate"] == b.cell["full_wrong_rate"]
    assert a.cell["consult_fact_ratio"] == b.cell["consult_fact_ratio"]


def test_write_csv(tmp_path):
    result = run_ed6_two_oracle_learned(_tiny())
    out = write_csv(result, tmp_path / "ed6_two_oracle_learned.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("non_redundant_rate,")
    assert len(lines) == 2  # header + one real-model row


def test_config_round_trips():
    cfg = ED6TwoOracleLearnedConfig.from_dict(
        {"train_iters": 20, "eval_driver": "adversarial", "n_steps": 12}
    )
    assert cfg.train_iters == 20
    assert cfg.eval_driver == "adversarial"
    assert cfg.n_steps == 12
