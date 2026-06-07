"""ED12 (partial-observation, learned arm) — the projections on the real M_θ, SPEC-7 §5.4.

The smoke instance of the learned partial-observation apparatus (torch extra): train a tiny flat
M_θ and check the deliverables are well-formed and structurally correct — the free-running
``bit ≤ observable`` dominance holds on every rollout, and the teacher-forced per-step rates order
``bit ≤ observable`` and ``bit ≤ consistency`` (a bit-correct step is correct under every view, by
construction). The substantive magnitudes come from the committed config run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from verisim.experiments.ed12_learned import (
    ED12LearnedConfig,
    ED12LearnedResult,
    run_ed12_learned,
    write_csv,
)


def _tiny() -> ED12LearnedConfig:
    return ED12LearnedConfig(
        train_seeds=(0, 1),
        train_steps_per_traj=16,
        train_iters=40,
        n_layer=1,
        n_embd=32,
        block_size=384,
        eval_seeds=(100, 101),
        n_steps=12,
    )


def test_run_is_well_formed():
    result = run_ed12_learned(_tiny())
    h = result.horizons
    assert {"bit_h", "obs_h", "cons_h", "observable_dominates_bit"} <= set(h)
    assert {"bit", "observable", "consistency", "steps"} <= set(result.accuracy)


def test_structural_dominance_and_projection_bounds():
    result = run_ed12_learned(_tiny())
    h = result.horizons
    # a bit-faithful step is necessarily observably faithful — holds every rollout, by construction
    assert h["observable_dominates_bit"] is True
    assert h["bit_h"] <= h["obs_h"] + 1e-9  # the observable horizon dominates the bit horizon
    # teacher-forced: a *bit*-correct step is correct under both other projections (all facts match
    # ⇒ the observable subset matches and the consistency view matches), so bit lower-bounds
    # both. (observable vs consistency are incomparable projections — neither implies the other — so
    # their *ordering* is an empirical finding, not asserted here.)
    a = result.accuracy
    assert a["bit"] <= a["observable"] + 1e-9
    assert a["bit"] <= a["consistency"] + 1e-9


def test_write_csv(tmp_path):
    result = run_ed12_learned(_tiny())
    out = write_csv(result, tmp_path / "ed12_learned.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0].startswith("panel,")
    assert any(line.startswith("horizon,") for line in lines)
    assert any(line.startswith("accuracy,") for line in lines)


def test_config_round_trips():
    cfg = ED12LearnedConfig.from_dict({"train_iters": 50, "n_steps": 10})
    assert cfg.train_iters == 50
    assert cfg.n_steps == 10
    assert isinstance(run_ed12_learned(_tiny()), ED12LearnedResult)
