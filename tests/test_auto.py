"""Autoresearch-ratchet tests (SPEC-2 §17.5 automation).

  - the proposer mutates exactly one knob to a different value, deterministically;
  - the search is a deterministic function of (base, space, seed);
  - the ``best_score`` envelope is monotone non-decreasing (the keep-if-better
    ratchet never regresses);
  - a kept trial is exactly one that beat the running best.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")

from verisim.auto.search import MutationSpace, SearchConfig, propose, run_search
from verisim.experiments.e1 import E1Config


def _tiny_search() -> SearchConfig:
    return SearchConfig(
        base=E1Config(
            train_seeds=(0, 1),
            train_steps_per_traj=12,
            train_iters=30,
            n_layer=1,
            n_embd=32,
            difficulties={"low": "weighted"},
            eval_seeds=(100, 101),
            eval_steps=6,
        ),
        space=MutationSpace(
            knobs={"n_embd": (32, 64), "train_iters": (30, 60), "lr": (1e-3, 3e-3)}
        ),
        n_trials=5,
        search_seed=0,
    )


def test_propose_changes_exactly_one_knob():
    base = _tiny_search().base
    space = _tiny_search().space
    rng = random.Random(3)
    for _ in range(20):
        cand = propose(base, space, rng)
        changed = [k for k in space.knobs if getattr(cand, k) != getattr(base, k)]
        assert len(changed) == 1
        knob = changed[0]
        assert getattr(cand, knob) in space.knobs[knob]
        assert getattr(cand, knob) != getattr(base, knob)


def test_propose_is_identity_when_no_knob_can_change():
    base = _tiny_search().base
    frozen = MutationSpace(knobs={"n_embd": (base.n_embd,)})
    assert propose(base, frozen, random.Random(0)) == base


def test_search_emits_a_record_per_trial_plus_base():
    config = _tiny_search()
    records = run_search(config)
    assert len(records) == config.n_trials + 1  # trial 0 = base
    assert [int(r.config["trial"]) for r in records] == list(range(config.n_trials + 1))
    assert records[0].config["kept"] is True  # base seeds the ratchet


def test_search_is_deterministic():
    a = run_search(_tiny_search())
    b = run_search(_tiny_search())
    assert [r.config["score"] for r in a] == [r.config["score"] for r in b]
    assert [r.config["best_score"] for r in a] == [r.config["best_score"] for r in b]


def test_best_score_is_monotone_and_kept_matches_improvement():
    records = run_search(_tiny_search())
    best = [float(r.config["best_score"]) for r in records]
    assert all(best[i] <= best[i + 1] for i in range(len(best) - 1)), best
    # best_score always equals the max score seen through that trial.
    running = 0.0
    for i, r in enumerate(records):
        running = max(running, float(r.config["score"])) if i else float(r.config["score"])
        assert float(r.config["best_score"]) == running
        if i:
            improved = float(r.config["score"]) > float(records[i - 1].config["best_score"])
            assert bool(r.config["kept"]) == improved
