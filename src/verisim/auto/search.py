"""The keep-if-better config ratchet (SPEC-2 §17.5 automation).

The loop, after Karpathy's ``autoresearch`` but oracle-gated:

    best <- base config; best_score <- score(base)
    for each trial:
        cand  <- propose(best)          # mutate one knob (seeded, deterministic)
        score <- score(cand)            # train + measure clean accuracy vs. ORACLE
        if score > best_score:          # "did we improve?" -- against ground truth
            best, best_score <- cand, score      # KEEP
        else:
            (roll back -- best is unchanged)     # REJECT

The score is the **mean clean (ρ=0) per-step teacher-forced accuracy** over the
config's difficulties x eval seeds -- verisim's single comparable scalar (its
"val_bpb"): higher is better, in ``[0, 1]``, and computed against the oracle's true
next state. ``H_ε`` is deliberately *not* the gate here: at the v0 scale it is ~0
everywhere (the H1 negative), a flat signal a ratchet cannot climb; clean accuracy
is the smooth, comparable signal (the same reasoning that makes ``autoresearch``
pick val_bpb over a sparse metric).

Everything is a function of ``(base config, mutation space, search seed)``: the
proposer is seeded, and each trial's score is deterministic given its config
(``train_model`` seeds torch and single-threads, SPEC-2 §12). One :class:`RunRecord`
is emitted per trial (the ``program.md``/leaderboard analogue), so the ratchet
trajectory is reproducible and plottable from records only (SPEC-2 §7.3).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.env.config import DEFAULT_CONFIG, EnvConfig
from verisim.env.state import State
from verisim.metrics.record import RunRecord, write_records
from verisim.model.vocab import Vocab
from verisim.model.world_model import NeuralWorldModel
from verisim.oracle.base import Oracle
from verisim.oracle.reference import ReferenceOracle

from ..experiments.e1 import E1Config, eval_actions, train_model
from ..experiments.e4 import teacher_forced_accuracy

# The §17.5 levers: model capacity, training budget, dataset size, learning rate.
# Each knob maps to the candidate values the proposer may set it to.
_DEFAULT_SPACE: dict[str, tuple[Any, ...]] = {
    "n_layer": (1, 2, 4),
    "n_embd": (32, 64, 128),
    "train_iters": (200, 400, 800),
    "train_steps_per_traj": (20, 40, 60),
    "lr": (1e-3, 3e-3, 1e-2),
}


@dataclass(frozen=True)
class MutationSpace:
    """The knobs the proposer may mutate and their candidate values."""

    knobs: dict[str, tuple[Any, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in _DEFAULT_SPACE.items()}
    )

    @staticmethod
    def from_dict(d: dict[str, list[Any]] | None) -> MutationSpace:
        if not d:
            return MutationSpace()
        return MutationSpace(knobs={k: tuple(v) for k, v in d.items()})


@dataclass(frozen=True)
class SearchConfig:
    name: str = "auto-search"
    base: E1Config = field(default_factory=E1Config)
    space: MutationSpace = field(default_factory=MutationSpace)
    n_trials: int = 12
    search_seed: int = 0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SearchConfig:
        base = SearchConfig()
        return SearchConfig(
            name=d.get("name", base.name),
            base=E1Config.from_dict(d.get("base", {})),
            space=MutationSpace.from_dict(d.get("space")),
            n_trials=d.get("n_trials", base.n_trials),
            search_seed=d.get("search_seed", base.search_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> SearchConfig:
        return SearchConfig.from_dict(json.loads(Path(path).read_text()))


def score_config(cfg: E1Config, vocab: Vocab, oracle: Oracle, env: EnvConfig) -> float:
    """The oracle-gated scalar: mean clean (ρ=0) per-step accuracy over all eval cells.

    Trains one model under ``cfg`` and measures, against the oracle's true next
    state, the fraction of steps whose predicted delta is exact -- averaged over the
    config's difficulties x eval seeds. Higher is better; deterministic given ``cfg``.
    """
    model = NeuralWorldModel(train_model(cfg, vocab, oracle, env), vocab)
    scores: list[float] = []
    for driver in cfg.difficulties.values():
        for seed in cfg.eval_seeds:
            actions = eval_actions(oracle, env, driver, seed, cfg.eval_steps)
            scores.append(teacher_forced_accuracy(model, oracle, State.empty(), actions))
    return fmean(scores) if scores else 0.0


def propose(best: E1Config, space: MutationSpace, rng: random.Random) -> E1Config:
    """Mutate exactly one knob of ``best`` to a *different* candidate value (seeded).

    A deterministic coordinate hill-climb step -- the reproducible analogue of an
    agent proposing one edit. Knobs whose only candidates equal the current value
    are skipped so every proposal is a real change.
    """
    current = {knob: getattr(best, knob) for knob in space.knobs}
    mutable = [
        knob
        for knob, candidates in space.knobs.items()
        if any(c != current[knob] for c in candidates)
    ]
    if not mutable:
        return best
    knob = rng.choice(sorted(mutable))
    choices = [c for c in space.knobs[knob] if c != current[knob]]
    return replace(best, **{knob: rng.choice(choices)})


def _trial_record(
    cfg: E1Config, *, trial: int, score: float, kept: bool, best_score: float, name: str
) -> RunRecord:
    return RunRecord(
        config={
            "experiment": name,
            "trial": trial,
            "score": score,
            "best_score": best_score,
            "kept": kept,
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "train_iters": cfg.train_iters,
            "train_steps_per_traj": cfg.train_steps_per_traj,
            "lr": cfg.lr,
        },
        seed=cfg.model_seed,
        epsilon=0.0,
        divergences=[],
    )


def run_search(
    config: SearchConfig | None = None, *, oracle: Oracle | None = None
) -> list[RunRecord]:
    """Run the keep-if-better ratchet; return one :class:`RunRecord` per trial.

    Trial 0 scores the base config (the starting ``best``); trials 1..N each propose
    a mutation, score it against the oracle, and keep it only if it beats the best.
    """
    config = config or SearchConfig()
    oracle = oracle or ReferenceOracle()
    env = DEFAULT_CONFIG
    vocab = Vocab(env)
    rng = random.Random(config.search_seed)

    best = config.base
    best_score = score_config(best, vocab, oracle, env)
    records = [
        _trial_record(
            best, trial=0, score=best_score, kept=True, best_score=best_score, name=config.name
        )
    ]

    for trial in range(1, config.n_trials + 1):
        candidate = propose(best, config.space, rng)
        score = score_config(candidate, vocab, oracle, env)
        kept = score > best_score
        if kept:
            best, best_score = candidate, score
        records.append(
            _trial_record(
                candidate,
                trial=trial,
                score=score,
                kept=kept,
                best_score=best_score,
                name=config.name,
            )
        )
    return records


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the autoresearch-style config ratchet (SPEC-2 §17.5 automation)."
    )
    parser.add_argument("--config", type=str, default=None, help="path to a search config")
    parser.add_argument("--out", type=str, default="runs/auto/log.jsonl")
    args = parser.parse_args()
    config = SearchConfig.from_json_file(args.config) if args.config else SearchConfig()
    records = run_search(config)
    path = write_records(records, args.out)
    kept = sum(bool(r.config["kept"]) for r in records[1:])
    best = max(float(r.config["score"]) for r in records)
    print(f"wrote {len(records)} trials to {path}; {kept} kept; best score {best:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
