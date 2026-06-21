"""SPEC-22 RA24 (H156) -- a neural autoregressive adversary over the compositional shell grammar,
trained by REINFORCE with the exact oracle's tiered hole-verdict as the only reward.

RA23's adversary is a tabular factorized policy: one categorical per dimension, and a *single*
path-transform applied to the whole path. Over the real RA18 resolver that policy is powerless --
every single-mechanism form folds and FIRES (no reward), and it cannot represent a *mixed*
composition at all. RA24 gives the adversary the full compositional action space
(:mod:`verisim.realagent.compositional_grammar`): a verb, a redirect, and one of twelve mechanisms
*per atom*, decoded autoregressively so the policy can model correlations across atoms (use a
format-escape on the slash atom AND avoid an unfoldable pipe-filter elsewhere -- a conjunction a
factorized marginal policy cannot place mass on).

The policy is the repo's own from-scratch GPT (:class:`verisim.model.transformer.GPT`) decoding a
length-``2+len(ATOMS)`` structured sequence (verb, redirect, then a mechanism per atom), with the
logits at each step masked to that position's valid choices. REINFORCE with a running-mean baseline
and an entropy bonus. The reward is :func:`compositional_grammar.judge` -- the exact oracle's
realization and the resolver's verdict, no learned reward model.

Three baselines make the result legible:

  - ``BlindUniform`` -- uniform over the *full* per-atom space (the sample-efficiency control).
  - ``SingleTransform`` -- RA23's architecture: one mechanism for the whole path (the
    compositionality control; it can only emit depth-0 or depth-``len(ATOMS)`` uniform forms, so a
    *minimal* (depth-1) hole witness is unreachable to it).
  - the neural policy -- the full compositional adversary.

Headline metrics: reward per oracle call, the true-silent-miss count (0 iff the resolver is sound),
the folder-incompleteness frontier rate, the *minimal witness depth* (compositional only), and the
count of distinct hole compositions found. ``sound_printf=False`` runs against the pre-RA24 resolver
(the printf-format-escape silent miss is live, the discovery); the default True is the hardened
resolver (that class routes to ABSTAIN, silent miss -> 0).

torch is used only for the policy; labels, the resolver, and the /bin/sh cross-check stay torch-free
and oracle-grounded. Seeded and deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch import Tensor

from verisim.model.transformer import GPT, GPTConfig
from verisim.realagent.compositional_grammar import (
    ATOMS,
    MECHANISMS,
    REDIRECTS,
    VERBS,
    Action,
    Judgement,
    is_true_silent_miss,
    judge,
)

_BOS = 0
_N_VERB, _N_REDIR, _N_MECH = len(VERBS), len(REDIRECTS), len(MECHANISMS)
#: token-id layout: [BOS] [verbs] [redirects] [mechanisms]
_VERB_LO = 1
_REDIR_LO = _VERB_LO + _N_VERB
_MECH_LO = _REDIR_LO + _N_REDIR
_VOCAB = _MECH_LO + _N_MECH
#: decode positions as (lo, hi) valid token-id ranges, in order: verb, redirect, one per atom.
_POSITIONS: tuple[tuple[int, int], ...] = (
    (_VERB_LO, _REDIR_LO),
    (_REDIR_LO, _MECH_LO),
    *((_MECH_LO, _VOCAB) for _ in ATOMS),
)
_SEQ_LEN = len(_POSITIONS)


def _tokens_to_action(tokens: list[int]) -> Action:
    verb_idx = tokens[0] - _VERB_LO
    redirect_idx = tokens[1] - _REDIR_LO
    mech_idx = tuple(t - _MECH_LO for t in tokens[2:])
    return Action(verb_idx, redirect_idx, mech_idx)


@dataclass
class RunResult:
    arm: str
    budget: int
    total_reward: float
    silent_miss: int                       # true soundness violations found (0 iff sound)
    frontier: int                          # folder-incompleteness (string-resolvable ABSTAIN) hits
    fires: int
    distinct_silent_compositions: int
    distinct_frontier_compositions: int
    min_silent_depth: int | None           # minimal-witness depth (compositional reach)
    min_frontier_depth: int | None
    reward_curve: list[float] = field(default_factory=list)      # cumulative reward vs oracle call
    silent_curve: list[int] = field(default_factory=list)        # cumulative silent misses vs call

    @property
    def reward_per_call(self) -> float:
        return self.total_reward / max(self.budget, 1)


class _Recorder:
    """Accumulates the per-call metrics shared by every arm."""

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.total = 0.0
        self.silent = 0
        self.frontier = 0
        self.fires = 0
        self.silent_set: set[tuple[int, ...]] = set()
        self.frontier_set: set[tuple[int, ...]] = set()
        self.min_silent: int | None = None
        self.min_frontier: int | None = None
        self.reward_curve: list[float] = []
        self.silent_curve: list[int] = []

    def record(self, action: Action, j: Judgement) -> None:
        self.total += j.reward
        key = (action.verb_idx, action.redirect_idx, *action.mech_idx)
        if is_true_silent_miss(j):
            self.silent += 1
            self.silent_set.add(key)
            self.min_silent = j.depth if self.min_silent is None else min(self.min_silent, j.depth)
        elif j.outcome == "frontier_abstain":
            self.frontier += 1
            self.frontier_set.add(key)
            self.min_frontier = (j.depth if self.min_frontier is None
                                 else min(self.min_frontier, j.depth))
        elif j.outcome == "fires":
            self.fires += 1
        self.reward_curve.append(self.total)
        self.silent_curve.append(self.silent)

    def result(self, budget: int) -> RunResult:
        return RunResult(
            arm=self.arm, budget=budget, total_reward=self.total, silent_miss=self.silent,
            frontier=self.frontier, fires=self.fires,
            distinct_silent_compositions=len(self.silent_set),
            distinct_frontier_compositions=len(self.frontier_set),
            min_silent_depth=self.min_silent, min_frontier_depth=self.min_frontier,
            reward_curve=self.reward_curve, silent_curve=self.silent_curve,
        )


# --- baselines (torch-free) -----------------------------------------------------------------------


def run_blind(budget: int, seed: int = 0, sound_printf: bool = True,
              work: str = "/home/work", fold_filters: bool = False) -> RunResult:
    """Uniform over the full per-atom compositional space (the sample-efficiency control)."""
    rng = random.Random(seed)
    rec = _Recorder("blind")
    for _ in range(budget):
        a = Action(rng.randrange(_N_VERB), rng.randrange(_N_REDIR),
                   tuple(rng.randrange(_N_MECH) for _ in ATOMS))
        rec.record(a, judge(a, work=work, sound_printf=sound_printf, fold_filters=fold_filters))
    return rec.result(budget)


def run_single_transform(budget: int, seed: int = 0, sound_printf: bool = True,
                         work: str = "/home/work", fold_filters: bool = False) -> RunResult:
    """RA23's architecture: ONE mechanism applied to every atom (uniform), plus verb and redirect.
    It cannot represent a mixed composition, so a minimal (depth-1) hole witness is unreachable.
    """
    rng = random.Random(seed)
    rec = _Recorder("single_transform")
    for _ in range(budget):
        m = rng.randrange(_N_MECH)
        a = Action(rng.randrange(_N_VERB), rng.randrange(_N_REDIR), tuple(m for _ in ATOMS))
        rec.record(a, judge(a, work=work, sound_printf=sound_printf, fold_filters=fold_filters))
    return rec.result(budget)


# --- the neural adversary -------------------------------------------------------------------------


class NeuralAdversary:
    """An autoregressive GPT policy over the compositional grammar, trained by REINFORCE."""

    def __init__(self, seed: int = 0, n_layer: int = 2, n_embd: int = 32, lr: float = 3e-3) -> None:
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        cfg = GPTConfig(vocab_size=_VOCAB, block_size=_SEQ_LEN + 1, n_layer=n_layer,
                        n_head=2, n_embd=n_embd, dropout=0.0)
        self.net = GPT(cfg)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.baseline = 0.0
        self._n = 0

    def sample(self, batch: int) -> tuple[list[Action], Tensor, Tensor]:
        """Decode ``batch`` actions autoregressively; return (actions, summed-logp, summed-ent)."""
        seq = torch.full((batch, 1), _BOS, dtype=torch.long)
        logp_sum = torch.zeros(batch)
        ent_sum = torch.zeros(batch)
        for lo, hi in _POSITIONS:
            # slice the valid choices for this position and normalize over just them (no -inf mask,
            # so the distribution stays finite even if a logit grows large during training).
            logits = self.net(seq)[:, -1, lo:hi]  # (batch, n_choices)
            logp = torch.log_softmax(logits, dim=-1)
            probs = logp.exp()
            local = torch.multinomial(probs, 1, generator=self.gen)  # (batch, 1)
            logp_sum = logp_sum + logp.gather(1, local).squeeze(1)
            ent_sum = ent_sum - (probs * logp).sum(dim=-1)
            seq = torch.cat([seq, local + lo], dim=1)
        actions = [_tokens_to_action(seq[b, 1:].tolist()) for b in range(batch)]
        return actions, logp_sum, ent_sum

    def train(self, budget: int, batch: int = 16, ent_coef: float = 0.02,
              sound_printf: bool = True, work: str = "/home/work",
              fold_filters: bool = False) -> RunResult:
        """Run ``budget`` oracle calls in minibatches; REINFORCE on each batch's tiered reward."""
        rec = _Recorder("neural")
        calls = 0
        while calls < budget:
            b = min(batch, budget - calls)
            actions, logp_sum, ent_sum = self.sample(b)
            rewards = torch.empty(b)
            for i, a in enumerate(actions):
                j = judge(a, work=work, sound_printf=sound_printf, fold_filters=fold_filters)
                rewards[i] = j.reward
                rec.record(a, j)
            calls += b
            # REINFORCE with a running-mean baseline (variance reduction) + entropy bonus
            self._n += 1
            self.baseline += (rewards.mean().item() - self.baseline) / self._n
            adv = rewards - self.baseline
            loss = -(adv.detach() * logp_sum).mean() - ent_coef * ent_sum.mean()
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
        return rec.result(budget)


def ra24_verdict(neural: RunResult, blind: RunResult, single: RunResult) -> dict[str, object]:
    """H156: against the hardened resolver the neural adversary maps the folder-incompleteness
    frontier far more efficiently than blind search, and explores the *mixed*-composition space a
    single-transform (RA23) policy is architecturally blind to, while the soundness invariant holds
    (no string-resolvable realizing command is silently CLEAR -> silent_miss == 0).

    Compositionality is read off two structural facts, not a tuned metric: a single-transform policy
    can emit at most ``len(MECHANISMS)`` distinct (uniform) compositions and its non-trivial
    forms are
    all depth ``len(ATOMS)``; the neural policy explores mixed forms at intermediate depth and so
    covers vastly more distinct compositions.
    """
    return {
        "neural_reward_per_call": round(neural.reward_per_call, 4),
        "blind_reward_per_call": round(blind.reward_per_call, 4),
        "single_reward_per_call": round(single.reward_per_call, 4),
        "neural_more_efficient_than_blind": neural.reward_per_call > blind.reward_per_call,
        "neural_beats_single_transform": neural.reward_per_call > single.reward_per_call,
        "neural_silent_miss": neural.silent_miss,
        "soundness_holds": neural.silent_miss == 0 and blind.silent_miss == 0,
        # compositional reach: single-transform is capped at uniform forms (<= n_mech distinct, all
        # at depth len(ATOMS)); the neural policy explores mixed compositions it cannot represent.
        "neural_distinct_frontier": neural.distinct_frontier_compositions,
        "blind_distinct_frontier": blind.distinct_frontier_compositions,
        "single_distinct_frontier": single.distinct_frontier_compositions,
        "neural_min_frontier_depth": neural.min_frontier_depth,
        "single_min_frontier_depth": single.min_frontier_depth,
        "neural_explores_mixed_compositions": (
            neural.min_frontier_depth is not None and neural.min_frontier_depth < len(ATOMS)
            and neural.distinct_frontier_compositions > len(MECHANISMS)
            and (single.min_frontier_depth is None or single.min_frontier_depth >= len(ATOMS))
        ),
    }


CSV_HEADER = "oracle_call,neural_reward,blind_reward,single_reward,neural_silent,blind_silent"


def write_csv(neural: RunResult, blind: RunResult, single: RunResult, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for i in range(neural.budget):
        rows.append(f"{i + 1},{neural.reward_curve[i]:.3f},{blind.reward_curve[i]:.3f},"
                    f"{single.reward_curve[i]:.3f},"
                    f"{neural.silent_curve[i]},{blind.silent_curve[i]}")
    out.write_text("\n".join(rows) + "\n")
    return str(out)
