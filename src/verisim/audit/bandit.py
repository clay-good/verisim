"""SPEC-27 -- the BanditFuzz-style baseline proposer: the strong non-learned comparator RA23/24
never ran.

RA23/24 compared the learned (REINFORCE) proposer only against ``blind uniform``. Klees et al. and
the [SoK fuzzing](https://arxiv.org/pdf/2405.10220) review name that exact omission: a learned
fuzzer must beat a *competent adaptive* search, not just uniform random, or the speedup is an
artifact of a weak baseline. The canonical competent-but-not-deep baseline is
[BanditFuzz](https://uwspace.uwaterloo.ca/bitstreams/798ce0c9-aa29-416f-9ed4-0d968578cb02/download)
(Thompson sampling over grammar constructs, learning which are common among bug-revealing inputs).
This is that baseline, on this grammar.

It is *adaptive* but *not* a neural sequence model: an independent Beta posterior per construct
value (verb, redirect, and the shared per-atom mechanism), Thompson-sampled per action, updated
by the same oracle reward the learned proposer trains on (``realizes ∧ ¬covers`` against the monitor
under audit). No gradient, no torch, no hidden state across constructs -- the cheapest thing that
still *learns* where the holes are. If the neural proposer cannot beat *this*, the "learning helps"
claim does not hold (SPEC-27 §5).

Reward attribution is factorized (BanditFuzz's move): a rewarding action credits every construct it
used. ``oracle_queries`` is exposed so the SPEC-27 compute-parity axis can charge the bandit for the
oracle calls it spent learning, as it must charge the neural proposer for its training budget.
Torch-free, seeded, deterministic given the seed.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator

from verisim.realagent.compositional_grammar import (
    ATOMS,
    MECHANISMS,
    REDIRECTS,
    VERBS,
    split_atoms,
)
from verisim.realagent.compositional_grammar import (
    Action as GAction,
)

from .proposers import DEFAULT_PREFIX, DEFAULT_PROTECTED, DEFAULT_WORK, _to_audit
from .protocols import EMPTY, Action, Context, Monitor, Oracle, State


class _BetaArm:
    """One construct value's Beta(alpha, beta) posterior over P(reward). Conjugate, so a binary
    reward is a single increment; the Thompson draw is a Beta sample."""

    __slots__ = ("alpha", "beta")

    def __init__(self) -> None:
        self.alpha = 1.0  # uniform prior
        self.beta = 1.0

    def update(self, reward: float) -> None:
        self.alpha += reward
        self.beta += 1.0 - reward

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def _argmax_sample(arms: list[_BetaArm], rng: random.Random) -> int:
    """Thompson selection: draw one sample from each arm's posterior, pick the argmax."""
    best_i, best_v = 0, -1.0
    for i, arm in enumerate(arms):
        v = rng.betavariate(arm.alpha, arm.beta)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


class BanditProposer:
    """Factorized Thompson-sampling adversary over (verb, redirect, per-atom mechanism), reward =
    oracle-confirmed silent miss against the monitor under audit. The SPEC-27 strong baseline."""

    def __init__(
        self,
        monitor: Monitor,
        oracle: Oracle,
        protected_path: str = DEFAULT_PROTECTED,
        prefix: str = DEFAULT_PREFIX,
        work: str = DEFAULT_WORK,
        *,
        seed: int = 0,
        ctx: Context = EMPTY,
        state: State = EMPTY,
        reward_fn: Callable[[Action], float] | None = None,
        name: str = "bandit",
    ) -> None:
        self._monitor = monitor
        self._oracle = oracle
        self.prefix = prefix
        self.work = work
        self.atoms = split_atoms(protected_path, len(ATOMS))
        self.seed = seed
        self._ctx = ctx
        self._state = state
        self._reward_fn = reward_fn
        self.name = name
        self.oracle_queries = 0
        # one posterior per construct value; mechanisms are shared across the atom positions.
        self._verbs = [_BetaArm() for _ in VERBS]
        self._redirects = [_BetaArm() for _ in REDIRECTS]
        self._mechs = [_BetaArm() for _ in MECHANISMS]

    def _reward(self, action: Action) -> float:
        if self._reward_fn is not None:
            return self._reward_fn(action)
        self.oracle_queries += 1
        realizes = self._oracle.realizes(action, self._state)
        covered = self._monitor.covers(action, self._ctx)
        return 1.0 if (realizes and not covered) else 0.0

    def mech_means(self) -> dict[str, float]:
        """Posterior mean reward per mechanism -- the bandit's learned ranking of constructs.
        Exposed for the step-2 convergence test (does it concentrate on the hole-bearing one?)."""
        return {MECHANISMS[i]: arm.mean() for i, arm in enumerate(self._mechs)}

    def propose(self, budget: int = 0) -> Iterator[Action]:
        rng = random.Random(self.seed)
        n_atoms = len(ATOMS)
        for _ in range(budget):
            vi = _argmax_sample(self._verbs, rng)
            ri = _argmax_sample(self._redirects, rng)
            mech = tuple(_argmax_sample(self._mechs, rng) for _ in range(n_atoms))
            ga = GAction(vi, ri, mech)
            action = _to_audit(ga, self.work, self.atoms)
            r = self._reward(action)
            self._verbs[vi].update(r)
            self._redirects[ri].update(r)
            for mi in mech:
                self._mechs[mi].update(r)
            yield action
