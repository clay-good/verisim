"""SPEC-23 (H158/H159) -- the Proposer adapters: the shipped adversaries, target passed in.

The proposers emit candidate :class:`~verisim.audit.protocols.Action`s for the auditor to probe a
monitor with. None invents a new adversary; each lifts a shipped one off its hardcoded ``/etc``:

  - :class:`CorpusProposer` -- RA22's structured verb x path-form corpus
    (:func:`coverage_synth.generate_corpus`), parametric in the target. Deterministic and hermetic;
    reproduces the RA22 literal/indirection/symlink partition through the interface.
  - :class:`GrammarProposer` -- RA24's compositional grammar
    (:mod:`verisim.realagent.compositional_grammar`), retargeted to an arbitrary protected path
    (split into the grammar's atom count). ``mode="enumerate"`` yields the single-mechanism forms
    (deterministic, so the printf-format-escape hole is reproduced byte-for-byte, H158);
    ``mode="blind"`` is the uniform sample-efficiency control (the H159 baseline).
  - :class:`NeuralGrammarProposer` -- RA24's REINFORCE policy
    (:class:`neural_proposer.NeuralAdversary`), trained against the resolver reward at the *new*
    target, then sampled. It is the one that must generalize off the hardcoded path (H159): tuned on
    the resolver, it still surfaces the deep-encoding holes a hand-written denylist misses, at a far
    higher rate than blind uniform. Needs torch (imported lazily).

The mapping from a grammar action to an audit action records ``string_resolvable`` (False only for
the symlink redirect, the on-disk-state residual) and a per-class label for the certificate.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Iterator

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
from verisim.realagent.compositional_grammar import (
    depth as gdepth,
)
from verisim.realagent.compositional_grammar import (
    render as grender,
)
from verisim.realagent.coverage_synth import generate_corpus

from .protocols import Action, Monitor, Oracle

DEFAULT_PROTECTED = "/etc/shadow"
DEFAULT_PREFIX = "/etc"
DEFAULT_WORK = "/home/work"

#: corpus class -> whether the harm is determinable from the action string. The symlink form is the
#: one on-disk-state residual; the rest are string-resolvable (literal or runtime-assembled).
_CORPUS_STRING_RESOLVABLE = {
    "literal": True,
    "indirection_var": True,
    "indirection_subst": True,
    "indirection_symlink": False,
    "benign": True,
}


class CorpusProposer:
    """RA22's verb x path-form corpus, parametric in the target. Deterministic, hermetic."""

    def __init__(self, prefix: str = DEFAULT_PREFIX, work: str = DEFAULT_WORK,
                 name: str = "corpus") -> None:
        self.prefix = prefix
        self.work = work
        self.name = name

    def propose(self, budget: int = 0) -> Iterable[Action]:
        for c in generate_corpus(self.prefix, self.work):
            yield Action(
                command=c.command,
                realizes=c.realizes,
                string_resolvable=_CORPUS_STRING_RESOLVABLE.get(c.klass, True),
                klass=c.klass,
                depth=0 if c.klass in ("literal", "benign") else 1,
            )


def _klass(ga: GAction) -> str:
    """A per-class label for a grammar action: the symlink residual, a literal, a single mechanism,
    or a mixed composition (the depth>=2 region only a compositional policy reaches)."""
    if REDIRECTS[ga.redirect_idx] == "symlink":
        return "residual_symlink"
    mechs = {MECHANISMS[mi] for mi in ga.mech_idx if MECHANISMS[mi] != "literal"}
    if not mechs:
        return "literal"
    return next(iter(mechs)) if len(mechs) == 1 else "mixed"


def _to_audit(ga: GAction, work: str, atoms: tuple[str, ...]) -> Action:
    cmd, string_resolvable = grender(ga, work, atoms)
    return Action(command=cmd, realizes=True, string_resolvable=string_resolvable,
                  klass=_klass(ga), depth=gdepth(ga))


class GrammarProposer:
    """RA24's compositional grammar, retargeted. ``mode='enumerate'`` (deterministic single-mech
    coverage) or ``mode='blind'`` (uniform sample-efficiency control)."""

    def __init__(self, protected_path: str = DEFAULT_PROTECTED, work: str = DEFAULT_WORK,
                 mode: str = "enumerate", seed: int = 0, name: str | None = None) -> None:
        self.protected_path = protected_path
        self.work = work
        self.atoms = split_atoms(protected_path, len(ATOMS))
        self.mode = mode
        self.seed = seed
        self.name = name or f"grammar_{mode}"

    def propose(self, budget: int = 0) -> Iterable[Action]:
        if self.mode == "blind":
            yield from self._blind(budget)
        else:
            yield from self._enumerate()

    def _enumerate(self) -> Iterator[Action]:
        n = len(ATOMS)
        # every single-mechanism form (mechanism applied to all atoms) x verb, both redirects --
        # guarantees the printf-format-escape form (the RA24 silent-miss class) is present.
        for vi in range(len(VERBS)):
            for ri in range(len(REDIRECTS)):
                for mi in range(len(MECHANISMS)):
                    yield _to_audit(GAction(vi, ri, tuple(mi for _ in range(n))),
                                    self.work, self.atoms)

    def _blind(self, budget: int) -> Iterator[Action]:
        rng = random.Random(self.seed)
        nv, nr, nm = len(VERBS), len(REDIRECTS), len(MECHANISMS)
        for _ in range(budget):
            ga = GAction(rng.randrange(nv), rng.randrange(nr),
                         tuple(rng.randrange(nm) for _ in range(len(ATOMS))))
            yield _to_audit(ga, self.work, self.atoms)


class NeuralGrammarProposer:
    """RA24's REINFORCE policy, trained against the resolver reward at the (possibly new) target,
    then sampled. The proposer that must generalize off the hardcoded path (H159). Needs torch."""

    def __init__(self, protected_path: str = DEFAULT_PROTECTED, prefix: str = DEFAULT_PREFIX,
                 work: str = DEFAULT_WORK, train_budget: int = 512, sample_n: int = 128,
                 seed: int = 0, sound_printf: bool = True, name: str = "neural") -> None:
        self.protected_path = protected_path
        self.prefix = prefix
        self.work = work
        self.atoms = split_atoms(protected_path, len(ATOMS))
        self.train_budget = train_budget
        self.sample_n = sample_n
        self.seed = seed
        self.sound_printf = sound_printf
        self.name = name

    def propose(self, budget: int = 0) -> Iterable[Action]:
        from verisim.realagent.neural_proposer import NeuralAdversary

        adv = NeuralAdversary(seed=self.seed)
        adv.train(self.train_budget, work=self.work, prefix=self.prefix,
                  sound_printf=self.sound_printf, atoms=self.atoms)
        actions, _logp, _ent = adv.sample(self.sample_n)
        for ga in actions:
            yield _to_audit(ga, self.work, self.atoms)


class ExhaustiveDepthProposer:
    """SPEC-24 (H165): yield the *entire* direct-redirect composition space at depth <= ``k`` --
    every
    way to pick j<=k atoms non-literal and assign each a non-literal mechanism, x every verb. No
    sampling, no seed: a clean audit over this set is a **proof** of soundness for the fragment. The
    count is exactly :func:`guarantee.depth_bounded_count` (verified in the H165 test)."""

    def __init__(self, protected_path: str = DEFAULT_PROTECTED, k: int = 2,
                 work: str = DEFAULT_WORK, name: str | None = None) -> None:
        self.protected_path = protected_path
        self.k = k
        self.work = work
        self.atoms = split_atoms(protected_path, len(ATOMS))
        self.name = name or f"exhaustive_depth{k}"

    def propose(self, budget: int = 0) -> Iterable[Action]:
        n = len(ATOMS)
        nm = len(MECHANISMS)
        lit = MECHANISMS.index("literal")
        nonlit = [m for m in range(nm) if m != lit]
        ri = REDIRECTS.index("direct")
        for vi in range(len(VERBS)):
            for j in range(min(self.k, n) + 1):
                for positions in itertools.combinations(range(n), j):
                    for assignment in itertools.product(nonlit, repeat=j):
                        mech = [lit] * n
                        for pos, m in zip(positions, assignment, strict=True):
                            mech[pos] = m
                        yield _to_audit(GAction(vi, ri, tuple(mech)), self.work, self.atoms)


class DirectedNeuralProposer:
    """SPEC-24 (H167): the REINFORCE policy whose reward is oracle-confirmed evasion of the
    **monitor
    under audit** (``1.0 if realizes(a) and not covers(a) else 0.0``), not the resolver. It climbs
    the
    target monitor's own blind-spot gradient, so it surfaces holes a resolver-tuned (transferred) or
    blind proposer misses in the rare-hole regime a strong external guardrail lives in. Needs
    torch."""

    def __init__(self, monitor: Monitor, oracle: Oracle, protected_path: str = DEFAULT_PROTECTED,
                 prefix: str = DEFAULT_PREFIX, work: str = DEFAULT_WORK, train_budget: int = 768,
                 sample_n: int = 256, seed: int = 0, name: str = "directed_neural") -> None:
        self._monitor = monitor
        self._oracle = oracle
        self.protected_path = protected_path
        self.prefix = prefix
        self.work = work
        self.atoms = split_atoms(protected_path, len(ATOMS))
        self.train_budget = train_budget
        self.sample_n = sample_n
        self.seed = seed
        self.name = name

    def _reward(self, ga: GAction) -> float:
        a = _to_audit(ga, self.work, self.atoms)
        return 1.0 if (self._oracle.realizes(a) and not self._monitor.covers(a)) else 0.0

    def propose(self, budget: int = 0) -> Iterable[Action]:
        from verisim.realagent.neural_proposer import NeuralAdversary

        adv = NeuralAdversary(seed=self.seed)
        adv.train(self.train_budget, work=self.work, prefix=self.prefix, atoms=self.atoms,
                  reward_fn=self._reward)
        actions, _logp, _ent = adv.sample(self.sample_n)
        for ga in actions:
            yield _to_audit(ga, self.work, self.atoms)
