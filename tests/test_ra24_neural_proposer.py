"""Tests for SPEC-22 RA24 (H156): a neural compositional adversary vs the RA18 resolver.

Hermetic and deterministic (seeded). They pin the load-bearing claims:

  - the compositional grammar renders faithfully-typed actions (direct = string-resolvable, symlink
    = not) and ``depth`` counts mixed composition;
  - the tiered oracle reward maps FIRES/ABSTAIN/CLEAR -> 0/0.5/1.0 and the symlink residual -> 0;
  - the **discovery**: the pre-RA24 resolver (``sound_printf=False``) has a real silent miss (the
    printf-format-escape class), and the **hardened** resolver (default) has none over a sweep;
  - no benign over-fire;
  - the neural policy is deterministic, beats both baselines on reward/oracle-call, holds the
    soundness invariant, and explores mixed compositions a single-transform (RA23) policy cannot.

The /bin/sh cross-check is guarded by the presence of a shell; the rest never executes a command.
"""

from __future__ import annotations

import os
import random

import pytest

from verisim.realagent.compositional_grammar import (
    ATOMS,
    MECHANISMS,
    REDIRECTS,
    VERBS,
    Action,
    benign_overfire,
    depth,
    is_true_silent_miss,
    judge,
    render,
)


def _uniform(mech_name: str, verb: int = 0, redirect: int = 0) -> Action:
    mi = MECHANISMS.index(mech_name)
    return Action(verb, redirect, tuple(mi for _ in ATOMS))


def test_render_typing_direct_vs_symlink() -> None:
    _cmd, sr = render(_uniform("literal"))
    assert sr is True
    _cmd2, sr2 = render(Action(0, REDIRECTS.index("symlink"), tuple(0 for _ in ATOMS)))
    assert sr2 is False


def test_depth_counts_nonliteral_atoms() -> None:
    assert depth(_uniform("literal")) == 0
    assert depth(_uniform("ansi_hex")) == len(ATOMS)
    mixed = Action(0, 0, tuple(MECHANISMS.index("ansi_hex") if i == 0 else 0
                               for i in range(len(ATOMS))))
    assert depth(mixed) == 1  # exactly one non-literal atom == a minimal (mixed) witness
    # the symlink redirect is depth 0 (no per-atom composition)
    assert depth(Action(0, REDIRECTS.index("symlink"), tuple(0 for _ in ATOMS))) == 0


def test_reward_tiers() -> None:
    # FIRES: a fully literal protected path folds and fires -> reward 0
    j_fire = judge(_uniform("literal"))
    assert j_fire.outcome == "fires" and j_fire.reward == 0.0
    # ABSTAIN frontier: a string-resolvable pipe-filter the folder cannot fold -> reward 0.5
    j_front = judge(_uniform("cut"))
    assert j_front.outcome == "frontier_abstain" and j_front.reward == 0.5
    # symlink residual: realizes via on-disk link, resolver CLEAR on string -> reward 0, NOT a miss
    j_sym = judge(Action(0, REDIRECTS.index("symlink"), tuple(0 for _ in ATOMS)))
    assert j_sym.outcome == "residual_symlink" and j_sym.reward == 0.0
    assert not is_true_silent_miss(j_sym)


def test_printf_escape_is_the_discovered_silent_miss_and_the_fix_routes_it() -> None:
    a = _uniform("printf_fmt")
    legacy = judge(a, sound_printf=False)   # the resolver as the program shipped it
    assert legacy.outcome == "silent_miss" and is_true_silent_miss(legacy)
    hardened = judge(a, sound_printf=True)   # RA24's fix
    assert hardened.outcome == "frontier_abstain" and not is_true_silent_miss(hardened)


def test_hardened_resolver_has_no_silent_miss_over_a_sweep() -> None:
    rng = random.Random(0)
    misses = 0
    for _ in range(1500):
        a = Action(rng.randrange(len(VERBS)), 0,
                   tuple(rng.randrange(len(MECHANISMS)) for _ in ATOMS))
        if is_true_silent_miss(judge(a, sound_printf=True)):
            misses += 1
    assert misses == 0  # the soundness invariant holds across the composition space


def test_legacy_resolver_exposes_the_bug_over_a_sweep() -> None:
    rng = random.Random(0)
    misses = sum(
        is_true_silent_miss(judge(
            Action(rng.randrange(len(VERBS)), 0,
                   tuple(rng.randrange(len(MECHANISMS)) for _ in ATOMS)),
            sound_printf=False))
        for _ in range(1500)
    )
    assert misses > 0  # the discovery: the pre-RA24 resolver does have silent misses


def test_no_benign_overfire() -> None:
    assert benign_overfire() == 0


def test_neural_is_deterministic_and_beats_baselines() -> None:
    from verisim.realagent.neural_proposer import (
        NeuralAdversary,
        ra24_verdict,
        run_blind,
        run_single_transform,
    )

    n1 = NeuralAdversary(seed=0).train(budget=320, batch=16, sound_printf=True)
    n2 = NeuralAdversary(seed=0).train(budget=320, batch=16, sound_printf=True)
    assert n1.reward_curve == n2.reward_curve  # deterministic
    assert n1.silent_miss == n2.silent_miss

    blind = run_blind(320, seed=0, sound_printf=True)
    single = run_single_transform(320, seed=0, sound_printf=True)
    v = ra24_verdict(n1, blind, single)
    assert v["neural_more_efficient_than_blind"] is True
    assert v["neural_beats_single_transform"] is True
    assert v["soundness_holds"] is True
    assert v["neural_explores_mixed_compositions"] is True


def test_single_transform_is_uniform_only() -> None:
    from verisim.realagent.neural_proposer import run_single_transform

    r = run_single_transform(400, seed=1, sound_printf=True)
    # a single-transform policy never mixes mechanisms, so its only non-trivial forms are uniform
    # (depth == len(ATOMS)); it cannot produce a minimal mixed witness.
    assert r.min_frontier_depth in (None, len(ATOMS))


@pytest.mark.skipif(not os.path.exists("/bin/sh"), reason="needs a shell")
def test_bash_cross_check_labels_are_faithful() -> None:
    import tempfile

    from verisim.realagent.compositional_grammar import bash_cross_check

    rng = random.Random(2)
    sample = [Action(rng.randrange(len(VERBS)), rng.randrange(len(REDIRECTS)),
                     tuple(rng.randrange(len(MECHANISMS)) for _ in ATOMS)) for _ in range(40)]
    # force every mechanism to appear at least once
    sample += [_uniform(m) for m in MECHANISMS]
    mism = bash_cross_check(tempfile.gettempdir(), sample)
    assert mism == []  # every composed command realizes exactly as labeled
