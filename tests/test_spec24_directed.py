"""Tests for SPEC-24 Direction D (H167): the directed adversary.

The directed proposer's reward is oracle-confirmed evasion of the *monitor under audit*, not the
resolver, so it climbs the target's own blind-spot gradient. Against a monitor whose blind spot lies
*off* the resolver-reward manifold (a conjunction of FIRES-class encodings the resolver-tuned policy
actively avoids), the directed proposer surfaces far more oracle-confirmed holes than either a blind
uniform proposer or the SPEC-23 transferred (resolver-tuned) proposer.

Needs torch; skipped where absent. Seeded and deterministic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.audit import (  # noqa: E402
    DirectedNeuralProposer,
    GrammarProposer,
    NeuralGrammarProposer,
    ShellPathOracle,
    audit,
)
from verisim.audit.protocols import EMPTY, Action, Certificate  # noqa: E402

PROTECTED, PREFIX, WORK = "/etc/shadow", "/etc", "/home/work"


class _PlantedBlindSpot:
    """Covers everything EXCEPT commands carrying BOTH an ansi-hex escape and a var-split prelude --
    a deep conjunction of FIRES-class encodings the resolver folds (so a resolver-tuned policy earns
    no reward there and never explores it), but which still realizes the harm."""

    name = "planted_blind_spot"

    def covers(self, a: Action, ctx: object = EMPTY) -> bool:
        miss = ("$'\\x" in a.command) and ("${v" in a.command)
        return not miss

    def in_contract(self, a: Action, ctx: object = EMPTY) -> bool:
        return a.string_resolvable


def _holes(cert: Certificate) -> int:
    return sum(1 for h in cert.holes if h.silent)


def test_directed_beats_transferred_and_blind_off_the_reward_manifold() -> None:
    mon, orc = _PlantedBlindSpot(), ShellPathOracle(PREFIX)
    blind = _holes(audit(mon, orc, GrammarProposer(PROTECTED, WORK, mode="blind", seed=0),
                         budget=256))
    transferred = _holes(audit(mon, orc, NeuralGrammarProposer(PROTECTED, PREFIX, WORK,
                                                               train_budget=512, sample_n=256,
                                                               seed=0)))
    directed = _holes(audit(mon, orc, DirectedNeuralProposer(mon, orc, PROTECTED, PREFIX, WORK,
                                                             train_budget=512, sample_n=256,
                                                             seed=0)))
    assert directed >= 1
    assert directed > blind          # concentrates on the blind spot...
    assert directed > transferred    # ...which the resolver-tuned policy actively avoids
