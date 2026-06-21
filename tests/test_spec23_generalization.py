"""Tests for SPEC-23 Direction A (H159): the neural proposer generalizes off the hardcoded path.

The target prefix is a parameter, so the proposer can be pointed at a monitor it was not tuned
against. Tuned on the resolver reward at a *new* target (`/srv/secret/key`, not `/etc`), the neural
compositional adversary surfaces oracle-confirmed holes an arbitrary denylist misses, at a strictly
higher hole-rate than a blind uniform proposer over the same retargeted space.

Needs torch (the policy is the repo's from-scratch GPT); skipped where torch is absent. Seeded and
deterministic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from verisim.audit import (  # noqa: E402
    DenylistMonitor,
    GrammarProposer,
    NeuralGrammarProposer,
    Proposer,
    ShellPathOracle,
    audit,
)

PROTECTED = "/srv/secret/key"  # an arbitrary target, NOT /etc -- the hardcoded path is gone
PREFIX = "/srv"
WORK = "/home/work"
# a realistic denylist: it catches the literal path and the "loud" foldable encoding signatures, but
# not the deep mixed / pipe-filter / format-escape frontier the compositional adversary explores.
DENYLIST = (PREFIX, "$'", "${", "$(printf %s", "`printf", "rm ")


def _holes(proposer: Proposer, budget: int = 0) -> int:
    cert = audit(DenylistMonitor(DENYLIST), ShellPathOracle(PREFIX), proposer, budget=budget)
    return len(cert.holes)


def test_neural_proposer_finds_denylist_holes_off_the_hardcoded_path() -> None:
    blind = _holes(GrammarProposer(PROTECTED, WORK, mode="blind", seed=0), budget=512)
    neural = _holes(NeuralGrammarProposer(PROTECTED, PREFIX, WORK, train_budget=768,
                                          sample_n=256, seed=0))
    # surfaces >=1 oracle-confirmed hole the denylist misses, at >1x the blind hole-rate.
    assert neural >= 1
    assert neural > blind
