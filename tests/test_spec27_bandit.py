"""SPEC-27 step 2 -- the BanditFuzz baseline learns where the hole lives.

The step-2 gate from ``plans/SPEC-27-honest-evaluation.md``: on a planted-hole target the bandit
must *converge to the hole-bearing constructs* (else it is not a competent baseline and beating it
proves nothing). Target: the RA18 resolver with the printf-format-escape hole OPEN
(``sound_printf=False``), whose unique reward-bearing mechanism is ``printf_fmt``. Also pins the
compute-parity accounting (``oracle_queries``) and determinism the SPEC-27 sweep relies on.
"""

from __future__ import annotations

from verisim.audit.bandit import BanditProposer
from verisim.audit.monitors import ResolverMonitor
from verisim.audit.oracles import ShellPathOracle

PREFIX = "/etc"
SHADOW = "/etc/shadow"
WORK = "/home/work"


def _bandit(seed: int = 0, reward_fn=None) -> BanditProposer:
    return BanditProposer(
        ResolverMonitor(PREFIX, sound_printf=False), ShellPathOracle(PREFIX),
        SHADOW, PREFIX, WORK, seed=seed, reward_fn=reward_fn,
    )


def test_bandit_converges_to_hole_mechanism() -> None:
    """printf_fmt (the one reward-bearing construct) must end up the top-ranked mechanism."""
    b = _bandit()
    list(b.propose(400))  # drive the posteriors
    means = b.mech_means()
    top = max(means, key=means.__getitem__)
    assert top == "printf_fmt", f"bandit ranked {top} over printf_fmt: {means}"
    assert means["printf_fmt"] > 0.9
    # and it is a clear winner, not a tie with the next construct.
    ranked = sorted(means.values(), reverse=True)
    assert ranked[0] - ranked[1] > 0.15


def test_bandit_charges_itself_for_oracle_calls() -> None:
    """Default reward spends one oracle call per proposal -- the compute-parity cost the SPEC-27
    sweep must charge it (as it charges the neural arm its training budget)."""
    b = _bandit()
    list(b.propose(250))
    assert b.oracle_queries == 250


def test_bandit_reward_fn_override_defers_accounting() -> None:
    """An injected reward_fn means the caller owns the oracle-call accounting; bandit counts 0."""
    calls = {"n": 0}

    def reward(_action) -> float:
        calls["n"] += 1
        return 0.0

    b = _bandit(reward_fn=reward)
    list(b.propose(50))
    assert b.oracle_queries == 0
    assert calls["n"] == 50


def test_bandit_deterministic_under_seed() -> None:
    a = _bandit(seed=3)
    b = _bandit(seed=3)
    sa = [x.command for x in a.propose(200)]
    sb = [x.command for x in b.propose(200)]
    assert sa == sb
    # different seed -> a different exploration path (else the seed is dead).
    c = [x.command for x in _bandit(seed=4).propose(200)]
    assert c != sa
