"""Invariants for the oracle-as-SCM machinery (SPEC-17 §2-3).

Structural, not magnitude (the macOS-first principle): abduction is bit-exact (the SCM contract);
rung-3 holds the future fixed except the intervention; an intervention's effect is computed from the
true counterfactual; and the downstream effect amplifies on a world with a persistent medium.
"""

from __future__ import annotations

from verisim.causal.scm import (
    Intervention,
    abduct_and_replay,
    abduction_exact,
    downstream_amplification,
    rung2_branch,
    rung3_counterfactual,
)
from verisim.experiments.sr_common import net_world


def _net():
    return net_world()


def test_abduction_is_bit_exact() -> None:
    w = _net()
    for seed in range(5):
        assert abduction_exact(w.make_actions, w.oracle_step, w.diverge, seed, 30)


def test_replay_reproduces_states() -> None:
    w = _net()
    s0, actions, states = abduct_and_replay(w.make_actions, w.oracle_step, 0, 20)
    assert len(states) == len(actions) + 1
    assert w.diverge(states[0], s0) == 0.0
    # Re-applying the actions matches the recorded states (determinism of F).
    state = s0
    for i, a in enumerate(actions):
        state = w.oracle_step(state, a)
        assert w.diverge(state, states[i + 1]) == 0.0


def test_rung3_holds_future_fixed_except_intervention() -> None:
    w = _net()
    s0, actions, states = abduct_and_replay(w.make_actions, w.oracle_step, 1, 16)
    # Intervening with the *factual* action changes nothing (counterfactual == factual).
    noop = Intervention(t=8, alt_action=actions[8])
    cf = rung3_counterfactual(w.oracle_step, s0, actions, noop)
    assert all(w.diverge(a, b) == 0.0 for a, b in zip(states, cf, strict=True))


def test_rung3_before_t_matches_factual() -> None:
    w = _net()
    s0, actions, states = abduct_and_replay(w.make_actions, w.oracle_step, 2, 16)
    alt = w.make_actions(99, 16)[1][8]  # some other action
    cf = rung3_counterfactual(w.oracle_step, s0, actions, Intervention(t=8, alt_action=alt))
    # States up to and including the intervention step's input are identical to the factual.
    for i in range(8 + 1):
        assert w.diverge(states[i], cf[i]) == 0.0


def test_rung2_is_one_step() -> None:
    w = _net()
    _, _actions, states = abduct_and_replay(w.make_actions, w.oracle_step, 3, 12)
    alt = w.make_actions(77, 12)[1][5]
    branch = rung2_branch(w.oracle_step, states[5], alt)
    assert w.diverge(branch, w.oracle_step(states[5], alt)) == 0.0


def test_downstream_amplification_nonnegative() -> None:
    w = _net()
    s0, actions, states = abduct_and_replay(w.make_actions, w.oracle_step, 4, 24)
    alt = w.make_actions(55, 24)[1][12]
    immediate, downstream = downstream_amplification(
        w.oracle_step, w.diverge, s0, actions, states, Intervention(t=12, alt_action=alt)
    )
    assert immediate >= 0.0 and downstream >= 0.0


def test_distributed_amplifies_more_than_network() -> None:
    # The persistent-medium world should show a larger downstream/immediate ratio than the network.
    from verisim.experiments.cx_common import _dist_cx, _net_cx

    def mean_amp(world) -> float:
        ratios = []
        for seed in range(8):
            s0, actions, states = abduct_and_replay(world.make_actions, world.oracle_step, seed, 30)
            alt = world.alt_action(states[15], 1000 + seed)
            imm, dn = downstream_amplification(
                world.oracle_step, world.diverge, s0, actions, states, Intervention(15, alt)
            )
            if imm > 0:
                ratios.append(dn / imm)
        return sum(ratios) / len(ratios) if ratios else 0.0

    assert mean_amp(_dist_cx()) > mean_amp(_net_cx())
