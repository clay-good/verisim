"""The oracle-as-SCM machinery: abduction, the three rungs, and the matched-coverage cut (SPEC-17
§2-3).

World-generic over a bundle that exposes ``make_actions(seed, n)`` (the seeded rollout -- the
exogenous
draw ``U``), ``oracle_step(s, a)`` (the structural equation ``F``), and ``diverge(a, b)`` (to check
bit-exactness and measure effects). The reference oracles satisfy the SCM contract literally, so:

  - **abduction** is reset+replay: re-running from the recorded seed recovers the exact ``U`` (the
    action sequence the seeded driver produced) and reproduces the factual trajectory bit-for-bit;
  - **rung 2** (intervention) is a one-step ``do(a')`` from a visited state;
  - **rung 3** (counterfactual) re-runs ``F`` forward holding the *other* exogenous draws (factual
    future actions) fixed, with one action overridden -- the true counterfactual the on-policy
    distribution cannot supply.

The added value of rung 3 over rung 2 is the **downstream** effect: where the world has a persistent
exogenous medium (the distributed partition/crash state), an intervention's effect amplifies as
``F``
re-runs forward; where it does not, rung 3 ≈ rung 2 (SPEC-17 §3, the do-calculus reading of H5).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

S = TypeVar("S")  # a world state
A = TypeVar("A")  # a world action

MakeActions = Callable[[int, int], "tuple[S, list[A]]"]
OracleStep = Callable[[S, A], S]
Diverge = Callable[[S, S], float]


@dataclass(frozen=True)
class Intervention(Generic[A]):
    """A ``do(X=x)``: override the action at step ``t`` with ``alt_action`` (SPEC-17 §3)."""

    t: int
    alt_action: A


def abduct_and_replay(
    make_actions: MakeActions[S, A], oracle_step: OracleStep[S, A], seed: int, n_steps: int
) -> tuple[S, list[A], list[S]]:
    """Abduct ``U`` from ``seed`` and replay ``F`` -- factual ``(s0, actions, states)`` (rung 1).

    ``states[i]`` is the state after the first ``i`` actions, so ``states[0] == s0`` and
    ``len(states) == n_steps + 1``. Because the oracle is deterministic given ``U`` (the seed), this
    is
    the exact recovery the SCM contract guarantees -- abduction as an ``O(1)`` lookup.
    """
    s0, actions = make_actions(seed, n_steps)
    states = [s0]
    state = s0
    for action in actions:
        state = oracle_step(state, action)
        states.append(state)
    return s0, actions, states


def abduction_exact(
    make_actions: MakeActions[S, A],
    oracle_step: OracleStep[S, A],
    diverge: Diverge[S],
    seed: int,
    n_steps: int,
) -> bool:
    """Whether two recoveries of ``U`` reproduce the factual trajectory bit-for-bit (H60).

    The empirical SCM-contract check: if abduction is exact, re-running from the recovered ``U`` is
    identical every time (divergence ``0`` at every step). A ``False`` here names seed-incomplete
    nondeterminism -- itself the finding that bounds which worlds admit an exact rung 3.
    """
    _, _, a_states = abduct_and_replay(make_actions, oracle_step, seed, n_steps)
    _, _, b_states = abduct_and_replay(make_actions, oracle_step, seed, n_steps)
    if len(a_states) != len(b_states):
        return False
    return all(diverge(x, y) == 0.0 for x, y in zip(a_states, b_states, strict=True))


def rung2_branch(oracle_step: OracleStep[S, A], state: S, alt_action: A) -> S:
    """Rung 2 -- the one-step intervention ``do(a')`` from a visited state (the shipped generator).
    """
    return oracle_step(state, alt_action)


def rung3_counterfactual(
    oracle_step: OracleStep[S, A],
    s0: S,
    actions: Sequence[A],
    intervention: Intervention[A],
) -> list[S]:
    """Rung 3 -- re-run ``F`` from ``s0`` with the factual future held fixed, one overridden.

    Abduction-action-prediction: the exogenous ``U`` (the factual action sequence) is held fixed
    except
    at ``intervention.t``, where ``do(alt_action)`` is applied; ``F`` is re-run forward. Returns the
    counterfactual trajectory ``[s0, …]`` (length ``len(actions) + 1``) -- the *true*
    counterfactual,
    not a one-step branch.
    """
    states = [s0]
    state = s0
    for i, action in enumerate(actions):
        used = intervention.alt_action if i == intervention.t else action
        state = oracle_step(state, used)
        states.append(state)
    return states


def downstream_amplification(
    oracle_step: OracleStep[S, A],
    diverge: Diverge[S],
    s0: S,
    actions: Sequence[A],
    factual_states: Sequence[S],
    intervention: Intervention[A],
) -> tuple[float, float]:
    """The (immediate, downstream) effect of an intervention -- rung 3 vs rung 2 (SPEC-17 §3).

    ``immediate`` = the one-step (rung-2) divergence the intervention causes at step ``t+1``;
    ``downstream`` = the terminal (rung-3) divergence after re-running ``F`` forward with the future
    held fixed. ``downstream > immediate`` means the effect amplifies through a persistent exogenous
    medium -- the signal that abduction (rung 3) is load-bearing on this world.
    """
    t = intervention.t
    immediate = diverge(
        factual_states[t + 1], rung2_branch(oracle_step, factual_states[t], intervention.alt_action)
    )
    cf_states = rung3_counterfactual(oracle_step, s0, actions, intervention)
    downstream = diverge(factual_states[-1], cf_states[-1])
    return immediate, downstream
