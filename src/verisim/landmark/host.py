"""The host-world landmark layer: privilege-graph planning (SPEC-12 §6 LP8, H38). Torch-free.

The third cross-world fork. The network landmark layer wires its graph on *reachability* (EN10), the
distributed twin on *consistency/partition* structure (ED12, [`dist`](./dist.py)); LP8-host asks
whether the *method* also transfers to the **host** world, whose security-relevant hidden state is
**privilege** (SPEC-6 §3.2: a non-root process gaining root is the escalation the defender cares
about). So the host landmark signature is the coarse **privilege/liveness class set** -- the set of
``(process state, uid)`` classes present (RUNNING/ZOMBIE × root/non-root), deliberately
*count-free*:
it drops the exact process population (every ``fork`` adds a pid and would make the signature track
the full process table, near-bit-exact) and keeps only *which* privilege/liveness classes exist --
the projection a privilege escalation or a process-death actually moves.

This is the honest test of H38's refutation branch: unlike reachability and partition, the host
world
has no *given* coarse projection, so the privilege-class set is a design choice, and whether
re-grounding on it composes is a measurement about which worlds admit landmark planning. The module
is the host twin of [`plan`](./plan.py) / [`dist`](./dist.py): the privilege signature and a
torch-free re-grounding hop executor generic over the :class:`~verisim.hostloop.model.HostModel`
protocol, property-testable with the dependency-free ``HostNullModel`` / ``HostOracleBackedModel``
baselines. The trained flat ``M_θ`` and the goal battery live in
:mod:`verisim.experiments.lp8_host`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verisim.host.action import HostAction
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostloop.model import HostModel

# A host landmark's identity: the coarse privilege/liveness class set -- the set of (state, uid)
# classes present over live processes (the SPEC-6 §3.2 privilege axis, the dist analogue of
# ``ReachSig`` / ``ConsistencySig``). Count-free: a second non-root fork does not change it, but a
# privilege escalation (a root process appears) or the death of the last process of a class does --
# the projection the host model can be faithful on, not the exact process table.
PrivilegeSig = frozenset[tuple[str, int]]


def privilege_signature(state: HostState) -> PrivilegeSig:
    """The coarse privilege signature: the set of ``(process state, uid)`` classes present."""
    return frozenset((proc.state, proc.uid) for proc in state.procs.values())


def privilege_facts(sig: PrivilegeSig) -> int:
    """The number of facts a privilege consult verifies (the coarse-projection consult cost)."""
    return len(sig)


@dataclass(frozen=True)
class HostRolloutTrace:
    """Per-step privilege/full correctness + cost for one host plan execution (LP8-host).

    ``priv_correct[t]`` is True iff the coupled state's *privilege signature* matches the truth
    after
    step ``t`` (the planning-relevant projection); ``full_correct[t]`` the stricter bit-exact match.
    ``goal_reached`` is whether privilege is correct at the final step, which is a *model*
    prediction
    (the goal is excluded from the re-ground boundaries), so it is non-tautological.
    """

    priv_correct: tuple[bool, ...]
    full_correct: tuple[bool, ...]
    n_consults: int
    goal_reached: bool

    @property
    def n_steps(self) -> int:
        return len(self.priv_correct)

    @property
    def priv_horizon(self) -> int:
        """Longest leading run of privilege-correct steps (the privilege-altitude ``H_ε``)."""
        return _leading_run(self.priv_correct)

    @property
    def full_horizon(self) -> int:
        """Longest leading run of bit-exact-correct steps (the bit-exact ``H_ε``)."""
        return _leading_run(self.full_correct)

    @property
    def goal_reached_exact(self) -> bool:
        """Goal reached under the *bit-exact* projection (the counterpart of the privilege one)."""
        return bool(self.full_correct[-1]) if self.full_correct else False


def _leading_run(flags: tuple[bool, ...]) -> int:
    h = 0
    for ok in flags:
        if not ok:
            break
        h += 1
    return h


def execute_host_plan(
    model: HostModel,
    start: HostState,
    actions: Sequence[HostAction],
    truth_states: Sequence[HostState],
    reground_at: frozenset[int],
    *,
    reground: bool,
) -> HostRolloutTrace:
    """Free-run ``model`` over ``actions``, re-grounding to truth at privilege boundaries
    (LP8-host).

    The host twin of :func:`verisim.landmark.plan.execute_plan`. ``truth_states[t]`` is the oracle's
    true state after action ``t``. When ``reground`` is True the coupled state is reset to the truth
    at every step in ``reground_at`` (the privilege-landmark boundaries -- the
    ``imagine``/``verify``
    re-grounding); when False it is pure free-running (``ρ = 0``). The final step must be excluded
    from ``reground_at`` so goal-reach is a genuine model prediction.
    """
    state = start
    priv_correct: list[bool] = []
    full_correct: list[bool] = []
    consults = 0
    for t, action in enumerate(actions):
        truth = truth_states[t]
        predicted = apply(state, model.predict_delta(state, action))  # IMAGINE (no oracle)
        if reground and t in reground_at:
            consults += 1
            state = truth  # VERIFY + CORRECT: re-ground at the privilege-landmark boundary
        else:
            state = predicted
        priv_correct.append(privilege_signature(state) == privilege_signature(truth))
        full_correct.append(state == truth)
    return HostRolloutTrace(
        priv_correct=tuple(priv_correct),
        full_correct=tuple(full_correct),
        n_consults=consults,
        goal_reached=bool(priv_correct[-1]) if priv_correct else False,
    )


__all__ = [
    "HostRolloutTrace",
    "PrivilegeSig",
    "execute_host_plan",
    "privilege_facts",
    "privilege_signature",
]
