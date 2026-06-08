"""The distributed-world landmark layer: consistency-graph planning (SPEC-12 §6 LP8, H38).
Torch-free.

The network landmark layer ([`graph`](./graph.py), [`plan`](./plan.py)) wires its graph on
*reachability* -- the control-plane projection the model is faithful on (EN10). LP8 asks whether the
*method* transfers (H38): re-run the LP3 planner on the distributed world, where the
planning-relevant
hidden state is not reachability but **consistency/partition structure** -- the state ED12 measured
and ED5/H19 found the model's *consistency* prediction outlasts its *bit-exact* one on. So the
distributed landmark signature is the coarse consistency structure (which objects are converged vs
split, the partition topology, the down nodes), and a hop is a consistency-changing op.

This module is the distributed twin of [`plan`](./plan.py): the consistency signature and a
torch-free
re-grounding hop executor generic over the :class:`~verisim.distloop.model.DistModel` protocol, so
the
deterministic half is property-testable with the dependency-free ``DistNullModel`` /
``DistOracleBackedModel`` baselines -- no GPU, no training (the NW0-NW3 discipline, SPEC-5 §13). The
trained flat ``M_θ`` and the goal battery live in the experiment
(:mod:`verisim.experiments.lp8_dist`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verisim.dist.action import DistAction
from verisim.dist.config import DistConfig
from verisim.dist.delta import apply
from verisim.dist.state import DistributedState
from verisim.distloop.model import DistModel
from verisim.distmetrics.divergence import object_consistency_view

# A distributed landmark's identity: the *coarse* partition/convergence structure -- per-object
# converged-vs-split, the partition topology, and the down-node set. This is the planning-relevant
# hidden state ED12 measured (partition/crash indistinguishability), the dist analogue of the
# network ``ReachSig`` and deliberately *coarse*: it drops each replica's exact (version, value) --
# which increments on every write and would make the signature near-bit-exact -- and keeps only the
# convergence/partition structure the model can be faithful on (ED5/H19: the consistency prediction
# outlasts the bit-exact one). The reachability->consistency projection swap is the whole point of
# the cross-world fork (SPEC-12 §6 LP8, H38).
ConsistencySig = tuple[
    frozenset[str],  # the converged (non-split) objects
    tuple[tuple[str, ...], ...],  # the canonical partition groups
    frozenset[str],  # the down (crashed) nodes
]


def consistency_signature(state: DistributedState, config: DistConfig) -> ConsistencySig:
    """The coarse consistency signature: converged objects + partition topology + down nodes."""
    converged = frozenset(
        obj for obj in config.objects if len(object_consistency_view(state, obj)) == 1
    )
    partitions = tuple(sorted(tuple(sorted(group)) for group in state.partitions))
    return (converged, partitions, frozenset(state.down))


def consistency_facts(sig: ConsistencySig) -> int:
    """The number of facts a consistency consult verifies (the coarse-projection consult cost)."""
    converged, partitions, down = sig
    return len(converged) + sum(len(group) for group in partitions) + len(down)


@dataclass(frozen=True)
class DistRolloutTrace:
    """Per-step consistency/full correctness + cost for one distributed plan execution (LP8).

    ``consistency_correct[t]`` is True iff the coupled state's *consistency signature* matches the
    truth after step ``t`` (the planning-relevant projection, the dist analogue of EN10's
    reachability); ``full_correct[t]`` the stricter bit-exact match (ED5/H19: consistency is
    expected
    to outlast it). ``goal_reached`` is whether consistency is correct at the final step, which is a
    *model* prediction (the goal is excluded from the re-ground boundaries), so it is
    non-tautological.
    """

    consistency_correct: tuple[bool, ...]
    full_correct: tuple[bool, ...]
    n_consults: int
    goal_reached: bool

    @property
    def n_steps(self) -> int:
        return len(self.consistency_correct)

    @property
    def consistency_horizon(self) -> int:
        """Longest leading run of consistency-correct steps (the consistency-altitude ``H_ε``)."""
        return _leading_run(self.consistency_correct)

    @property
    def full_horizon(self) -> int:
        """Longest leading run of bit-exact-correct steps (the ED5/H19 bit-exact ``H_ε``)."""
        return _leading_run(self.full_correct)

    @property
    def goal_reached_exact(self) -> bool:
        """Goal reached under the *bit-exact* projection (the ED5/H19 counterpart of
        consistency)."""
        return bool(self.full_correct[-1]) if self.full_correct else False


def _leading_run(flags: tuple[bool, ...]) -> int:
    h = 0
    for ok in flags:
        if not ok:
            break
        h += 1
    return h


def execute_dist_plan(
    model: DistModel,
    config: DistConfig,
    start: DistributedState,
    actions: Sequence[DistAction],
    truth_states: Sequence[DistributedState],
    reground_at: frozenset[int],
    *,
    reground: bool,
) -> DistRolloutTrace:
    """Free-run ``model`` over ``actions``, re-grounding to truth at consistency boundaries (LP8).

    The distributed twin of :func:`verisim.landmark.plan.execute_plan`. ``truth_states[t]`` is the
    oracle's true state after action ``t``. When ``reground`` is True the coupled state is reset to
    the truth at every step in ``reground_at`` (the consistency-landmark boundaries -- the
    ``imagine``/``verify`` re-grounding); when False it is pure free-running (``ρ = 0``). The final
    step must be excluded from ``reground_at`` so goal-reach is a genuine model prediction.
    """
    state = start
    consistency_correct: list[bool] = []
    full_correct: list[bool] = []
    consults = 0
    for t, action in enumerate(actions):
        truth = truth_states[t]
        predicted = apply(state, model.predict_delta(state, action))  # IMAGINE (no oracle)
        if reground and t in reground_at:
            consults += 1
            state = truth  # VERIFY + CORRECT: re-ground at the consistency-landmark boundary
        else:
            state = predicted
        consistency_correct.append(
            consistency_signature(state, config) == consistency_signature(truth, config)
        )
        full_correct.append(state == truth)
    return DistRolloutTrace(
        consistency_correct=tuple(consistency_correct),
        full_correct=tuple(full_correct),
        n_consults=consults,
        goal_reached=bool(consistency_correct[-1]) if consistency_correct else False,
    )


__all__ = [
    "ConsistencySig",
    "DistRolloutTrace",
    "consistency_facts",
    "consistency_signature",
    "execute_dist_plan",
]
