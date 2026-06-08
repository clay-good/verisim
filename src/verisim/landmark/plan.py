"""The high-level planner: graph search over the verified landmark graph + the hop executor (LP3).

SPEC-12 §3 splits planning into two altitudes. The **high-level planner** does the multi-hop
reasoning by *graph search* over the oracle-verified landmark graph (never the per-step model, never
an LLM walking the graph -- §2.2): :func:`shortest_landmark_path` returns the fewest-hop subgoal
sequence between two landmarks. The **low-level controller** executes each hop with the shipped
``imagine``/``verify`` loop -- here :func:`execute_plan`, the reachability-altitude reading of that
loop: the model free-runs the actions of a hop (``imagine``, no oracle) and the oracle re-grounds
the coupled state at the *landmark boundary* (``verify``+correct), so within a hop the model runs on
its own and only at the boundary does it pay one consult.

This is the move SPEC-10's HS3 wall forces (SPEC-12 §1): a model whose free-running horizon is
pinned near zero cannot be rolled forward to a distant goal, but it *can* be trusted across a short
hop and re-grounded at the next landmark. The headline (H33, LP3) is that re-grounding at landmark
boundaries composes the model's short competence radius into long-range goal reach that flat
free-running (``reground=False``, ``ρ = 0``) cannot reach -- free-running compounds (HS3) while hops
re-ground.

Everything here is torch-free (the NW0-NW3 discipline, SPEC-5 §13): :func:`execute_plan` is generic
over the :class:`~verisim.netloop.model.NetModel` protocol and consumes a *precomputed* ground-truth
rollout, so it is fully deterministic and property-testable with the dependency-free
``NetNullModel``/``NetOracleBackedModel`` baselines -- no GPU, no training. The torch graph arm and
the goal battery live in the experiment (:mod:`verisim.experiments.lp3`).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from verisim.landmark.graph import LandmarkGraph, ReachSig, reach_signature
from verisim.net.action import NetAction
from verisim.net.state import NetworkState
from verisim.netdelta.apply import apply
from verisim.netloop.model import NetModel


def shortest_landmark_path(graph: LandmarkGraph, src: int, dst: int) -> list[int] | None:
    """Fewest-hop landmark id sequence ``src -> ... -> dst`` over verified edges (``None`` if none).

    Breadth-first search over the verified edge set -- the high-level planner of SPEC-12 §3: the
    multi-hop reasoning is done by graph search, server-side, *not* by the per-step model and *not*
    by an LLM walking the graph (§2.2, the NLGraph result). Because every edge is oracle-confirmed
    (LP2's zero-false-paths guarantee), any path this returns is a sequence of real reachability
    transitions -- the soundness no static attack-graph tool has (§8).
    """
    if src == dst:
        return [src]
    prev: dict[int, int] = {src: src}
    queue: deque[int] = deque([src])
    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if v not in prev:
                prev[v] = u
                if v == dst:
                    path = [dst]
                    while path[-1] != src:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(v)
    return None


@dataclass(frozen=True)
class Hop:
    """One leg of a landmark plan: a multi-step free-run to the next subgoal landmark.

    ``actions`` is the action subsequence the controller free-runs within the hop; ``dst_sig`` the
    reachability signature of the landmark the hop re-grounds to at its boundary. A plan with hops
    of length ``L > 1`` is what keeps re-grounding non-trivial (``ρ = 1/L < 1``): the model runs on
    its own for ``L`` steps before the oracle corrects it (the SPEC-12 §10 honesty requirement).
    """

    actions: tuple[NetAction, ...]
    dst_sig: ReachSig


@dataclass(frozen=True)
class RolloutTrace:
    """What :func:`execute_plan` records: per-step reachability/full-state correctness + cost.

    ``reach_correct[t]`` is True iff the coupled state's *reachability signature* matches the truth
    after step ``t`` (the planning-relevant projection, EN10); ``full_correct[t]`` the stricter
    exact-state match (the HS3 reading). ``n_consults`` is the oracle budget spent (re-grounds);
    ``goal_reached`` is whether the goal's reachability is correct at the final step -- and that
    final step is a *model* prediction, never a re-ground (the goal is excluded from the
    boundaries), so goal-reach is non-tautological.
    """

    reach_correct: tuple[bool, ...]
    full_correct: tuple[bool, ...]
    n_consults: int
    goal_reached: bool

    @property
    def n_steps(self) -> int:
        return len(self.reach_correct)

    @property
    def reach_horizon(self) -> int:
        """Longest leading run of reachability-correct steps (the reachability-altitude ``H_ε``)."""
        h = 0
        for ok in self.reach_correct:
            if not ok:
                break
            h += 1
        return h

    @property
    def rho(self) -> int:
        """Consults spent; divide by :attr:`n_steps` for the oracle budget ``ρ``."""
        return self.n_consults


def execute_plan(
    model: NetModel,
    start: NetworkState,
    actions: Sequence[NetAction],
    truth_states: Sequence[NetworkState],
    reground_at: frozenset[int],
    *,
    reground: bool,
) -> RolloutTrace:
    """Free-run ``model`` over ``actions``, re-grounding to truth at the landmark boundaries (LP3).

    ``truth_states[t]`` is the oracle's true state *after* action ``t`` (the precomputed
    ground-truth rollout, length ``len(actions)``). When ``reground`` is True the coupled state is
    reset to the truth at every step in ``reground_at`` -- the landmark boundaries, the
    ``verify``+correct of the imagine/verify loop -- so the model only free-runs *within* a hop;
    when False it is pure free-running (``ρ = 0``, the HS3 baseline). ``reground_at`` must exclude
    the final step so the goal is reached by a model prediction, not a re-ground (the §10 honesty).
    """
    state = start
    reach_correct: list[bool] = []
    full_correct: list[bool] = []
    consults = 0
    for t, action in enumerate(actions):
        truth = truth_states[t]
        predicted = apply(state, model.predict_delta(state, action))  # IMAGINE (no oracle)
        if reground and t in reground_at:
            consults += 1
            state = truth  # VERIFY + CORRECT: re-ground to truth at the landmark boundary
        else:
            state = predicted
        reach_correct.append(reach_signature(state) == reach_signature(truth))
        full_correct.append(state == truth)
    return RolloutTrace(
        reach_correct=tuple(reach_correct),
        full_correct=tuple(full_correct),
        n_consults=consults,
        goal_reached=bool(reach_correct[-1]) if reach_correct else False,
    )


__all__ = ["Hop", "RolloutTrace", "execute_plan", "shortest_landmark_path"]
