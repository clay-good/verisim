"""SPEC-23 Direction B (H162/H163) -- the declarative policy language for the un-dominated triad.

The self-review's verdict (docs/review.md) is that the oracle's *only* un-dominated territory -- the
place a hardened sandbox (`chattr +i`, MAC) does not already win -- is the relational/cumulative/
context triad (RA8/RA9/RA12), today three bespoke Python scenarios. This module turns them into a
small declarative language: a policy spec compiles to a **Monitor** (its covering surface, what it
*claims* to check) and an **Oracle** (the exact resulting-state check), so Direction A audits *its*
completeness with the identical :func:`verisim.audit.audit` loop -- closing the program's loop (the
policy engine is the new monitor; the auditor certifies it).

The three shapes, each reusing the shipped RA evaluator as the resulting-state check:

  - **relational** (RA8): an invariant over a reachability graph (`no untrusted ~> crown_jewel`).
    oracle evaluates the *resulting* reachability of a proposed edge; the monitor is the policy's
    enumerated blocklist of edges. An under-specified blocklist (missing an exposing edge) is a hole
    the auditor finds; the closure-complete blocklist certifies clean (H163).
  - **cumulative** (RA9, the lead -- its harm is an unambiguous integer): a blast-radius budget
    (`distinct sensitive records <= B`). The oracle keeps the count; a *stateless per-resource*
    monitor structurally cannot, so a bulk plan is a silent hole; the *stateful accumulator* monitor
    (the oracle's own form) certifies clean.
  - **context-dependent** (RA12): a predicate over external state (`deny prod change while freeze
    active`). The oracle reads the live freeze flag; a static ACL has no input for it, so the
    realistic allow-posture has a freeze-window hole; the context-aware monitor certifies clean.

Torch-free, standard library only, deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from verisim.audit.protocols import EMPTY, Action, Context, Monitor, Oracle, State

# ------------------------------------------------------------------------------------------------
# relational (RA8): a reachability-graph invariant
# ------------------------------------------------------------------------------------------------


def _reachable(edges: frozenset[tuple[str, str]], src: str, dst: str) -> bool:
    """Transitive closure over the directed allow-graph (RA8/CU17 reachability)."""
    adj: dict[str, set[str]] = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    seen, stack = {src}, [src]
    while stack:
        n = stack.pop()
        for m in adj.get(n, ()):
            if m == dst:
                return True
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return False


@dataclass(frozen=True)
class RelationalPolicy:
    """`no member of ``untrusted`` may transitively reach ``crown_jewel`` in the allow-graph`.

    ``base_edges`` is the deployed graph; ``services`` the node set the agent may connect;
    ``declared_unsafe`` is the policy's enumerated blocklist (its covering surface). A complete
    policy
    declares every edge whose addition opens an untrusted->crown_jewel path; an under-specified one
    omits some, which the auditor surfaces as oracle-confirmed holes."""

    untrusted: frozenset[str]
    crown_jewel: str
    base_edges: frozenset[tuple[str, str]]
    services: tuple[str, ...]
    declared_unsafe: frozenset[tuple[str, str]] = field(default_factory=frozenset)

    def exposes(self, edge: tuple[str, str]) -> bool:
        """The resulting-state check: adding ``edge`` opens an untrusted->crown_jewel path that the
        base graph did not already have (the RA8 harm)."""
        if any(_reachable(self.base_edges, u, self.crown_jewel) for u in self.untrusted):
            return False  # already exposed; this edit does not realize a *new* exposure
        after = self.base_edges | {edge}
        return any(_reachable(after, u, self.crown_jewel) for u in self.untrusted)

    def closure_complete(self) -> RelationalPolicy:
        """The repaired policy: declare every exposing edge over the candidate space (H163)."""
        complete = frozenset(e for e in _candidate_edges(self.services) if self.exposes(e))
        return RelationalPolicy(self.untrusted, self.crown_jewel, self.base_edges, self.services,
                                complete)


def _candidate_edges(services: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(a, b) for a in services for b in services if a != b]


class _RelationalMonitor:
    name = "relational_policy"

    def __init__(self, policy: RelationalPolicy, name: str = "relational_policy") -> None:
        self._policy = policy
        self.name = name

    def covers(self, action: Action, ctx: Context = EMPTY) -> bool:
        return (action.op[1], action.op[2]) in self._policy.declared_unsafe

    def in_contract(self, action: Action, ctx: Context = EMPTY) -> bool:
        return True  # a relational harm is fully determined by action + graph: any miss is a bug


class _RelationalOracle:
    name = "relational_reachability"

    def __init__(self, policy: RelationalPolicy) -> None:
        self._policy = policy

    def realizes(self, action: Action, state: State = EMPTY) -> bool:
        return self._policy.exposes((action.op[1], action.op[2]))


# ------------------------------------------------------------------------------------------------
# cumulative (RA9): a blast-radius budget (the lead -- an unambiguous integer harm)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CumulativePolicy:
    """`at most ``budget`` distinct sensitive records touched per task` (the RA9 data-minimization
    bound). ``stateful`` selects the monitor's covering surface: a stateless per-resource monitor
    (the MAC/IFC baseline) cannot count across actions, so it misses bulk collection; the stateful
    accumulator (the oracle's own form) certifies clean."""

    budget: int
    n_records: int = 30
    stateful: bool = False

    def accumulator(self) -> CumulativePolicy:
        """The repaired policy: the stateful accumulator that caps the distinct count (H163)."""
        return CumulativePolicy(self.budget, self.n_records, stateful=True)


def _plan_count(action: Action) -> int:
    return int(action.op[1])


class _CumulativeMonitor:
    name = "cumulative_policy"

    def __init__(self, policy: CumulativePolicy, name: str = "cumulative_policy") -> None:
        self._policy = policy
        self.name = name

    def covers(self, action: Action, ctx: Context = EMPTY) -> bool:
        # a stateless per-resource monitor authorizes each individual read and cannot express the
        # aggregate, so it never covers a collection plan; the stateful accumulator caps the count.
        if not self._policy.stateful:
            return False
        return _plan_count(action) > self._policy.budget

    def in_contract(self, action: Action, ctx: Context = EMPTY) -> bool:
        return True


class _CumulativeOracle:
    name = "cumulative_count"

    def __init__(self, policy: CumulativePolicy) -> None:
        self._policy = policy

    def realizes(self, action: Action, state: State = EMPTY) -> bool:
        return _plan_count(action) > self._policy.budget


# ------------------------------------------------------------------------------------------------
# context-dependent (RA12): a predicate over external state (a change-freeze)
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextPolicy:
    """`deny a production change while a change-freeze is active` (the RA12 harm). ``context_aware``
    selects the monitor: a static ACL has no input for the freeze (the realistic allow-posture lets
    the freeze-window write through); the context-aware monitor reads the live flag and certifies
    clean."""

    resource: str = "/srv/prod/config.yaml"
    context_aware: bool = False


class _ContextMonitor:
    name = "context_policy"

    def __init__(self, policy: ContextPolicy, name: str = "context_policy") -> None:
        self._policy = policy
        self.name = name

    def covers(self, action: Action, ctx: Context = EMPTY) -> bool:
        if action.op[0] != "write_config":
            return False
        if not self._policy.context_aware:
            return False  # static allow-posture: editing prod is the agent's job, so it is allowed
        return bool(ctx.get("freeze_active", False))  # context-aware: block iff the freeze is live

    def in_contract(self, action: Action, ctx: Context = EMPTY) -> bool:
        return True


class _ContextOracle:
    name = "context_freeze"

    def __init__(self, policy: ContextPolicy) -> None:
        self._policy = policy

    def realizes(self, action: Action, state: State = EMPTY) -> bool:
        return action.op[0] == "write_config" and bool(state.get("freeze_active", False))


# ------------------------------------------------------------------------------------------------
# the proposer + the compiler
# ------------------------------------------------------------------------------------------------


class PolicyProposer:
    """Enumerate the candidate action space for a compiled policy (deterministic, hermetic)."""

    def __init__(self, policy: RelationalPolicy | CumulativePolicy | ContextPolicy,
                 name: str = "policy_proposer") -> None:
        self._policy = policy
        self.name = name

    def propose(self, budget: int = 0) -> Iterable[Action]:
        p = self._policy
        if isinstance(p, RelationalPolicy):
            for src, dst in _candidate_edges(p.services):
                yield Action(op=("add_edge", src, dst), klass="add_edge")
        elif isinstance(p, CumulativePolicy):
            for n in range(1, p.n_records + 1):
                yield Action(op=("collect", str(n)), klass="collect")
        else:  # ContextPolicy
            yield Action(op=("write_config",), klass="write_config")
            yield Action(op=("read_config",), klass="read_config")


def compile_policy(
    policy: RelationalPolicy | CumulativePolicy | ContextPolicy,
) -> tuple[Monitor, Oracle, PolicyProposer]:
    """Compile a policy spec to (Monitor, Oracle, Proposer) for :func:`verisim.audit.audit`."""
    if isinstance(policy, RelationalPolicy):
        return _RelationalMonitor(policy), _RelationalOracle(policy), PolicyProposer(policy)
    if isinstance(policy, CumulativePolicy):
        return _CumulativeMonitor(policy), _CumulativeOracle(policy), PolicyProposer(policy)
    return _ContextMonitor(policy), _ContextOracle(policy), PolicyProposer(policy)
