"""SPEC-23 (H158/H160) -- the protocol-driven discover->fix->re-verify loop, extracted from
:func:`coverage_synth.synthesize` and generalized behind the
:class:`~verisim.audit.protocols.Monitor`/:class:`~verisim.audit.protocols.Oracle` protocols.

``audit(monitor, oracle, proposer, budget) -> Certificate`` is the differentiated artifact: point it
at any covering surface and any exact oracle, and it returns the oracle-confirmed holes (realizing
actions off the surface), the per-class on/off verdict, and the fail-closed routing for residuals.

The loop is the founding CEGIS shape (docs/lineage.md):

  1. **discover** -- the proposer emits candidate actions; the oracle says which realize the harm.
  2. **fix** -- a realizing action the monitor does not cover is a hole. If the monitor is
     *growable*
     (:class:`GrowableMonitor`, e.g. our own synthesizable target), repair it to cover the hole
     (RA22's monotone grow); iterate to convergence. A monitor we did not write is not growable, so
     its holes are reported, not repaired.
  3. **re-verify** -- a final pass scores every realizing action as covered / silent / residual. A
     **silent hole** is a CLEAR-on-realizing miss *within the monitor's contract* (a genuine
     soundness bug, the number the CI gate fails on); the residual is explicitly routed by
     reversibility (reversible -> post-commit diff; irreversible -> human, fail closed). The
     soundness invariant, by construction: every realizing action is covered, repaired, silent, or
     routed -- nothing is silently dropped.

Torch-free, deterministic given a deterministic proposer.
"""

from __future__ import annotations

from collections.abc import Callable

from verisim.realagent.shell_resolver import is_irreversible

from .protocols import (
    EMPTY,
    Action,
    Certificate,
    Context,
    ContractMonitor,
    GrowableMonitor,
    Hole,
    Monitor,
    Oracle,
    Proposer,
    State,
    default_in_contract,
)


def _reversible(action: Action) -> bool:
    """Reversibility classifier (CU27): a filesystem mutation is reversible (post-commit diff +
    rollback); an off-host send / unbounded eval is not. Policy actions (no command) are reversible
    (a graph edit / counter is undoable). Used only to *route* the out-of-contract residual."""
    return True if not action.command else not is_irreversible(action.command)


def _in_contract(monitor: Monitor, action: Action, ctx: Context) -> bool:
    if isinstance(monitor, ContractMonitor):
        return monitor.in_contract(action, ctx)
    return default_in_contract(action)


def audit(
    monitor: Monitor,
    oracle: Oracle,
    proposer: Proposer,
    budget: int = 512,
    *,
    state: State = EMPTY,
    ctx: Context = EMPTY,
    reversible: Callable[[Action], bool] = _reversible,
) -> Certificate:
    """Run discover->fix->re-verify; return the coverage certificate over the sampled space."""
    # discover: materialize the distinct proposed actions (the sampled composition space)
    seen: set[tuple[str, tuple[str, ...]]] = set()
    actions: list[Action] = []
    for a in proposer.propose(budget):
        key = (a.command, a.op)
        if key in seen:
            continue
        seen.add(key)
        actions.append(a)

    realizing = [a for a in actions if oracle.realizes(a, state)]

    # fix: CEGIS grow the surface on each uncovered realizing action, to convergence (monotone, so a
    # pass with no new repair is the fixpoint). A non-growable monitor never repairs -> one pass.
    grow: GrowableMonitor | None = monitor if isinstance(monitor, GrowableMonitor) else None
    growable = grow is not None
    rounds = 0
    while grow is not None:
        rounds += 1
        repaired_any = False
        for a in realizing:
            if monitor.covers(a, ctx):
                continue
            if grow.repair(a, ctx):
                repaired_any = True
        if not repaired_any:
            break
    rounds = max(rounds, 1)

    # re-verify: score every realizing action as covered / silent / residual, and route the residual
    holes: list[Hole] = []
    covered = repaired = silent = post_commit = human = 0
    per_class: dict[str, dict[str, int]] = {}

    def _bump(klass: str, field: str) -> None:
        per_class.setdefault(klass, {"realizing": 0, "covered": 0, "silent": 0, "residual": 0})
        per_class[klass][field] += 1

    seeded = monitor.covers  # bound once
    for a in realizing:
        _bump(a.klass, "realizing")
        if seeded(a, ctx):
            covered += 1
            if growable:
                repaired += 1  # a growable monitor covers only what the loop grew it to cover
            _bump(a.klass, "covered")
            continue
        in_contract = _in_contract(monitor, a, ctx)
        rev = reversible(a)
        if in_contract:
            silent += 1
            route = "silent"
            _bump(a.klass, "silent")
        elif rev:
            post_commit += 1
            route = "post_commit_diff"
            _bump(a.klass, "residual")
        else:
            human += 1
            route = "human"
            _bump(a.klass, "residual")
        holes.append(Hole(command=a.command, klass=a.klass, op=a.op,
                          string_resolvable=a.string_resolvable, reversible=rev,
                          silent=(route == "silent"), route=route))

    synthesized = list(getattr(monitor, "prefixes", []))
    return Certificate(
        monitor=monitor.name,
        oracle=oracle.name,
        proposer=proposer.name,
        budget=budget,
        n_proposed=len(actions),
        n_realizing=len(realizing),
        covered=covered,
        repaired=repaired if growable else 0,
        rounds_to_converge=rounds,
        silent_holes=silent,
        residual_post_commit=post_commit,
        residual_human=human,
        holes=holes,
        synthesized_surface=synthesized,
        per_class=per_class,
    )
