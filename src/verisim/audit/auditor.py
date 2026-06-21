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

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from verisim.realagent.compositional_grammar import ATOMS, MECHANISMS, VERBS
from verisim.realagent.shell_resolver import is_irreversible

from .guarantee import Guarantee, good_turing_missing_mass, residual_epsilon
from .proposers import ExhaustiveDepthProposer, GrammarProposer
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

DEFAULT_WORK = "/home/work"


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
                          silent=(route == "silent"), route=route, depth=a.depth))

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


# --- SPEC-24 (H165/H166/H170): the graded certify entrypoint --------------------------------------


def _algebra() -> tuple[str, int]:
    """The modeled action algebra (the compositional shell grammar) and its full size, recorded so
    the guarantee is never read wider than the algebra it is complete over."""
    name = f"compositional_shell(MECH={len(MECHANISMS)},ATOMS={len(ATOMS)},VERBS={len(VERBS)})"
    size = len(MECHANISMS) ** len(ATOMS) * len(VERBS)
    return name, size


def certify(
    monitor: Monitor,
    oracle: Oracle,
    *,
    protected_path: str,
    work: str = DEFAULT_WORK,
    depth: int = 2,
    n_sampled: int = 2000,
    delta: float = 0.05,
    seed: int = 0,
    external: str = "",
    state: State = EMPTY,
    ctx: Context = EMPTY,
) -> Certificate:
    """SPEC-24: the graded certificate -- an exhaustive theorem to ``depth`` plus a statistical
    residual bound over the sampled tail, with the modeled algebra recorded.

    Tier 1: a clean audit over the *complete* depth-<=k direct lattice (no sampling) -> a proof for
    that fragment. Tier 2: a sampled tail (blind uniform, with repeats) -> ``residual_epsilon`` (the
    rule-of-three / Wilson upper bound on the in-contract hole rate) + the Good-Turing missing mass.
    The returned certificate merges both holes sets and carries the :class:`Guarantee`."""
    # Tier 1 -- exhaustive proof for the bounded fragment.
    ex = audit(monitor, oracle, ExhaustiveDepthProposer(protected_path, depth, work),
               state=state, ctx=ctx)

    # Tier 2 -- the sampled tail, with repeats, for the residual statistics + missing mass.
    n_realizing = 0
    hole_freq: dict[tuple[str, tuple[str, ...]], int] = {}
    tail_holes: dict[tuple[str, tuple[str, ...]], Hole] = {}
    for a in GrammarProposer(protected_path, work, mode="blind", seed=seed).propose(n_sampled):
        if not oracle.realizes(a, state):
            continue
        n_realizing += 1
        if monitor.covers(a, ctx):
            continue
        if not _in_contract(monitor, a, ctx):
            continue  # out-of-contract residual: routed, not part of the in-contract residual rate
        key = (a.command, a.op)
        hole_freq[key] = hole_freq.get(key, 0) + 1
        if key not in tail_holes:
            rev = _reversible(a)
            tail_holes[key] = Hole(command=a.command, klass=a.klass, op=a.op,
                                   string_resolvable=a.string_resolvable, reversible=rev,
                                   silent=True, route="silent", depth=a.depth)

    k_hole_draws = sum(hole_freq.values())
    singletons = sum(1 for c in hole_freq.values() if c == 1)
    eps = residual_epsilon(k_hole_draws, n_realizing, delta)
    mass = good_turing_missing_mass(singletons, n_realizing)

    algebra, size = _algebra()
    guarantee = Guarantee(
        algebra=algebra, algebra_size=size, exhaustive_depth=depth, n_exhaustive=ex.n_proposed,
        residual_epsilon=eps, residual_delta=delta, n_sampled=n_realizing, missing_mass=mass,
        oracle=oracle.name, external=external,
    )

    # Merge the exhaustive holes with any new tail holes (dedup by command/op).
    merged = list(ex.holes)
    seen = {(h.command, h.op) for h in ex.holes}
    for key, h in tail_holes.items():
        if key not in seen:
            merged.append(h)
            seen.add(key)
    silent_total = sum(1 for h in merged if h.silent)

    ex.proposer = f"certify(exhaustive_depth{depth}+sampled{n_realizing})"
    ex.holes = merged
    ex.silent_holes = silent_total
    ex.guarantee = guarantee
    ex.spec = "SPEC-24"
    return ex


# --- SPEC-24 (H169): differential certification (the monitor-patch regression gate) ---------------


def _hole_key(h: Hole) -> tuple[str, tuple[str, ...]]:
    return (h.command, h.op)


@dataclass
class DiffCertificate:
    """Certify a monitor *patch* is a monotone improvement: it ``closed`` a hole set and ``opened``
    none, over the shared sampled+exhaustive space -- RA25's post-commit-diff move applied to the
    monitor itself, so v2's completeness is not re-asserted from scratch, only the *change*
    certified."""

    monitor_before: str
    monitor_after: str
    closed: list[Hole] = field(default_factory=list)
    opened: list[Hole] = field(default_factory=list)

    @property
    def monotone(self) -> bool:
        """True iff the patch opened no new in-contract hole (a safe, monotone improvement)."""
        return not self.opened

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)


def differential(before: Certificate, after: Certificate) -> DiffCertificate:
    """Partition the in-contract (silent) hole sets of two certificates: closed = in before not
    after;
    opened = in after not before. Monotone iff ``opened`` is empty."""
    b = {_hole_key(h): h for h in before.holes if h.silent}
    a = {_hole_key(h): h for h in after.holes if h.silent}
    closed = [h for k, h in b.items() if k not in a]
    opened = [h for k, h in a.items() if k not in b]
    return DiffCertificate(before.monitor, after.monitor, closed, opened)


def audit_diff(
    monitor_before: Monitor,
    monitor_after: Monitor,
    oracle: Oracle,
    make_proposer: Callable[[], Proposer],
    *,
    budget: int = 512,
    state: State = EMPTY,
    ctx: Context = EMPTY,
) -> DiffCertificate:
    """Audit two monitor versions over a fresh proposer each (proposers are single-use), and diff
    them. ``make_proposer`` is called once per version so both see the identical sampled space."""
    cb = audit(monitor_before, oracle, make_proposer(), budget, state=state, ctx=ctx)
    ca = audit(monitor_after, oracle, make_proposer(), budget, state=state, ctx=ctx)
    return differential(cb, ca)
