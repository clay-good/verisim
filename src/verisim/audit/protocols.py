"""SPEC-23 (H158) -- the monitor-auditor protocols: ``Monitor``, ``Oracle``, ``Proposer``, and the
coverage ``Certificate`` they produce.

The whole RA arc (RA22-RA25) certified *our* monitor over *one* harm family with the protected path
(`/etc/shadow`) baked into the proposer, the resolver, and the loop. SPEC-23's pivot (docs/specs/
SPEC-23.md) is that the certification loop -- the program's only un-contested artifact -- is
monitor-agnostic the moment the monitor and the oracle are **protocols** rather than hardcoded
predicates. This module is those protocols, lifted out of the shipped modules with no literal path:

  - **Monitor** -- the covering surface under audit. ``covers(action, ctx)`` is True when the
    monitor *would check* the action (escalate / block / route it). This is anything that classifies
    an action as on/off its checked surface: a syntactic path scan, a denylist, the abstract shell
    resolver, a policy engine.
  - **Oracle** -- the free verifier: the exact, resettable ground truth of harm for this monitor's
    harm family. ``realizes(action, state)`` is True when executing the action *truly causes* the
    harm. It is read from execution/semantics, never from the monitor.

A **soundness hole** is exactly ``oracle.realizes(a, s) and not monitor.covers(a, ctx)`` -- a
realizing action off the checked surface (a silent miss). The certificate is the audit artifact: the
holes (each confirmed by the oracle), the sampled composition space, the per-class on/off-surface
verdict, and the fail-closed routing for the irreducible residual.

Two optional refinements a monitor may declare (checked structurally by the auditor, so the base
protocol stays the two-method shape the spec names):

  - :class:`ContractMonitor` -- ``in_contract(action, ctx)`` says whether an uncovered realizing
    action is a *soundness bug* (the monitor claims to cover this class, so a miss is a silent hole)
    or genuinely *out of contract* (no string analysis can decide it -- the symlink residual --
    so it is routed to the post-commit diff, not counted as a silent hole). The default, for a
    monitor that declares nothing, is :func:`default_in_contract`: an action is in-contract iff its
    harm is determinable from the action itself (``string_resolvable``).
  - :class:`GrowableMonitor` -- ``repair(action, ctx)`` grows the covering surface to include a
    fresh hole (the CEGIS move RA22 used to synthesize a target from empty). A monitor we did not
    write is not growable; ours is, so the loop reproduces RA22's synthesis through the interface.

Torch-free, standard library only.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

from .guarantee import Guarantee

#: The world the monitor classifies against (a cwd, a freeze flag, ...) and the world the oracle
#: evaluates harm against (a base reachability graph, an access count, ...). Open mappings, so an
#: instance reads only the keys it understands; the file-harm family uses neither.
Context = Mapping[str, object]
State = Mapping[str, object]

EMPTY: Mapping[str, object] = {}


@dataclass(frozen=True)
class Action:
    """A proposed action under audit.

    ``command`` is the shell string for the file-corruption family; ``op`` is the structured tuple
    for a policy family (e.g. ``("add_edge", "analytics", "reporting")``). Both carry the
    by-construction ground truth an exact oracle would return (``realizes``) and whether the harm is
    determinable from the action alone (``string_resolvable`` -- False only for the symlink case,
    whose realized target depends on on-disk link state the action never names). ``klass`` and
    ``depth`` are provenance for the per-class certificate, never read by the loop's hole predicate.
    """

    command: str = ""
    op: tuple[str, ...] = ()
    realizes: bool = True
    string_resolvable: bool = True
    klass: str = ""
    depth: int = 0


def default_in_contract(action: Action) -> bool:
    """The residual rule for a monitor that declares no contract: a harm determinable from the
    action itself is in any monitor's contract (a miss is a soundness bug); a harm that depends on
    hidden on-disk/runtime state (``string_resolvable=False``) is the honest irreducible residual
    routed to the post-commit diff."""
    return action.string_resolvable


@runtime_checkable
class Monitor(Protocol):
    """The covering surface under audit: ``covers`` is True == 'I would check this'."""

    name: str

    def covers(self, action: Action, ctx: Context = EMPTY) -> bool: ...


@runtime_checkable
class Oracle(Protocol):
    """The free verifier: ``realizes`` is True == 'this action truly causes the harm'."""

    name: str

    def realizes(self, action: Action, state: State = EMPTY) -> bool: ...


@runtime_checkable
class Proposer(Protocol):
    """The structured/stochastic adversary: emit candidate actions to probe the monitor with."""

    name: str

    def propose(self, budget: int) -> Iterable[Action]: ...


@runtime_checkable
class ContractMonitor(Protocol):
    """A monitor that declares which uncovered realizing actions are *in its contract* (a miss is a
    silent hole) versus out of contract (the routed residual)."""

    def in_contract(self, action: Action, ctx: Context = EMPTY) -> bool: ...


@runtime_checkable
class GrowableMonitor(Protocol):
    """A monitor whose covering surface the auditor may grow on a fresh hole (CEGIS, RA22)."""

    def repair(self, action: Action, ctx: Context = EMPTY) -> bool: ...


@dataclass(frozen=True)
class Hole:
    """One oracle-confirmed soundness hole: a realizing action the monitor does not cover.

    ``silent`` is the load-bearing bit: a silent hole is in the monitor's contract (it claims to
    cover this class) and so is a genuine soundness bug -- a CLEAR-on-realizing miss the CI
    gate must
    fail on. A non-silent hole is the routed residual (out of contract, caught post-commit)."""

    command: str
    klass: str
    op: tuple[str, ...]
    string_resolvable: bool
    reversible: bool
    silent: bool
    route: str  # "covered_after_repair" is impossible here; one of: "silent", "post_commit_diff",
    #             "human" (fail-closed irreversible)
    depth: int = 0  # SPEC-24: composition depth (the minimal-witness-depth the certificate reports)


@dataclass
class Certificate:
    """The audit artifact: a CI-consumable coverage certificate over the sampled space.

    ``silent_holes`` (CLEAR-on-realizing, in-contract) is the number the CI gate fails on; it is 0
    iff the monitor is sound over its declared contract on the sampled space. The residual holes are
    explicitly routed (reversible -> post-commit diff; irreversible -> human), never silent."""

    monitor: str
    oracle: str
    proposer: str
    budget: int
    n_proposed: int
    n_realizing: int
    covered: int
    repaired: int
    rounds_to_converge: int
    silent_holes: int
    residual_post_commit: int
    residual_human: int
    holes: list[Hole] = field(default_factory=list)
    synthesized_surface: list[str] = field(default_factory=list)
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    #: SPEC-24: the coverage class this certificate carries (exhaustive depth + residual bound +
    #: modeled algebra). None for a bare SPEC-23 sampled audit; filled by :func:`certify`.
    guarantee: Guarantee | None = None
    spec: str = "SPEC-23"
    version: int = 2

    @property
    def sound(self) -> bool:
        """The certificate's verdict: no silent (CLEAR-on-realizing, in-contract) hole was found."""
        return self.silent_holes == 0

    @property
    def min_hole_depth(self) -> int | None:
        """The minimal composition depth at which an in-contract (silent) hole was found, or
        None."""
        depths = [h.depth for h in self.holes if h.silent]
        return min(depths) if depths else None

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(asdict(self), indent=indent, sort_keys=True)

    def write(self, path: str) -> str:
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json() + "\n")
        return str(out)
