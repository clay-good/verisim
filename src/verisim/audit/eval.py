"""SPEC-27 -- the logged audit harness: one proposer, one (monitor, oracle), an ordered discovery
curve.

The shipped :func:`verisim.audit.auditor.audit` materializes every proposed action, then scores
the set -- it answers "is the monitor sound over this sample?" but throws away *when* each hole was
found. SPEC-27 asks a different, comparative question -- *which proposer finds the soundness hole
first, on equal oracle calls?* -- so it must process the proposer's stream **in emission order** and
record the discovery curve. That is the only behavioral difference; the hole predicate is identical
to ``audit``'s (same :func:`_in_contract`, ``realizes ∧ ¬covers`` silent-miss test), so a run over
a deterministic proposer agrees with the certificate it would have produced (asserted in
``tests/test_spec27_eval.py``).

No new mechanism, no CEGIS repair: SPEC-27 measures hole *discovery*, not repair, so a growable
monitor would confound the race (a repair closes a hole mid-run). The harness scores against the
monitor as given and never calls ``repair``. Torch-free; the neural arm imports torch lazily inside
its own proposer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .auditor import _in_contract
from .protocols import EMPTY, Context, Monitor, Oracle, Proposer, State


@dataclass
class RunLog:
    """One proposer's audit run, with the discovery curve the comparative metrics are read off.

    The primary SPEC-27 metrics live here: ``first_silent_call`` (time-to-first soundness bug, in
    oracle calls) and ``silent_classes`` (distinct in-contract hole *classes* found at budget). The
    raw ``holes_per_call`` is the RA23 metric, kept but secondary -- it rewards camping a hole-rich
    region, which the class/first-bug metrics do not.
    """

    arm: str
    monitor: str
    oracle: str
    target: str
    seed: int
    budget: int
    n_probed: int = 0  # distinct actions actually probed (one oracle call each)
    oracle_calls: int = 0  # == n_probed; named for the compute-parity axis
    n_realizing: int = 0
    silent_holes: int = 0  # distinct in-contract holes (by command/op) -- the soundness bugs
    silent_classes: list[str] = field(default_factory=list)  # distinct klass among silent holes
    first_silent_call: int | None = None  # oracle-call index (1-based) of the first silent miss
    first_class_call: dict[str, int] = field(default_factory=dict)  # klass -> first-found call idx
    residual_routed: int = 0  # out-of-contract holes (routed, not soundness bugs)
    wall_clock_s: float = 0.0
    #: (oracle_call, cumulative distinct silent classes) -- the discovery curve for plotting.
    discovery_curve: list[tuple[int, int]] = field(default_factory=list)

    @property
    def holes_per_call(self) -> float:
        return self.silent_holes / self.oracle_calls if self.oracle_calls else 0.0


def run_audit_logged(
    monitor: Monitor,
    oracle: Oracle,
    proposer: Proposer,
    *,
    budget: int,
    arm: str | None = None,
    target: str = "",
    seed: int = 0,
    state: State = EMPTY,
    ctx: Context = EMPTY,
) -> RunLog:
    """Probe ``monitor`` with ``proposer``'s stream in emission order, one oracle call per distinct
    action, capped at ``budget`` calls. Record the ordered discovery of in-contract soundness holes.

    The silent-miss predicate is identical to :func:`audit`: a realizing action the monitor does not
    cover and that is *in the monitor's contract* (so the miss is a genuine soundness bug, not the
    routed residual). Deduplicates by (command, op) exactly as ``audit`` does, so the realizing/hole
    counts match the certificate over the same deterministic sample.
    """
    log = RunLog(
        arm=arm if arm is not None else str(getattr(proposer, "name", "?")),
        monitor=monitor.name,
        oracle=oracle.name,
        target=target,
        seed=seed,
        budget=budget,
    )
    seen: set[tuple[str, tuple[str, ...]]] = set()
    silent_keys: set[tuple[str, tuple[str, ...]]] = set()
    seen_classes: dict[str, None] = {}  # insertion-ordered set of silent klasses
    t0 = time.perf_counter()

    for a in proposer.propose(budget):
        key = (a.command, a.op)
        if key in seen:
            continue
        seen.add(key)
        if log.oracle_calls >= budget:
            break
        log.oracle_calls += 1
        log.n_probed += 1
        call = log.oracle_calls

        if not oracle.realizes(a, state):
            log.discovery_curve.append((call, len(seen_classes)))
            continue
        log.n_realizing += 1

        if monitor.covers(a, ctx):
            log.discovery_curve.append((call, len(seen_classes)))
            continue

        if not _in_contract(monitor, a, ctx):
            # out-of-contract realizing action: the honest routed residual (post-commit diff or
            # human), never a silent soundness bug. Counted, not a hole.
            log.residual_routed += 1
            log.discovery_curve.append((call, len(seen_classes)))
            continue

        # a genuine silent miss: realizing, uncovered, in-contract.
        if key not in silent_keys:
            silent_keys.add(key)
            log.silent_holes += 1
            if log.first_silent_call is None:
                log.first_silent_call = call
            if a.klass not in seen_classes:
                seen_classes[a.klass] = None
                log.first_class_call[a.klass] = call
        log.discovery_curve.append((call, len(seen_classes)))

    log.wall_clock_s = time.perf_counter() - t0
    log.silent_classes = list(seen_classes)
    return log
