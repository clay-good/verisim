"""SPEC-22 CU32 (H125): the verification-latency barrier -- the THROUGHPUT cost of safety.

The entire CU arc prices verification in **oracle calls**: CU10-CU26 minimize *how many* consults
cover the danger, CU15 attacks that budget, CU27 routes the costly preview onto the irreversible
slice. Every one of those results assumes the consult is **instantaneous and blocking** -- the gate
pauses, asks the oracle, and the verdict is back before the agent commits. A real verifier is not
free in wall-clock. Replaying a plan through a sandbox or a real kernel takes time (CU28's
``SandboxOracle`` is ~5ms/step over a multi-step preview); a SOC analyst or a SOAR approval gate
takes seconds to minutes. Verification has a **latency** ``L``, and during ``L`` the agent is either
stalled or running ahead. CU32 opens that axis: *given a verifier with latency ``L``, how fast can a
safe agent act, and what does safety cost in THROUGHPUT?*

THE LATENCY-HIDING TRAP. The obvious engineering move to hide latency is to **pipeline**: issue the
consult but do not wait -- commit the action speculatively and reconcile when the verdict lands
``L`` steps later (exactly how a CPU hides memory latency). On a **reversible** danger this is safe:
the late verdict still arrives in time to **roll back** to the journaled pre-state (the CU27
after-commit discipline, now merely deferred). But on an **irreversible** danger the pipeline is a
trap: the send has already left the boundary by the time the verdict confirms it was a breach. The
verdict is correct and useless. Pipelining re-opens **exactly the irreversible slice CU27 isolated**
-- the one place after-commit could never save.

THE LATENCY THEOREM. Under verification latency ``L > 0``, safety on the irreversible slice requires
a **synchronous barrier**: the agent must *stall* ``L`` wall-clock units before committing an
irreversible, covered action -- there is no latency-hiding for irreversibility. Reversible actions
need no barrier: their consult can be pipelined (deferred), costing at worst a deeper rollback on
the rare realized breach, never a stall. So the routing CU27 derived from reversibility -- stall the
irreversible, defer the reversible -- is *also* the routing that minimizes throughput cost under
latency. The minimum throughput cost of safety is ``L`` times the number of irreversible covered
consults, and nothing more.

THE COST LAW (the new quantitative content -- throughput, not call count). With wall-clock
``= horizon + L * stalls`` and throughput ``= horizon / wall-clock``:

  - ``pipeline_all`` (hide all latency): throughput **1.0** (never stalls) but adversarially
    breached on the irreversible slice -- fast and unsafe;
  - ``barrier_all`` (stall every consult, the naive safe gate): safe everywhere but stalls on the
    reversible consults too, so throughput decays in ``L`` faster than it must;
  - ``routed`` (stall irreversible / pipeline reversible): safe everywhere at throughput
    ``horizon / (horizon + L * irreversible_consults)`` -- the unique safe-and-fast corner, which
    recovers the ``(1 - reversible_fraction)`` of throughput ``barrier_all`` wastes.

As the irreversible fraction ``f -> 0`` the throughput cost of safety vanishes at *any* latency (a
fully reversible world is safe for free -- pipeline everything, roll back the rare breach); as
``f -> 1`` the safe action rate is capped at ``1 / (L * covered_rate)``. **Verification latency
makes safety cost throughput, and the bill is ``L`` times the irreversible danger rate: you can
defer verification you can undo, you must stall for verification you can't.**

Substrate: the two CU21 network dangers CU27 already grounds -- exfiltration (an irreversible
*send*, ``net_flow_arm``) and segmentation exposure (a reversible *posture*, ``net_reach_arm``) --
so the danger genesis, attacker arsenals, and covering targets are bit-identical to CU21/CU27.
Torch-free, deterministic; the worst-case omitter is the model (the schedule keys on the oracle, the
covering surface, and reversibility, never the model's competence). **Refuted if**
``pipeline_all`` is
adversarially safe on the irreversible slice, ``routed`` is not safe everywhere, routed throughput
does not strictly exceed ``barrier_all`` for ``L > 0`` and a non-trivial reversible surface, or the
throughput cost does not scale with ``L`` and the irreversible fraction ``f``.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from verisim.acd.unified_targeting import (
    Danger,
    OmitterDefender,
    Target,
    World,
    net_flow_arm,
    net_reach_arm,
)

State = object
Action = object

Policy = Literal["unverified", "pipeline_all", "barrier_all", "routed"]
POLICIES: tuple[Policy, ...] = ("unverified", "pipeline_all", "barrier_all", "routed")

DangerClass = Literal["reversible", "irreversible"]


@dataclass(frozen=True)
class LatScenario:
    """One deployment + the static reversibility of its danger (a true property, not a label).

    ``reversible`` is whether reverting to the journaled pre-state undoes the breach: a segmentation
    posture lives in state -> reversible; an exfil *send* has left the boundary -> irreversible.
    ``target`` is the CU21 covering surface (the consult rule); the consult itself costs ``L``
    wall-clock when the policy stalls for it.
    """

    world: World
    start: State
    actions: tuple[Action, ...]
    danger: Danger
    target: Target
    reversible: bool


def _stalls_for(policy: Policy, reversible: bool) -> bool:
    """Does this policy STALL (synchronously block ``L``) on a consulted action of this class?

    ``barrier_all`` stalls every consult; ``routed`` stalls only the irreversible class (the
    reversible class is pipelined/deferred); ``pipeline_all`` and ``unverified`` never stall.
    """
    if policy == "barrier_all":
        return True
    if policy == "routed":
        return not reversible
    return False


def _consults(policy: Policy) -> bool:
    """Does this policy consult the oracle at all? (``unverified`` trusts the blind model.)"""
    return policy != "unverified"


def run_policy(
    sc: LatScenario, policy: Policy, latency: int
) -> tuple[bool, int, int, int]:
    """Run the benign mission under one policy at latency ``L``.

    Returns ``(breached, consults, stalls, rollbacks)``. The covering target fires on the danger
    surface; a consulted action either STALLS (synchronous barrier -> the oracle blocks a true
    breach before it commits, never breaches) or is PIPELINED (committed now, verdict reconciled
    ``L`` steps later -> a reversible breach is rolled back, an irreversible breach has already
    left).
    Wall-clock is ``horizon + L * stalls`` (see :func:`throughput`).
    """
    omitter = OmitterDefender()
    state = sc.start
    breached = False
    consults = stalls = rollbacks = 0
    for action in sc.actions:
        realizes = sc.danger.realizes(state, action)
        on_surface = sc.target(state, action)
        if _consults(policy) and on_surface:
            consults += 1
            if _stalls_for(policy, sc.reversible):  # synchronous barrier: block before commit
                stalls += 1
                if realizes:
                    continue  # the oracle blocked the danger -> skip the action, no breach
            else:  # pipelined: commit now, reconcile the verdict L steps late
                if realizes and sc.reversible:
                    rollbacks += 1  # late verdict still in time to roll back -> averted
                elif realizes:
                    breached = True  # irreversible: the send left before the verdict returned
        else:  # not consulted -> trust the blind model (which foresees nothing)
            if realizes and not omitter.foresees(state, action):
                breached = True
        state = sc.world.advance(state, action)
    return breached, consults, stalls, rollbacks


def _attack_breaches(sc: LatScenario, policy: Policy, state: State, atk: Action) -> bool:
    """Does an injected danger action (each ``atk`` realizes) win against this policy from here?"""
    on_surface = sc.target(state, atk)
    if not (_consults(policy) and on_surface):
        return True  # unconsulted (or unverified) -> the blind model misses it
    if _stalls_for(policy, sc.reversible):
        return False  # synchronous barrier: the oracle blocks the realized danger before commit
    return not sc.reversible  # pipelined: reversible -> rolled back; irreversible -> the send left


def adversarial_policy(sc: LatScenario, policy: Policy) -> bool:
    """Worst-case over the attacker's choice of danger action AND timing (the CU11 probe)."""
    state = sc.start
    for action in sc.actions:
        for atk in sc.danger.attacks(state):
            if _attack_breaches(sc, policy, state, atk):
                return True
        state = sc.world.advance(state, action)
    return False


# --------------------------------------------------------------------------------------------------
# The two arms: irreversible (exfil send) + reversible (segmentation posture), from CU21 grounding.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CU32Config:
    max_episodes: int = 60  # cap per danger class (the net arms over-produce deployments)
    latency: int = 8  # the reference verifier latency L (wall-clock units per stalled consult)
    latencies: tuple[int, ...] = (0, 1, 2, 4, 8, 16)  # the safe-throughput-vs-latency sweep
    fractions: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)  # the irreversible-mix sweep

    @staticmethod
    def smoke() -> CU32Config:
        return CU32Config(max_episodes=8, latencies=(0, 4, 16), fractions=(0.0, 0.5, 1.0))


def _lat_scenarios(config: CU32Config) -> tuple[list[LatScenario], list[LatScenario], int]:
    """Build the irreversible (exfil) and reversible (exposure) deployment pools + the horizon."""
    exfil = net_flow_arm()  # CU10: a flow to a crown jewel -- an irreversible send
    expose = net_reach_arm()  # CU17: a jewel reachable from the untrusted set -- reversible posture
    irreversible = [
        LatScenario(exfil.world, sc.start, sc.actions, sc.danger, sc.target, reversible=False)
        for sc in exfil.scenarios[: config.max_episodes]
    ]
    reversible = [
        LatScenario(expose.world, sc.start, sc.actions, sc.danger, sc.target, reversible=True)
        for sc in expose.scenarios[: config.max_episodes]
    ]
    return reversible, irreversible, exfil.horizon


def throughput(horizon: int, mean_stalls: float, latency: int) -> float:
    """Safe action rate: actions per wall-clock unit, ``horizon / (horizon + L*stalls)``."""
    wall = horizon + latency * mean_stalls
    return horizon / wall if wall > 0 else 0.0


@dataclass(frozen=True)
class PolicyCell:
    """One (policy, danger class): breach (random + adversarial), consult/stall cost, throughput."""

    policy: str
    danger_class: str
    random_breach: float
    adversarial_breach: float
    mean_consults: float
    mean_stalls: float
    throughput: float  # at the reference latency L


def _cell(
    scenarios: list[LatScenario], policy: Policy, danger_class: DangerClass, horizon: int,
    latency: int,
) -> PolicyCell:
    runs = [run_policy(sc, policy, latency) for sc in scenarios]
    adv = [adversarial_policy(sc, policy) for sc in scenarios]
    mean_stalls = fmean(s for _, _, s, _ in runs) if runs else 0.0
    return PolicyCell(
        policy=policy,
        danger_class=danger_class,
        random_breach=fmean(b for b, _, _, _ in runs) if runs else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_consults=fmean(c for _, c, _, _ in runs) if runs else 0.0,
        mean_stalls=mean_stalls,
        throughput=throughput(horizon, mean_stalls, latency),
    )


@dataclass(frozen=True)
class LatencyPoint:
    """Safe-throughput-vs-latency curve: each policy's throughput at latency ``L`` (50/50 mix)."""

    latency: int
    pipeline_all_throughput: float
    barrier_all_throughput: float
    routed_throughput: float


@dataclass(frozen=True)
class FractionPoint:
    """Irreversible-mix sweep at the reference ``L``: throughput + pipeline's residual breach."""

    irreversible_fraction: float
    routed_throughput: float
    barrier_all_throughput: float
    pipeline_all_throughput: float
    pipeline_all_breach: float  # residual adversarial breach if you hide all latency


@dataclass(frozen=True)
class CU32Result:
    horizon: int
    latency: int
    n_reversible: int
    n_irreversible: int
    cells: list[PolicyCell]
    latency_curve: list[LatencyPoint]
    fraction_law: list[FractionPoint]

    def cell(self, policy: str, danger_class: str) -> PolicyCell:
        for c in self.cells:
            if c.policy == policy and c.danger_class == danger_class:
                return c
        raise KeyError(f"no cell for {policy}/{danger_class}")


def _mixed_stalls(rev: PolicyCell, irr: PolicyCell, f: float) -> float:
    """Mean stalls of a pool fraction ``f`` irreversible (a weighted average of the classes)."""
    return f * irr.mean_stalls + (1 - f) * rev.mean_stalls


def run_cu32(config: CU32Config | None = None) -> CU32Result:
    """Score the four policies per danger class, then sweep throughput over ``L`` and the mix."""
    config = config or CU32Config()
    reversible, irreversible, horizon = _lat_scenarios(config)
    L = config.latency

    cells: list[PolicyCell] = []
    for policy in POLICIES:
        cells.append(_cell(reversible, policy, "reversible", horizon, L))
        cells.append(_cell(irreversible, policy, "irreversible", horizon, L))

    def look(policy: str, danger_class: str) -> PolicyCell:
        return next(c for c in cells if c.policy == policy and c.danger_class == danger_class)

    # The safe-throughput-vs-latency curve, on a balanced 50/50 mix (stalls = average of the two).
    latency_curve: list[LatencyPoint] = []
    for lat in config.latencies:
        def tput(policy: str, lat: int = lat) -> float:
            stalls = _mixed_stalls(look(policy, "reversible"), look(policy, "irreversible"), 0.5)
            return throughput(horizon, stalls, lat)
        latency_curve.append(LatencyPoint(
            latency=lat,
            pipeline_all_throughput=tput("pipeline_all"),
            barrier_all_throughput=tput("barrier_all"),
            routed_throughput=tput("routed"),
        ))

    # The irreversible-mix sweep at the reference L: routed recovers throughput as f -> 0, while
    # pipeline_all keeps full throughput but pays a residual breach that grows with f.
    rt_rev, rt_irr = look("routed", "reversible"), look("routed", "irreversible")
    ba_rev, ba_irr = look("barrier_all", "reversible"), look("barrier_all", "irreversible")
    pa_rev, pa_irr = look("pipeline_all", "reversible"), look("pipeline_all", "irreversible")
    fraction_law = [
        FractionPoint(
            irreversible_fraction=f,
            routed_throughput=throughput(horizon, _mixed_stalls(rt_rev, rt_irr, f), L),
            barrier_all_throughput=throughput(horizon, _mixed_stalls(ba_rev, ba_irr, f), L),
            pipeline_all_throughput=throughput(horizon, _mixed_stalls(pa_rev, pa_irr, f), L),
            pipeline_all_breach=f * pa_irr.adversarial_breach + (1 - f) * pa_rev.adversarial_breach,
        )
        for f in config.fractions
    ]

    return CU32Result(
        horizon=horizon, latency=L,
        n_reversible=len(reversible), n_irreversible=len(irreversible),
        cells=cells, latency_curve=latency_curve, fraction_law=fraction_law,
    )


def _is_increasing(xs: list[float]) -> bool:
    from itertools import pairwise

    return all(b >= a - 1e-9 for a, b in pairwise(xs)) and xs[-1] > xs[0] + 1e-9


def _is_decreasing(xs: list[float]) -> bool:
    from itertools import pairwise

    return all(b <= a + 1e-9 for a, b in pairwise(xs)) and xs[-1] < xs[0] - 1e-9


def cu32_verdict(result: CU32Result) -> dict[str, object]:
    """H125: pipelining (latency-hiding) re-breaches the irreversible slice; routing by
    reversibility
    is safe everywhere and is the cheapest-throughput safe policy; safety costs ``L`` x irreversible
    rate.
    """
    pa_rev = result.cell("pipeline_all", "reversible")
    pa_irr = result.cell("pipeline_all", "irreversible")
    rt_rev = result.cell("routed", "reversible")
    rt_irr = result.cell("routed", "irreversible")
    ba_rev = result.cell("barrier_all", "reversible")
    ba_irr = result.cell("barrier_all", "irreversible")

    routed_safe = max(
        rt_rev.random_breach, rt_rev.adversarial_breach,
        rt_irr.random_breach, rt_irr.adversarial_breach,
    )
    barrier_safe = max(
        ba_rev.adversarial_breach, ba_irr.adversarial_breach,
    )
    # the safe-throughput curve: routed strictly beats barrier_all for L>0 (it never stalls the
    # reversible consults), and both decay with L; pipeline_all is flat at 1.0 (and unsafe).
    pos_L = [p for p in result.latency_curve if p.latency > 0]
    routed_beats_barrier = all(
        p.routed_throughput > p.barrier_all_throughput + 1e-9 for p in pos_L
    ) if pos_L else False
    # the representative throughput saving is on a MIXED pool: on a fully-irreversible pool routed
    # == barrier (every irreversible consult must stall), so the saving is exactly the reversible
    # fraction you pipeline. Read it at the reference L on the balanced 50/50 mix.
    ref = next((p for p in result.latency_curve if p.latency == result.latency), None)
    saving_mixed = (
        ref.routed_throughput / ref.barrier_all_throughput
        if ref is not None and ref.barrier_all_throughput > 0 else float("inf")
    )
    law = result.fraction_law
    return {
        # the latency-hiding trap: pipeline_all is safe on reversible, breached on irreversible
        "pipeline_reversible_safe": pa_rev.adversarial_breach <= 1e-9,
        "pipeline_irreversible_fails": pa_irr.adversarial_breach > 0.5,
        "pipeline_irreversible_adv_breach": pa_irr.adversarial_breach,
        "pipeline_full_throughput": all(
            abs(p.pipeline_all_throughput - 1.0) <= 1e-9 for p in result.latency_curve
        ),
        # routing by reversibility: safe everywhere, and the only safe policy that stalls only the
        # irreversible slice (reversible consults are pipelined -> zero stalls -> free throughput)
        "routed_safe_everywhere": routed_safe <= 1e-9,
        "routed_reversible_never_stalls": rt_rev.mean_stalls <= 1e-9,
        "barrier_all_safe_everywhere": barrier_safe <= 1e-9,
        # the throughput law: safety costs L x irreversible rate; routed beats barrier for L>0
        "routed_beats_barrier_for_latency": routed_beats_barrier,
        "routed_throughput_at_L": rt_irr.throughput,  # the irreversible (worst) class at the ref L
        "barrier_all_throughput_at_L": ba_irr.throughput,
        # the representative saving (50/50 mix at the ref L): routed pipelines the reversible half
        "routed_throughput_mixed": ref.routed_throughput if ref is not None else 0.0,
        "barrier_throughput_mixed": ref.barrier_all_throughput if ref is not None else 0.0,
        "throughput_saving_vs_barrier": saving_mixed,
        # decays with latency (both safe policies), flat-1 for pipeline
        "routed_throughput_decays_with_latency": _is_decreasing(
            [p.routed_throughput for p in result.latency_curve]
        ),
        # the mix law: routed throughput RISES as f -> 0 (safety free in a reversible world), while
        # pipeline_all keeps full throughput but its residual breach RISES with f
        "routed_throughput_rises_as_reversible": _is_decreasing(
            [p.routed_throughput for p in law]  # decreasing in f == rising as f -> 0
        ),
        "routed_free_when_fully_reversible": abs(law[0].routed_throughput - 1.0) <= 1e-9,
        "pipeline_breach_grows_with_irreversibility": _is_increasing(
            [p.pipeline_all_breach for p in law]
        ) and law[0].pipeline_all_breach <= 1e-9,
        "horizon": result.horizon,
        "latency": result.latency,
    }


CSV_HEADER = (
    "section,policy,danger_class,latency,irreversible_fraction,random_breach,adversarial_breach,"
    "mean_consults,mean_stalls,throughput,pipeline_breach"
)


def write_csv(result: CU32Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in result.cells:
        rows.append(
            f"cell,{c.policy},{c.danger_class},{result.latency},,{c.random_breach:.6f},"
            f"{c.adversarial_breach:.6f},{c.mean_consults:.6f},{c.mean_stalls:.6f},"
            f"{c.throughput:.6f},"
        )
    for p in result.latency_curve:
        rows.append(
            f"latency_curve,pipeline_all,,{p.latency},,,,,,{p.pipeline_all_throughput:.6f},"
        )
        rows.append(
            f"latency_curve,barrier_all,,{p.latency},,,,,,{p.barrier_all_throughput:.6f},"
        )
        rows.append(
            f"latency_curve,routed,,{p.latency},,,,,,{p.routed_throughput:.6f},"
        )
    for q in result.fraction_law:
        rows.append(
            f"fraction_law,routed,,{result.latency},{q.irreversible_fraction:.3f},,,,,"
            f"{q.routed_throughput:.6f},"
        )
        rows.append(
            f"fraction_law,barrier_all,,{result.latency},{q.irreversible_fraction:.3f},,,,,"
            f"{q.barrier_all_throughput:.6f},"
        )
        rows.append(
            f"fraction_law,pipeline_all,,{result.latency},{q.irreversible_fraction:.3f},,,,,"
            f"{q.pipeline_all_throughput:.6f},{q.pipeline_all_breach:.6f}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
