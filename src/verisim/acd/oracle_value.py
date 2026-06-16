"""SPEC-22 CU33 (H126): the value of the oracle -- the cost-optimal verification policy.

The whole CU arc reports two numbers per policy: a breach rate and an oracle-call count. It never
puts them in a common currency, so every "Nx cheaper" claim treats safety and cost as incomparable.
A defender's real objective is **expected operational loss**::

    L(policy) = C * p_breach(policy)  +  c * calls(policy)

where ``C`` is the cost of one breach (a successful exfil / corruption / outage -- the catastrophic
loss) and ``c`` is the cost of one oracle call (the verifier's wall-clock / money / analyst time).
Both are real, defender-supplied prices. CU33 converts the arc into a CISO's decision rule.

THE NON-OBVIOUS MECHANISM. The natural practitioner mental model is a *tuning dial*: "spend more
verification when stakes are high, less when low, find the sweet spot." Under a worst-case adversary
that dial is an **illusion**, and the reason is the CU11/CU21 coverage theorem:

  - Every NON-COVERING policy (uniform at any rho<1, model self-targeting, an asset shortcut) has
    adversarial ``p_breach = 1`` -- pinned, at *every* sub-oracle budget. So raising its budget only
    raises ``c * calls`` while the breach stays catastrophic: every interior budget is strictly
    worse than doing nothing. There is no sweet spot to tune to.
  - Every COVERING policy (the structure target, the full oracle) has adversarial ``p_breach = 0``.
    Its loss is purely ``c * calls``, and the structure target Pareto-DOMINATES the full oracle
    (same zero breach, strictly fewer calls) -- so the full oracle is never cost-optimal.

The efficient frontier under an adversary therefore collapses to TWO points: **accept the loss**
(the free agent, ``L = C``) or **cover it** (the structure target, ``L = c * calls_structure``). The
whole decision is one threshold::

    verify the structure surface  <=>  C / c  >  calls_structure

i.e. buy the oracle the moment a breach costs more than the (tiny) covering surface -- a handful of
oracle calls. Below it, the honest answer is *don't verify*: a low-stakes / expensive-verifier
regime where the oracle is not worth its price.

THE HONEST CONTRAST (the dial is real against nature). Against a RANDOM workload uniform's breach
falls smoothly with the budget, so the loss ``C * rand(rho) + c * calls(rho)`` does have an interior
optimum -- the dial works. It is the *adversary* that destroys it: with the breach pinned at 1 for
every rho<1, the adversarial loss is non-decreasing in the budget until the full-oracle cliff, so
the optimal uniform budget is binary (0 or 1), never interior, and even its high end is beaten by
the structure target. Cyber is adversarial, so the dial a practitioner reaches for does not exist;
the coverage theorem replaces it with a binary "cover or accept the loss", and coverage is cheap.

Torch-free; reuses :func:`verisim.acd.unified_targeting.run_arm` across all four worlds. The
economic layer (:func:`expected_loss`, :func:`critical_ratio`, :func:`optimal_policy`) is a set of
pure functions over the per-policy ``(breach, calls)`` points the arc already measures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from verisim.acd.unified_targeting import (
    Arm,
    ArmResult,
    Cell,
    _uniform_knee,
    host_arm,
    net_flow_arm,
    run_arm,
)

# --------------------------------------------------------------------------------------------------
# The economic layer: pure functions over per-policy (breach, calls) points.
# --------------------------------------------------------------------------------------------------

_EPS = 1e-9


@dataclass(frozen=True)
class PolicyPoint:
    """One policy's operating point: its worst-case + average breach and its mean oracle calls.

    ``adversarial_breach`` is the cyber-relevant (worst-case-over-attacker-timing) breach rate;
    ``random_breach`` is the breach against a random workload (nature). ``calls`` is the mean number
    of oracle consultations per deployment -- the thing ``c`` is the unit price of.
    """

    name: str
    label: str
    adversarial_breach: float
    random_breach: float
    calls: float


def expected_loss(
    point: PolicyPoint, breach_cost: float, call_cost: float, *, adversarial: bool = True
) -> float:
    """Expected operational loss ``C * p_breach + c * calls`` (worst-case breach by default)."""
    breach = point.adversarial_breach if adversarial else point.random_breach
    return breach_cost * breach + call_cost * point.calls


def dominates(a: PolicyPoint, b: PolicyPoint, *, adversarial: bool = True) -> bool:
    """Pareto dominance on ``(breach, calls)``: ``a`` weakly beats ``b`` on both, strictly on one.

    A dominated policy is never cost-optimal for any positive ``(C, c)`` -- a total-order fact, not
    a numeric coincidence. (Used to show the structure target dominates the full oracle.)
    """
    ab = a.adversarial_breach if adversarial else a.random_breach
    bb = b.adversarial_breach if adversarial else b.random_breach
    weak = ab <= bb + _EPS and a.calls <= b.calls + _EPS
    strict = ab < bb - _EPS or a.calls < b.calls - _EPS
    return weak and strict


def optimal_policy(
    points: list[PolicyPoint], breach_cost: float, call_cost: float, *, adversarial: bool = True
) -> PolicyPoint:
    """The expected-loss-minimizing policy at one ``(C, c)`` (ties broken toward fewer calls)."""
    return min(
        points,
        key=lambda p: (expected_loss(p, breach_cost, call_cost, adversarial=adversarial), p.calls),
    )


def critical_ratio(
    structure: PolicyPoint, baseline: PolicyPoint, *, adversarial: bool = True
) -> float:
    """The ``C/c`` at which the covering policy's loss equals the do-nothing baseline's.

    Solving ``ratio * b_base + calls_base = ratio * b_struct + calls_struct`` for ratio. With the
    free baseline (breach 1, calls 0) and a covering target (breach 0), this is exactly
    ``calls_structure`` -- "verify whenever a breach costs more than this many oracle calls."
    Returns +inf if the two breaches are equal (no crossing).
    """
    bb = baseline.adversarial_breach if adversarial else baseline.random_breach
    bs = structure.adversarial_breach if adversarial else structure.random_breach
    denom = bb - bs
    if denom <= _EPS:
        return float("inf")
    return (structure.calls - baseline.calls) / denom


# --------------------------------------------------------------------------------------------------
# Extracting the policy points from a unified-targeting ArmResult.
# --------------------------------------------------------------------------------------------------


def _point(cell: Cell, name: str, label: str) -> PolicyPoint:
    return PolicyPoint(
        name=name, label=label,
        adversarial_breach=cell.adversarial_breach,
        random_breach=cell.random_breach,
        calls=cell.mean_calls,
    )


# The standard policy set the arc compares (free = do-nothing; structure = the covering target).
POLICY_ORDER = ("free", "uniform_knee", "model", "structure", "full_oracle")
_NONCOVERING = ("uniform_knee", "model")


def policy_points(arm: ArmResult) -> list[PolicyPoint]:
    """The five comparable operating points: free / uniform-knee / model / structure / full."""
    free = arm.uniform[0]
    knee = _uniform_knee(arm)
    return [
        _point(free, "free", "free (do nothing)"),
        _point(knee, "uniform_knee", f"uniform knee (rho={knee.rho:g})"),
        _point(arm.model, "model", "model self-targeting"),
        _point(arm.target, "structure", f"structure ({arm.target_name})"),
        _point(arm.full_oracle, "full_oracle", "full oracle"),
    ]


# --------------------------------------------------------------------------------------------------
# The dial: the uniform breach-vs-budget curve, under nature vs an adversary.
# --------------------------------------------------------------------------------------------------


def _strictly_decreasing(xs: list[float]) -> bool:
    from itertools import pairwise

    return all(b <= a + _EPS for a, b in pairwise(xs)) and xs[-1] < xs[0] - _EPS


def _flat_until_last(xs: list[float]) -> bool:
    """The adversarial dial: constant across every sub-oracle budget, dropping only at the last (the
    full oracle). A flat dial that does nothing until you pay for everything."""
    if len(xs) < 2:
        return False
    head = xs[:-1]
    return max(head) - min(head) <= _EPS and xs[-1] < head[-1] - _EPS


# --------------------------------------------------------------------------------------------------
# Per-arm economics + the swept curves for the figure.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmEconomics:
    world_name: str
    points: list[PolicyPoint]
    structure_calls: float
    full_calls: float
    structure_call_saving: float          # full / structure (the legibility number)
    critical_ratio_units: float           # the C/c deploy threshold (= structure calls)
    structure_dominates_full: bool        # the full oracle is Pareto-dominated -> never optimal
    full_oracle_ever_optimal: bool        # over the ratio grid (predicted: False)
    noncovering_ever_optimal: bool        # uniform-knee / model ever the argmin (predicted: False)

    def point(self, name: str) -> PolicyPoint:
        for p in self.points:
            if p.name == name:
                return p
        raise KeyError(name)


@dataclass(frozen=True)
class DialPoint:
    """The uniform schedule at one budget rho: its breach vs nature and vs an adversary, and cost.

    Sweeping rho traces "the dial" a practitioner reaches for. The random breach slopes down with
    the budget (a real dial); the adversarial breach is flat (the dial does nothing) until the
    full-oracle cliff.
    """

    rho: float
    random_breach: float
    adversarial_breach: float
    calls: float


@dataclass(frozen=True)
class LossPoint:
    """The expected loss (in units of ``c``) of each frontier policy at one stake ratio."""

    ratio: float
    free_loss: float
    structure_loss: float
    full_loss: float
    optimal_name: str


@dataclass(frozen=True)
class CU33Result:
    horizon: int
    ratios: tuple[float, ...]
    arms: list[ArmEconomics]
    demo_world: str
    dial_curve: list[DialPoint]           # uniform breach vs budget (the dial), demo arm
    loss_curve: list[LossPoint]           # expected loss vs stakes (value of the oracle), demo arm
    nature_dial_is_sloped: bool           # random breach falls with the budget (a real dial)
    adversary_dial_is_flat: bool          # adversarial breach is flat until the full-oracle cliff

    def arm(self, world_name: str) -> ArmEconomics:
        for a in self.arms:
            if a.world_name == world_name:
                return a
        raise KeyError(world_name)


@dataclass(frozen=True)
class CU33Config:
    """The economic sweep: which arms, the stake-ratio grid, and the demo arm for the curves."""

    ratios: tuple[float, ...] = (
        0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 18.0, 26.0, 40.0, 60.0, 100.0, 200.0
    )
    demo_world: str = "network"
    arm_factory: Callable[[], list[Arm]] | None = None

    def arms(self) -> list[Arm]:
        if self.arm_factory is not None:
            return self.arm_factory()
        # Network (the most-quoted arm) + host (the cross-world confirmation). Both torch-free.
        return [net_flow_arm(), host_arm()]


def _arm_economics(arm: ArmResult, ratios: tuple[float, ...]) -> ArmEconomics:
    points = policy_points(arm)
    by = {p.name: p for p in points}
    structure, full = by["structure"], by["full_oracle"]
    saving = full.calls / structure.calls if structure.calls > _EPS else float("inf")
    crit = critical_ratio(structure, by["free"])
    grid_breach = [r / 1.0 for r in ratios]  # ratio = C/c, work in units of c (call_cost = 1)
    full_ever = any(optimal_policy(points, r, 1.0).name == "full_oracle" for r in grid_breach)
    nc_ever = any(
        optimal_policy(points, r, 1.0).name in _NONCOVERING for r in grid_breach
    )
    return ArmEconomics(
        world_name=arm.world_name, points=points,
        structure_calls=structure.calls, full_calls=full.calls,
        structure_call_saving=saving, critical_ratio_units=crit,
        structure_dominates_full=dominates(structure, full),
        full_oracle_ever_optimal=full_ever, noncovering_ever_optimal=nc_ever,
    )


def run_cu33(config: CU33Config | None = None) -> CU33Result:
    """Run the unified arms, then layer the economic decision rule + the dial-collapse curves."""
    config = config or CU33Config()
    arms = config.arms()
    results = [run_arm(a) for a in arms]
    econ = [_arm_economics(r, config.ratios) for r in results]

    demo = next((r for r in results if r.world_name == config.demo_world), results[0])
    demo_econ = next(e for e in econ if e.world_name == demo.world_name)
    free = demo_econ.point("free")
    structure = demo_econ.point("structure")
    full = demo_econ.point("full_oracle")

    dial_curve = [
        DialPoint(
            rho=c.rho if c.rho is not None else 1.0,
            random_breach=c.random_breach,
            adversarial_breach=c.adversarial_breach,
            calls=c.mean_calls,
        )
        for c in demo.uniform
    ]
    loss_curve = [
        LossPoint(
            ratio=r,
            free_loss=expected_loss(free, r, 1.0),
            structure_loss=expected_loss(structure, r, 1.0),
            full_loss=expected_loss(full, r, 1.0),
            optimal_name=optimal_policy(demo_econ.points, r, 1.0).name,
        )
        for r in config.ratios
    ]
    nature_sloped = _strictly_decreasing([p.random_breach for p in dial_curve])
    adversary_flat = _flat_until_last([p.adversarial_breach for p in dial_curve])
    return CU33Result(
        horizon=demo.horizon, ratios=config.ratios, arms=econ, demo_world=demo.world_name,
        dial_curve=dial_curve, loss_curve=loss_curve,
        nature_dial_is_sloped=nature_sloped, adversary_dial_is_flat=adversary_flat,
    )


def cu33_verdict(result: CU33Result) -> dict[str, object]:
    """H126: under an adversary the safety/cost dial collapses to a binary "cover or accept the
    loss"; the structure target dominates the full oracle and is optimal above a tiny C/c threshold;
    the dial is real only against nature.
    """
    arms = result.arms
    return {
        "n_worlds": len(arms),
        # the dominance / optimality facts (the dial collapse)
        "structure_dominates_full_everywhere": all(a.structure_dominates_full for a in arms),
        "full_oracle_never_optimal_everywhere": all(not a.full_oracle_ever_optimal for a in arms),
        "noncovering_never_optimal_everywhere": all(not a.noncovering_ever_optimal for a in arms),
        # the decision rule: verify iff C/c > calls_structure (a handful of oracle calls)
        "critical_ratios": {a.world_name: a.critical_ratio_units for a in arms},
        "max_critical_ratio": max(a.critical_ratio_units for a in arms),
        "structure_call_savings": {a.world_name: a.structure_call_saving for a in arms},
        # the honest contrast: the dial is real vs nature, an illusion vs an adversary
        "nature_dial_is_sloped": result.nature_dial_is_sloped,
        "adversary_dial_is_flat": result.adversary_dial_is_flat,
        "dial_collapses_under_adversary": result.nature_dial_is_sloped
        and result.adversary_dial_is_flat,
        "demo_world": result.demo_world,
        "horizon": result.horizon,
    }


CSV_HEADER = "section,world,policy,ratio,adversarial_breach,random_breach,calls,value"


def write_csv(result: CU33Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for a in result.arms:
        for p in a.points:
            rows.append(
                f"point,{a.world_name},{p.name},,{p.adversarial_breach:.6f},"
                f"{p.random_breach:.6f},{p.calls:.6f},"
            )
        rows.append(f"critical_ratio,{a.world_name},structure,,,,{a.structure_calls:.6f},"
                    f"{a.critical_ratio_units:.6f}")
    for d in result.dial_curve:
        rows.append(
            f"dial,{result.demo_world},uniform,{d.rho:.3f},{d.adversarial_breach:.6f},"
            f"{d.random_breach:.6f},{d.calls:.6f},"
        )
    for lp in result.loss_curve:
        rows.append(
            f"loss,{result.demo_world},free,{lp.ratio:.3f},,,,{lp.free_loss:.6f}"
        )
        rows.append(
            f"loss,{result.demo_world},structure,{lp.ratio:.3f},,,,{lp.structure_loss:.6f}"
        )
        rows.append(
            f"loss,{result.demo_world},full_oracle,{lp.ratio:.3f},,,,{lp.full_loss:.6f}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
