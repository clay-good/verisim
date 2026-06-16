"""SPEC-22 CU35 (H128): the verifier-fidelity condition -- the *dual* coverage law.

The whole targeting arc (CU10-CU34) hard-codes one assumption it never names: when the schedule
*consults*, the oracle returns the **exact** verdict (``block = realizes`` in
:func:`~verisim.acd.unified_targeting.run_scenario`). The arc's only variable was the **target** --
*which* actions to consult -- and its central theorem is that the target must **cover** the danger
surface (``realizes => target``). But a real deployed verifier is not a free, bit-exact oracle: it
is itself a *model* of the world (a sandbox, an emulator, a cheaper reference) that can drift,
exactly along the structure/content boundary the foundation measured. CU35 turns that variable.

THE DUAL CONDITION. The arc gave one coverage condition on the *target*; CU35 gives the matching
condition on the *verifier*, and they are independent and both necessary:

  - **CU21 (target coverage):**  ``realizes(s, a) => target(s, a)``  -- the target must *fire* on
    every danger action (else the danger is never consulted; CU17/CU21 boundary).
  - **CU35 (verifier fidelity):**  for every danger action, the consulted verifier's verdict equals
    the oracle's -- the verifier must be *faithful on* the danger surface (else the danger is
    consulted but waved through).

THE NON-OBVIOUS PAYOFF -- LOCALIZATION. Under a covering target the verifier is consulted *only* on
the danger's (sparse) surface. So:

  - a verifier that is **globally wrong but exact on the danger grammar** is exactly as safe as a
    perfect oracle (breach 0): its off-surface drift is never consulted, and where it *is* consulted
    its verdict matters only for *utility* (false blocks), never for *safety*. A cheap sandbox is a
    safe verifier **iff** it is faithful on the danger surface -- a far weaker, checkable condition
    than a high-fidelity replica of production.
  - a verifier that is **faithful everywhere except the surface** (an on-surface omitter) is as
    blind as no gate at all -- breach returns to the unverified rate even with a perfectly covering
    target. This is CU8's omission bias, now relocated from the agent's world model into the
    *verifier* itself: a verifier hides danger by omission precisely where it omits.

So safety has TWO knobs and they are not symmetric: on-surface verifier fidelity is load-bearing
(a sloped faithful-horizon); off-surface fidelity is irrelevant to safety (flat), buying only
utility. This is the structure/content boundary (SPEC-10/SPEC-11) and the faithful horizon applied
to the verifier rather than the agent: the verifier needs faithfulness exactly where it gates, and
that surface is sparse.

Substrate: the worst-case methodology of the arc -- the exact reference oracle defines ``realizes``
(ground truth and the *ideal* verifier), the agent's own world model is the worst-case omitter
(consults nothing off-surface), and the deployed verifier is a parameterized imperfect model whose
fidelity on / off the danger surface we sweep. Torch-free, deterministic; reuses the CU21 net / host
arms verbatim.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from verisim.acd.unified_targeting import (
    Arm,
    Scenario,
    Target,
    World,
    covers,
    host_arm,
    net_flow_arm,
    net_reach_arm,
)

# --------------------------------------------------------------------------------------------------
# A parameterized imperfect verifier (the variable the arc never turned).
# --------------------------------------------------------------------------------------------------


def _unit(key: str) -> float:
    """A deterministic hash of ``key`` into [0, 1) (process-stable, unlike the salted built-in)."""
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class ExactVerifier:
    """The arc's hidden assumption made explicit: returns the oracle's exact verdict."""

    realizes: Callable[[object, object], bool]

    def verdict(self, state: object, action: object) -> bool:
        return bool(self.realizes(state, action))


@dataclass(frozen=True)
class SurfaceOmitter:
    """A verifier faithful OFF the danger surface but with a fidelity gap landing ON it.

    Off the surface (``not realizes``) it agrees with the oracle (no danger). On the surface it
    flags only a fraction ``phi`` of danger actions and OMITS the rest -- its blind spot is the
    danger grammar itself. ``phi=1`` is the exact verifier; ``phi=0`` is blind on the surface (the
    CU8 omission bias relocated into the verifier). Deterministic per action via :func:`_unit`.
    """

    realizes: Callable[[object, object], bool]
    phi: float

    def verdict(self, state: object, action: object) -> bool:
        if not self.realizes(state, action):
            return False
        return _unit(repr(action)) < self.phi


@dataclass(frozen=True)
class OffSurfaceDrifter:
    """A verifier EXACT on the danger surface but arbitrarily wrong off it.

    On the surface (``realizes``) it returns the oracle's verdict exactly. Off the surface it
    hallucinates danger on a fraction ``1 - psi`` of benign actions -- a low-fidelity sandbox that
    nonetheless models the danger grammar correctly. ``psi=1`` never hallucinates; ``psi=0``
    hallucinates everywhere off-surface. Under a covering target this NEVER causes a missed danger
    (every danger action is on-surface and flagged), only false blocks (a utility cost).
    """

    realizes: Callable[[object, object], bool]
    psi: float

    def verdict(self, state: object, action: object) -> bool:
        if self.realizes(state, action):
            return True
        return _unit("off:" + repr(action)) >= self.psi


class Verifier(Protocol):
    """Structural type: any object with ``verdict(state, action) -> bool`` (the consulted gate)."""

    def verdict(self, state: object, action: object) -> bool: ...


# A factory builds a verifier from a scenario's own ``realizes`` (each segment carries its own
# danger), so one fidelity setting applies consistently across an arm's heterogeneous scenarios.
Realizes = Callable[[object, object], bool]
MakeVerifier = Callable[[Realizes], Verifier]


def make_exact() -> MakeVerifier:
    return lambda realizes: ExactVerifier(realizes)


def make_surface_omitter(phi: float) -> MakeVerifier:
    return lambda realizes: SurfaceOmitter(realizes, phi)


def make_offsurface_drifter(psi: float) -> MakeVerifier:
    return lambda realizes: OffSurfaceDrifter(realizes, psi)


def _target_of(scenario: Scenario, *, use_shortcut: bool) -> Target:
    tgt = scenario.shortcut if use_shortcut else scenario.target
    return tgt if tgt is not None else (lambda s, a: False)


# --------------------------------------------------------------------------------------------------
# Running the covering-target schedule with an imperfect verifier on the consult.
# --------------------------------------------------------------------------------------------------


def run_with_verifier(
    world: World, scenario: Scenario, verifier: Verifier, target: Target,
) -> tuple[bool, int, int]:
    """Run the benign mission under the target schedule, consulting ``verifier`` (not the oracle).

    The agent consults iff ``target(s, a)``; on consult it gates on the *verifier's* verdict (which
    may drift), else it trusts its own world model -- the worst-case omitter (blocks nothing).
    Returns ``(breached, calls, false_blocks)``: a breach is a danger action executed uncaught; a
    false block is a benign (non-realizing) action the verifier wrongly aborted (the utility cost).
    """
    danger = scenario.danger
    state = scenario.start
    breached = False
    calls = 0
    false_blocks = 0
    for action in scenario.actions:
        realizes = danger.realizes(state, action)
        consult = bool(target(state, action))
        block = verifier.verdict(state, action) if consult else False
        if consult:
            calls += 1
        if realizes and not block:
            breached = True
        if block and not realizes:
            false_blocks += 1
        if not (realizes and block):
            state = world.advance(state, action)
    return breached, calls, false_blocks


def adversarial_with_verifier(
    world: World, scenario: Scenario, verifier: Verifier, target: Target,
) -> bool:
    """Worst case over the attacker's danger action and timing, gating on the imperfect verifier.

    Before each step the attacker tries every action in the danger's arsenal; the deployment is
    breached if any one is not blocked (not consulted by the target, or consulted but the verifier
    waves it through). The mirror of :func:`~verisim.acd.unified_targeting.adversarial_scenario`
    with the exact oracle replaced by ``verifier``.
    """
    danger = scenario.danger
    state = scenario.start
    for action in scenario.actions:
        for atk in danger.attacks(state):
            consult = bool(target(state, atk))
            block = verifier.verdict(state, atk) if consult else False
            if not block:
                return True
        realizes = danger.realizes(state, action)
        consult = bool(target(state, action))
        block = verifier.verdict(state, action) if consult else False
        if not (realizes and block):
            state = world.advance(state, action)
    return False


def faithful_on_surface(world: World, scenario: Scenario, verifier: Verifier) -> bool:
    """The dual of :func:`~verisim.acd.unified_targeting.covers`: does the verifier flag every
    danger action along the trajectory? (the verifier-fidelity condition's empirical check).
    """
    state = scenario.start
    for action in scenario.actions:
        for atk in scenario.danger.attacks(state):
            if not verifier.verdict(state, atk):
                return False
        state = world.advance(state, action)
    return True


# --------------------------------------------------------------------------------------------------
# The sweeps + the 2x2.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FidelityPoint:
    """One verifier fidelity setting on one arm: adversarial + benign breach + utility cost."""

    fidelity: float
    adversarial_breach: float
    random_breach: float
    mean_false_blocks: float


@dataclass(frozen=True)
class ArmFidelity:
    """One arm swept over on-surface (phi) and off-surface (psi) verifier fidelity."""

    world_name: str
    n_scenarios: int
    on_surface: list[FidelityPoint]  # SurfaceOmitter over phi (off-surface exact)
    off_surface: list[FidelityPoint]  # OffSurfaceDrifter over psi (on-surface exact)
    target_covers: bool


def _adv_rate(
    world: World, scenarios: list[Scenario], make: MakeVerifier, *, use_shortcut: bool = False,
) -> float:
    return fmean(
        adversarial_with_verifier(
            world, sc, make(sc.danger.realizes), _target_of(sc, use_shortcut=use_shortcut)
        )
        for sc in scenarios
    )


def _point(
    world: World, scenarios: list[Scenario], make: MakeVerifier, fidelity: float,
) -> FidelityPoint:
    runs = [
        run_with_verifier(world, sc, make(sc.danger.realizes), _target_of(sc, use_shortcut=False))
        for sc in scenarios
    ]
    return FidelityPoint(
        fidelity=fidelity,
        adversarial_breach=_adv_rate(world, scenarios, make),
        random_breach=fmean(b for b, _, _ in runs),
        mean_false_blocks=fmean(fb for _, _, fb in runs),
    )


def sweep_arm(arm: Arm, fidelities: tuple[float, ...], max_scenarios: int) -> ArmFidelity:
    """Sweep one arm's covering target over on-surface (phi) and off-surface (psi) fidelity.

    on_surface: the verifier is exact off the danger surface and faithful to degree ``phi`` ON it
    (SurfaceOmitter) -- predicts a sloped faithful-horizon (safety degrades as phi falls).
    off_surface: the verifier is exact ON the surface and drifts to degree ``psi`` off it
    (OffSurfaceDrifter) -- predicts flat safety (breach 0 for all psi) and rising false blocks.
    """
    scenarios = arm.scenarios[:max_scenarios]
    on_surface = [
        _point(arm.world, scenarios, make_surface_omitter(phi), phi) for phi in fidelities
    ]
    off_surface = [
        _point(arm.world, scenarios, make_offsurface_drifter(psi), psi) for psi in fidelities
    ]
    target_covers = all(covers(arm.world, sc) for sc in scenarios)
    return ArmFidelity(
        world_name=arm.world_name, n_scenarios=len(scenarios),
        on_surface=on_surface, off_surface=off_surface, target_covers=target_covers,
    )


@dataclass(frozen=True)
class TwoByTwo:
    """Adversarial breach across {target covers?} x {verifier faithful on surface?} on one arm."""

    world_name: str
    covers_exact: float  # covering target + exact verifier   (predicted: 0 -- the only safe corner)
    covers_blind: float  # covering target + on-surface-blind  (predicted: leaks -- CU35 negative)
    leak_exact: float  # non-covering target + exact verifier  (predicted: leaks -- CU21 negative)
    leak_blind: float  # non-covering target + on-surface-blind (predicted: leaks)


def two_by_two(arm: Arm, max_scenarios: int) -> TwoByTwo:
    """The independence of the two coverage conditions: only (covers AND faithful) is safe.

    Uses an arm that carries both a covering target and a non-covering shortcut (CU17 segmentation:
    the reachability closure covers; the connect shortcut does not). The verifier is either exact or
    blind on the danger surface (SurfaceOmitter phi=0).
    """
    scenarios = [sc for sc in arm.scenarios[:max_scenarios] if sc.shortcut is not None]
    if not scenarios:
        raise ValueError(f"arm {arm.world_name!r} has no shortcut for the 2x2")
    exact = make_exact()
    blind = make_surface_omitter(0.0)
    return TwoByTwo(
        world_name=arm.world_name,
        covers_exact=_adv_rate(arm.world, scenarios, exact),
        covers_blind=_adv_rate(arm.world, scenarios, blind),
        leak_exact=_adv_rate(arm.world, scenarios, exact, use_shortcut=True),
        leak_blind=_adv_rate(arm.world, scenarios, blind, use_shortcut=True),
    )


# --------------------------------------------------------------------------------------------------
# Config + top-level run + verdict.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CU35Config:
    fidelities: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    max_scenarios: int = 80

    @staticmethod
    def smoke() -> CU35Config:
        return CU35Config(fidelities=(0.0, 1.0), max_scenarios=8)


@dataclass(frozen=True)
class CU35Result:
    arms: list[ArmFidelity]
    grid: TwoByTwo


def run_cu35(config: CU35Config | None = None) -> CU35Result:
    """Run the fidelity sweeps (net + host) and the 2x2 (segmentation, covers + shortcut)."""
    config = config or CU35Config()
    arms = [
        sweep_arm(net_flow_arm(), config.fidelities, config.max_scenarios),
        sweep_arm(host_arm(), config.fidelities, config.max_scenarios),
    ]
    grid = two_by_two(net_reach_arm(), config.max_scenarios)
    return CU35Result(arms=arms, grid=grid)


def _exact(points: list[FidelityPoint]) -> FidelityPoint:
    return max(points, key=lambda p: p.fidelity)


def _blind(points: list[FidelityPoint]) -> FidelityPoint:
    return min(points, key=lambda p: p.fidelity)


def _non_increasing(values: list[float]) -> bool:
    from itertools import pairwise

    return all(a >= b - 1e-9 for a, b in pairwise(values))


def _flat(values: list[float], tol: float = 1e-9) -> bool:
    return max(values) - min(values) <= tol


def arm_verdict(arm: ArmFidelity) -> dict[str, object]:
    on = arm.on_surface
    off = arm.off_surface
    interior = [p for p in on if p.fidelity < 1.0 - 1e-9]
    return {
        "world": arm.world_name,
        "target_covers": arm.target_covers,
        # on-surface fidelity is load-bearing: exact -> safe, blind -> leaks (both axes)
        "exact_verifier_safe": _exact(on).adversarial_breach <= 1e-9,
        "surface_blind_verifier_leaks": _blind(on).adversarial_breach >= 0.5,
        # vs nature: a sloped faithful-horizon (random breach falls monotonically with phi)
        "random_axis_is_a_horizon": _non_increasing([p.random_breach for p in on])
        and _blind(on).random_breach >= 0.5 and _exact(on).random_breach <= 1e-9,
        # vs the adversary: a cliff -- every partial fidelity leaks, only phi=1 is safe (CU11/CU33)
        "adversarial_is_a_cliff": all(p.adversarial_breach >= 0.5 for p in interior)
        and _exact(on).adversarial_breach <= 1e-9,
        # off-surface fidelity is irrelevant to safety (flat at 0) but costs utility (false blocks)
        "offsurface_breach_flat_safe": _flat([p.adversarial_breach for p in off])
        and max(p.adversarial_breach for p in off) <= 1e-9,
        "offsurface_drift_costs_utility":
            _blind(off).mean_false_blocks > _exact(off).mean_false_blocks + 1e-9,
        "blind_adversarial_breach": _blind(on).adversarial_breach,
        "exact_adversarial_breach": _exact(on).adversarial_breach,
        "blind_random_breach": _blind(on).random_breach,
        "offsurface_max_breach": max(p.adversarial_breach for p in off),
        "offsurface_blind_false_blocks": _blind(off).mean_false_blocks,
    }


def cu35_verdict(result: CU35Result) -> dict[str, object]:
    """H128: the verifier must be faithful on the danger surface (the dual of CU21 target coverage);
    on-surface fidelity is load-bearing, off-surface fidelity is irrelevant to safety, and both
    coverage conditions are independently necessary (the 2x2: only covers AND faithful is safe).
    """
    arms = [arm_verdict(a) for a in result.arms]
    g = result.grid
    return {
        "n_worlds": len(arms),
        "exact_verifier_safe_everywhere": all(bool(a["exact_verifier_safe"]) for a in arms),
        "surface_blind_leaks_everywhere": all(
            bool(a["surface_blind_verifier_leaks"]) for a in arms
        ),
        "random_axis_is_a_horizon_everywhere": all(
            bool(a["random_axis_is_a_horizon"]) for a in arms
        ),
        "adversarial_is_a_cliff_everywhere": all(
            bool(a["adversarial_is_a_cliff"]) for a in arms
        ),
        "offsurface_fidelity_irrelevant_to_safety": all(
            bool(a["offsurface_breach_flat_safe"]) for a in arms
        ),
        "offsurface_drift_costs_utility_somewhere": any(
            bool(a["offsurface_drift_costs_utility"]) for a in arms
        ),
        # the 2x2: both conditions independently necessary, only the (covers, exact) corner is safe
        "grid_covers_exact_safe": g.covers_exact <= 1e-9,
        "grid_covers_blind_leaks": g.covers_blind >= 0.5,
        "grid_leak_exact_leaks": g.leak_exact >= 0.5,
        "grid_leak_blind_leaks": g.leak_blind >= 0.5,
        "both_conditions_necessary": (
            g.covers_exact <= 1e-9
            and g.covers_blind >= 0.5
            and g.leak_exact >= 0.5
            and g.leak_blind >= 0.5
        ),
        "arms": arms,
        "grid": {
            "world": g.world_name,
            "covers_exact": g.covers_exact, "covers_blind": g.covers_blind,
            "leak_exact": g.leak_exact, "leak_blind": g.leak_blind,
        },
    }


CSV_HEADER = "world,sweep,fidelity,adversarial_breach,random_breach,mean_false_blocks,n_scenarios"


def write_csv(result: CU35Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for arm in result.arms:
        for sweep, points in (("on_surface", arm.on_surface), ("off_surface", arm.off_surface)):
            for p in points:
                rows.append(
                    f"{arm.world_name},{sweep},{p.fidelity:.3f},{p.adversarial_breach:.6f},"
                    f"{p.random_breach:.6f},{p.mean_false_blocks:.6f},{arm.n_scenarios}"
                )
    g = result.grid
    for label, val in (("covers_exact", g.covers_exact), ("covers_blind", g.covers_blind),
                       ("leak_exact", g.leak_exact), ("leak_blind", g.leak_blind)):
        rows.append(f"{g.world_name},grid_{label},,{val:.6f},,,")
    out.write_text("\n".join(rows) + "\n")
    return str(out)


def main() -> None:  # pragma: no cover - exercised via the experiment CLI
    result = run_cu35()
    v = cu35_verdict(result)
    print("CU35 / H128 -- the verifier-fidelity condition (the dual coverage law):")
    for arm in result.arms:
        a = arm_verdict(arm)
        print(f"\n  {arm.world_name} (target covers={arm.target_covers}, n={arm.n_scenarios}):")
        print(f"    {'phi (on-surface)':>18s}  adv_breach  false_blocks")
        for p in arm.on_surface:
            print(f"    {p.fidelity:18.2f}  {p.adversarial_breach:10.3f}  "
                  f"{p.mean_false_blocks:11.2f}")
        print(f"    {'psi (off-surface)':>18s}  adv_breach  false_blocks")
        for p in arm.off_surface:
            print(f"    {p.fidelity:18.2f}  {p.adversarial_breach:10.3f}  "
                  f"{p.mean_false_blocks:11.2f}")
        print(f"    exact safe={a['exact_verifier_safe']} / blind leaks="
              f"{a['surface_blind_verifier_leaks']}; off-surface flat-safe="
              f"{a['offsurface_breach_flat_safe']} costs utility="
              f"{a['offsurface_drift_costs_utility']}")
    g = result.grid
    print(f"\n  2x2 ({g.world_name}): covers+exact={g.covers_exact:.3f}  "
          f"covers+blind={g.covers_blind:.3f}  leak+exact={g.leak_exact:.3f}  "
          f"leak+blind={g.leak_blind:.3f}")
    print(f"    both conditions necessary = {v['both_conditions_necessary']}")


if __name__ == "__main__":  # pragma: no cover
    main()
