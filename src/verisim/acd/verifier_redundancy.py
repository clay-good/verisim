"""SPEC-22 CU39 (H132): the redundant verifier -- defense in depth requires failure INDEPENDENCE.

CU38 proved the verifier-side composition theorem: a panel of cheap, each-partial monitors whose
faithful surfaces *tile* the danger surface is exactly as safe as a perfect oracle. But it tiled
*across* the CIA legs with members held **exact** on their channel (a state-diff monitor literally
observes the post-state delta, so it is bit-exact on integrity). CU35's whole premise is the
opposite: a real deployed verifier is a *model* that drifts -- faithful on its surface only with
probability ``phi < 1`` (a sandbox misses a syscall, a DLP rule has a gap, a monitor is mis-tuned).
CU38 never asks what its tiling theorem does when the members themselves are imperfect.

CU39 asks it, and the answer is the oldest principle in security finally given a coverage-theoretic
statement: **defense in depth** -- stack redundant verifiers so a danger one misses another catches.
The result is that depth is real but it is *not free* and *not automatic*:

  - **A single imperfect monitor on a leg is gameable.** With ``phi < 1`` the monitor omits a
    ``(1 - phi)`` fraction of the danger surface, and the worst-case adversary (CU4/CU11) fires
    exactly there: adversarial breach is pinned near the no-gate rate (CU35's cliff -- a partial
    verifier is adversarially as good as no verifier).

  - **Redundancy helps iff the members fail INDEPENDENTLY.** OR-combine ``m`` monitors (block iff
    any blocks, CU38's ensemble): the ensemble omits a danger action iff *every* member omits it.
      * **Homogeneous** redundancy -- ``m`` copies of the same monitor (or correlated monitors that
        share a blind spot) -- omits exactly the shared blind spot, *for any ``m``*. The adversary's
        target never moves; adversarial breach is **flat in ``m``**. Running the same scanner twice
        buys nothing.
      * **Heterogeneous** redundancy -- ``m`` monitors with *independent* blind spots (different
        observation channels / features) -- omits a danger action only where *all ``m``* are
        independently blind, a ``(1 - phi)**m`` fraction. As ``m`` grows the members' faithful
        surfaces TILE the leg and the adversary runs out of fully-omitted actions: adversarial
        breach falls to **0** at a knee ``m* ~ ln A / ln(1 / (1 - phi))`` (``A`` = arsenal size).

  - **Depth costs a compounding false-block tax.** An OR of ``m`` members false-alarms iff *any*
    member does, so off-surface false blocks compound as ``1 - psi**m`` -- you pay for depth in
    utility, not safety (CU37's precision tax, now multiplied by the stack height). Homogeneous
    redundancy pays no extra tax (identical false alarms) precisely because it buys no extra safety.

THE DEFENDER PRINCIPLE. Defense in depth is the CU38 tiling theorem operating *within* one danger
leg at the sub-action granularity, and the heterogeneity that gave CU38 *breadth* (tile across CIA)
is exactly what gives *depth* (tile a single leg with independent-blind-spot members). Stacking
diverse monitors -- different channels, different engines -- multiplies safety against an adversary;
stacking copies of the same monitor does not, because the adversary defeats correlated redundancy
for free. Add depth where a single monitor's fidelity is uncertain, choose the members to fail
independently, and budget the compounding false-block tax.

Substrate: the CU21 net / host arms verbatim (the covering target consults every danger action), the
CU35 imperfect-verifier dial (``phi`` on-surface omission, ``psi`` off-surface false alarm) made
salt-keyed so a panel's blind spots are independent or shared, and the CU38 OR ensemble. Torch-free,
deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

from verisim.acd.unified_targeting import (
    Arm,
    Scenario,
    host_arm,
    net_flow_arm,
)
from verisim.acd.verifier_ensemble import EnsembleVerifier
from verisim.acd.verifier_fidelity import (
    Realizes,
    Verifier,
    _target_of,
    _unit,
    adversarial_with_verifier,
    faithful_on_surface,
    run_with_verifier,
)

# --------------------------------------------------------------------------------------------------
# An imperfect monitor with an independently-placeable blind spot (CU35's dial, salt-keyed).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImperfectMonitor:
    """A real, drifting verifier: faithful on its surface only to degree ``phi``.

    On the danger surface (``realizes``) it flags a danger action only if it falls outside its
    blind spot -- a ``(1 - phi)`` fraction keyed by ``salt`` (CU35's :class:`SurfaceOmitter`,
    made salt-addressable so a panel can be given independent or shared blind spots). Off the
    surface it false-alarms on a ``(1 - psi)`` fraction (CU35's :class:`OffSurfaceDrifter`, also
    salted). ``phi=1, psi=1`` is the exact oracle. The ``salt`` is the *identity* of the monitor:
    two monitors with the same salt share a blind spot (correlated); different salts fail
    independently.
    """

    realizes: Realizes
    phi: float
    psi: float
    salt: str

    def verdict(self, state: object, action: object) -> bool:
        if self.realizes(state, action):
            # faithful iff this action is outside the monitor's (salt-keyed) on-surface blind spot
            return _unit(f"on:{self.salt}:{action!r}") < self.phi
        # off-surface: a false alarm iff this action is inside the (salt-keyed) hallucination set
        return _unit(f"off:{self.salt}:{action!r}") >= self.psi


def homogeneous(realizes: Realizes, phi: float, psi: float, m: int) -> Verifier:
    """``m`` copies of the SAME monitor -- correlated failure, a shared blind spot.

    All members carry salt ``"0"``, so they omit exactly the same danger actions; the OR ensemble is
    identical to a single member however large ``m`` is. The control: redundancy sans independence.
    """
    members = tuple(ImperfectMonitor(realizes, phi, psi, "0") for _ in range(m))
    return EnsembleVerifier(members)


def heterogeneous(realizes: Realizes, phi: float, psi: float, m: int) -> Verifier:
    """``m`` monitors with INDEPENDENT blind spots -- distinct salts ``"0".."m-1"``.

    Each omits an independent ``(1 - phi)`` fraction of the surface, so the OR ensemble omits a
    danger action only where all ``m`` are independently blind (a ``(1 - phi)**m`` fraction). The
    heterogeneous panel a defender builds from diverse channels/engines -- true defense in depth.
    """
    members = tuple(ImperfectMonitor(realizes, phi, psi, str(i)) for i in range(m))
    return EnsembleVerifier(members)


# --------------------------------------------------------------------------------------------------
# Sweeping the stack height m for both redundancy regimes.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DepthPoint:
    """One stack height ``m`` of one redundancy regime: worst-case safety + benign cost."""

    regime: str  # 'homogeneous' | 'heterogeneous'
    phi: float
    m: int
    adversarial_breach: float  # worst case over the attacker (0 = as safe as the perfect oracle)
    random_breach: float
    mean_false_blocks: float  # the compounding utility tax
    faithful_on_surface: float  # fraction of scenarios on which the ensemble flags every danger


def _measure(
    arm: Arm, scenarios: list[Scenario], regime: str, phi: float, psi: float, m: int,
) -> DepthPoint:
    def make(realizes: Realizes) -> Verifier:
        return (homogeneous if regime == "homogeneous" else heterogeneous)(realizes, phi, psi, m)

    built = [(sc, make(sc.danger.realizes), _target_of(sc, use_shortcut=False)) for sc in scenarios]
    runs = [run_with_verifier(arm.world, sc, v, tgt) for sc, v, tgt in built]
    adv = [adversarial_with_verifier(arm.world, sc, v, tgt) for sc, v, tgt in built]
    faithful = [faithful_on_surface(arm.world, sc, v) for sc, v, _ in built]
    return DepthPoint(
        regime=regime,
        phi=phi,
        m=m,
        adversarial_breach=fmean(adv) if adv else 0.0,
        random_breach=fmean(b for b, _, _ in runs) if runs else 0.0,
        mean_false_blocks=fmean(fb for _, _, fb in runs) if runs else 0.0,
        faithful_on_surface=fmean(float(f) for f in faithful) if faithful else 0.0,
    )


@dataclass(frozen=True)
class ArmDepth:
    """One arm swept over stack height for homogeneous + heterogeneous redundancy at each phi."""

    world_name: str
    n_scenarios: int
    psi: float
    # keyed by phi: the m-sweep for each regime
    homogeneous: dict[float, list[DepthPoint]]
    heterogeneous: dict[float, list[DepthPoint]]


def sweep_arm(
    arm: Arm, phis: tuple[float, ...], heights: tuple[int, ...], psi: float, max_scenarios: int,
) -> ArmDepth:
    scenarios = arm.scenarios[:max_scenarios]
    homo: dict[float, list[DepthPoint]] = {}
    hetero: dict[float, list[DepthPoint]] = {}
    for phi in phis:
        homo[phi] = [_measure(arm, scenarios, "homogeneous", phi, psi, m) for m in heights]
        hetero[phi] = [_measure(arm, scenarios, "heterogeneous", phi, psi, m) for m in heights]
    return ArmDepth(
        world_name=arm.world_name,
        n_scenarios=len(scenarios),
        psi=psi,
        homogeneous=homo,
        heterogeneous=hetero,
    )


# --------------------------------------------------------------------------------------------------
# Config + top-level run + verdict.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CU39Config:
    phis: tuple[float, ...] = (0.3, 0.5, 0.7)
    heights: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    psi: float = 0.8  # off-surface fidelity (1 - psi false-alarms per consulted benign action)
    max_scenarios: int = 80
    headline_phi: float = 0.5  # the phi for the homogeneous-vs-heterogeneous contrast panel

    @staticmethod
    def smoke() -> CU39Config:
        return CU39Config(phis=(0.5,), heights=(1, 2, 4), max_scenarios=8, headline_phi=0.5)


@dataclass(frozen=True)
class CU39Result:
    arms: list[ArmDepth]
    headline_phi: float


def run_cu39(config: CU39Config | None = None) -> CU39Result:
    """Sweep the redundant verifier (homogeneous vs heterogeneous, over m and phi) on net + host."""
    config = config or CU39Config()
    arms = [
        sweep_arm(net_flow_arm(), config.phis, config.heights, config.psi, config.max_scenarios),
        sweep_arm(host_arm(), config.phis, config.heights, config.psi, config.max_scenarios),
    ]
    return CU39Result(arms=arms, headline_phi=config.headline_phi)


def _nearest_phi(keys: list[float], target: float) -> float:
    return min(keys, key=lambda k: abs(k - target))


def _non_increasing(values: list[float]) -> bool:
    from itertools import pairwise

    return all(a >= b - 1e-9 for a, b in pairwise(values))


def _flat(values: list[float], tol: float = 1e-9) -> bool:
    return max(values) - min(values) <= tol


def arm_verdict(arm: ArmDepth, headline_phi: float) -> dict[str, object]:
    phi = _nearest_phi(list(arm.heterogeneous.keys()), headline_phi)
    homo = arm.homogeneous[phi]
    hetero = arm.heterogeneous[phi]
    homo_adv = [p.adversarial_breach for p in homo]
    hetero_adv = [p.adversarial_breach for p in hetero]
    hetero_fb = [p.mean_false_blocks for p in hetero]
    # the defense-in-depth knee: smallest m at which heterogeneous reaches the oracle (breach 0)
    knee = next((p.m for p in hetero if p.adversarial_breach <= 1e-9), None)
    return {
        "world": arm.world_name,
        "headline_phi": phi,
        # a single imperfect monitor (m=1) is gameable (the cliff -- CU35)
        "single_monitor_gameable": hetero[0].adversarial_breach >= 0.5,
        # homogeneous redundancy is flat in m (correlated blind spot -- buys nothing)
        "homogeneous_flat_in_m": _flat(homo_adv) and min(homo_adv) >= 0.5,
        # heterogeneous redundancy falls monotonically and reaches the oracle (breach 0)
        "heterogeneous_monotone_down": _non_increasing(hetero_adv),
        "heterogeneous_reaches_oracle": min(hetero_adv) <= 1e-9,
        "defense_in_depth_knee": knee,
        # independence is load-bearing: at max m, heterogeneous is strictly safer than homogeneous
        "independence_is_load_bearing": hetero_adv[-1] < homo_adv[-1] - 1e-9,
        # depth costs a compounding false-block tax (rises with m)
        "depth_costs_utility": hetero_fb[-1] > hetero_fb[0] + 1e-9,
        "homogeneous_adv_breach": homo_adv,
        "heterogeneous_adv_breach": hetero_adv,
        "heterogeneous_false_blocks": hetero_fb,
        "heights": [p.m for p in hetero],
    }


def cu39_verdict(result: CU39Result) -> dict[str, object]:
    """H132: defense in depth is real but requires failure independence -- heterogeneous redundancy
    (independent blind spots) drives adversarial breach to the oracle's 0 at a knee, homogeneous
    redundancy (a shared blind spot) is flat in stack height; depth costs a compounding false-block
    tax. The CU38 tiling theorem operating within one danger leg at the sub-action granularity.
    """
    arms = [arm_verdict(a, result.headline_phi) for a in result.arms]
    return {
        "n_worlds": len(arms),
        "single_monitor_gameable_everywhere": all(
            bool(a["single_monitor_gameable"]) for a in arms
        ),
        "homogeneous_flat_everywhere": all(bool(a["homogeneous_flat_in_m"]) for a in arms),
        "heterogeneous_reaches_oracle_everywhere": all(
            bool(a["heterogeneous_reaches_oracle"]) for a in arms
        ),
        "heterogeneous_monotone_everywhere": all(
            bool(a["heterogeneous_monotone_down"]) for a in arms
        ),
        "independence_load_bearing_everywhere": all(
            bool(a["independence_is_load_bearing"]) for a in arms
        ),
        "depth_costs_utility_somewhere": any(bool(a["depth_costs_utility"]) for a in arms),
        "arms": arms,
    }


CSV_HEADER = (
    "world,regime,phi,m,adversarial_breach,random_breach,mean_false_blocks,"
    "faithful_on_surface,n_scenarios"
)


def write_csv(result: CU39Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for arm in result.arms:
        regimes = (("homogeneous", arm.homogeneous), ("heterogeneous", arm.heterogeneous))
        for regime, table in regimes:
            for phi, points in table.items():
                for p in points:
                    rows.append(
                        f"{arm.world_name},{regime},{phi:.3f},{p.m},{p.adversarial_breach:.6f},"
                        f"{p.random_breach:.6f},{p.mean_false_blocks:.6f},"
                        f"{p.faithful_on_surface:.6f},{arm.n_scenarios}"
                    )
    out.write_text("\n".join(rows) + "\n")
    return str(out)


def main() -> None:  # pragma: no cover - exercised via the experiment CLI
    result = run_cu39()
    v = cu39_verdict(result)
    print("CU39 / H132 -- the redundant verifier (defense in depth requires failure independence):")
    for arm in result.arms:
        a = arm_verdict(arm, result.headline_phi)
        print(f"\n  {arm.world_name} (phi={a['headline_phi']}, n={arm.n_scenarios}, "
              f"psi={arm.psi}):")
        print(f"    {'m':>3s}  {'homo adv':>9s}  {'hetero adv':>10s}  {'hetero false-blocks':>19s}")
        homo = arm.homogeneous[a["headline_phi"]]  # type: ignore[index]
        hetero = arm.heterogeneous[a["headline_phi"]]  # type: ignore[index]
        for ph, pe in zip(homo, hetero, strict=True):
            print(f"    {pe.m:3d}  {ph.adversarial_breach:9.3f}  {pe.adversarial_breach:10.3f}  "
                  f"{pe.mean_false_blocks:19.3f}")
        print(f"    single monitor gameable={a['single_monitor_gameable']}  "
              f"homo flat={a['homogeneous_flat_in_m']}  "
              f"hetero->oracle at m={a['defense_in_depth_knee']}")
    print(f"\n  independence load-bearing everywhere = "
          f"{v['independence_load_bearing_everywhere']}")
    print(f"  depth costs a compounding false-block tax = {v['depth_costs_utility_somewhere']}")


if __name__ == "__main__":  # pragma: no cover
    main()
