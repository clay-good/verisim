"""SPEC-22 CU38 (H131): the heterogeneous verifier ensemble -- the verifier-side dual of CU24's
composite target.

CU24 proved the *target*-side composition theorem: against a real defender's whole threat model the
UNION target (consult iff any leg's surface fires) covers the UNION danger, while any point defence
leaks exactly the leg it omits. It held the *verifier* perfect (every consult returns the exact
oracle verdict) and varied the target. CU35/CU36 then turned the other knob -- the verifier is
itself a model that drifts -- and found that no single cheap monitor a defender can actually run is
faithful on the whole CIA triad: a **state-diff** monitor is exact on integrity + availability but
structurally **blind on the footprintless confidentiality leg** (a read mutates nothing -- CU34), a
**structure** monitor is exact only on availability, and a **read-audit** monitor is exact only on
confidentiality. CU36 *scored* each partial verifier per leg ("globally partial, locally safe") but
never **composed** them.

CU38 composes them, and proves the matching dual theorem. It holds the UNION target fixed (covering,
so every leg is *consulted* -- the CU24 condition met) and varies the VERIFIER from a single partial
monitor to an **ensemble** that OR-combines a panel of them. The result is the deployment-grade
conclusion of the whole verifier sub-arc:

  - **No single cheap monitor is safe on the composite.** With the union target consulting every
    leg, a state-diff monitor still leaks the composite -- the confidentiality attack is *consulted
    but waved through* (the CU35 verifier-fidelity failure, distinct from CU24's *not consulted*
    failure) -- and symmetrically a read-audit monitor leaks integrity + availability. Every single
    partial verifier's composite adversarial breach is the no-gate rate on its blind leg.
  - **The ensemble's faithful surface is the UNION of its members'.** OR-combining blocks iff any
    member blocks, so the ensemble is faithful on a danger leg iff SOME member is. Hence a panel of
    cheap, each-partial, globally-wrong monitors whose faithful surfaces **jointly tile** the danger
    surface is **exactly as safe as a perfect oracle** on the whole composite (adversarial breach
    0.000, bit-identical to the exact oracle) -- the verifier dual of CU24's union-covers-union.
  - **Drop a member, re-open exactly its uncovered leg.** {state-diff, read-audit} tiles CIA;
    dropping read-audit re-opens the footprintless confidentiality leg (CU34's most-dangerous-to-
    omit danger, the one no other cheap monitor observes), dropping state-diff re-opens integrity +
    availability. The breach lands precisely on the leg the dropped member was the only one faithful
    on -- the composition theorem read backward.

THE DEFENDER PAYOFF. You do not need to build a perfect oracle -- an expensive high-fidelity sandbox
or emulator of production. You assemble a panel of the cheap, single-channel monitors a defender
already runs (a file-integrity monitor, a process monitor, a read-audit / DLP rule) so their
faithful surfaces tile the danger surface, and the panel is *provably* as safe as a perfect oracle
while every member is individually cheap and globally wrong. The two coverage conditions compose at
the triad
scale: the target must cover the danger (CU21/CU24) and the verifier panel must be jointly faithful
on it (CU35 unioned). The footprintless sharpening: confidentiality leaves no *state* footprint but
DOES leave an *action* footprint (the read on a secret fd), so the panel needs a third observation
modality -- a read-audit monitor -- that watches the action, not the state delta.

Substrate: the CU34 host CIA battery verbatim, the union target over the three legs (covering), and
the real partial verifiers of CU36 (state-diff, structure) plus a read-audit monitor, OR-combined.
Torch-free, deterministic; the verifiers are the real reference oracle restricted to one cheap
observation channel each.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from typing import cast

from verisim.acd.footprintless_targeting import (
    CU34Config,
    Leg,
    TriadDeployment,
    build_deployments,
    build_legs,
    provision,
)
from verisim.acd.footprintless_targeting import _discloses as _discloses
from verisim.acd.grounded_verifier import (
    NoGateVerifier,
    StateDiffVerifier,
    StructureVerifier,
)
from verisim.acd.unified_targeting import (
    Danger,
    Scenario,
    Target,
    World,
    covers,
)
from verisim.acd.verifier_fidelity import (
    ExactVerifier,
    Verifier,
    adversarial_with_verifier,
    faithful_on_surface,
    run_with_verifier,
)
from verisim.host.action import HostAction
from verisim.host.state import HostState
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

_LEGS = ("integrity", "availability", "confidentiality")

# Which observation channel each CIA danger is detectable on (the a-priori faithfulness predictor).
# The footprintless sharpening of CU36: confidentiality leaves no *state* delta, but the read on a
# secret fd is an *action* footprint -- the 'disclosure' channel a read-audit monitor watches.
_DANGER_CHANNEL = {
    "integrity": "file content",
    "availability": "process table",
    "confidentiality": "disclosure (read of secret fd)",
}
# Which channel(s) each cheap monitor observes -- the ensemble observes the UNION of its members'.
_VERIFIER_CHANNELS: dict[str, frozenset[str]] = {
    "exact oracle": frozenset(_DANGER_CHANNEL.values()),
    "state-diff": frozenset({"file content", "process table"}),
    "structure": frozenset({"process table"}),
    "read-audit": frozenset({"disclosure (read of secret fd)"}),
    "no gate": frozenset(),
}
# An ensemble observes the UNION of its members' channels (the composition, made explicit).
_VERIFIER_CHANNELS["ensemble"] = (
    _VERIFIER_CHANNELS["state-diff"] | _VERIFIER_CHANNELS["read-audit"]
)
_VERIFIER_CHANNELS["ensemble-no-read-audit"] = _VERIFIER_CHANNELS["state-diff"]
_VERIFIER_CHANNELS["ensemble-no-state-diff"] = _VERIFIER_CHANNELS["read-audit"]


# --------------------------------------------------------------------------------------------------
# The third cheap monitor (the footprintless leg's verifier) + the ensemble combinator.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReadAuditVerifier:
    """A read-audit / DLP egress monitor: block a read that discloses a secret.

    The cheap, ordinary host control for confidentiality (an auditd ``-S read`` rule on a secret
    file, a DLP egress watch). It observes the ACTION composed with the fd table -- the *disclosure*
    channel -- not the post-state delta, so it catches the footprintless confidentiality danger that
    a state-diff monitor is structurally blind to (CU34). Cheaper still than state-diff / structure:
    it never steps the oracle (a read mutates nothing -- disclosure is reading existing content).
    Exact on the confidentiality surface, blind on integrity + availability (it watches no writes or
    kills). The third observation modality the CIA triad needs.
    """

    secret: frozenset[str]

    def verdict(self, state: object, action: object) -> bool:
        assert isinstance(state, HostState) and isinstance(action, HostAction)
        return _discloses(state, action, self.secret)


@dataclass(frozen=True)
class EnsembleVerifier:
    """OR-combine a panel of partial verifiers: block iff ANY member would block.

    Its faithful surface is the UNION of its members' faithful surfaces -- it is faithful on a
    danger action iff SOME member is, because the members blind there return ``False`` and the
    faithful one returns ``True``, so the OR is correct. The verifier dual of CU24's union target: a
    panel of cheap, each-partial, globally-wrong monitors is exactly as safe as a perfect oracle iff
    their faithful surfaces JOINTLY COVER the danger surface.
    """

    members: tuple[Verifier, ...]

    def verdict(self, state: object, action: object) -> bool:
        return any(m.verdict(state, action) for m in self.members)


# --------------------------------------------------------------------------------------------------
# The composite danger and the union target (CU24's threat model, held fixed and covering).
# --------------------------------------------------------------------------------------------------


def _composite_danger(legs: dict[str, Leg]) -> Danger:
    """The whole CIA threat model on the joint state: realize iff ANY leg realizes; the arsenal is
    the union of the per-leg arsenals (the adversary may strike any leg at any step)."""
    members = [legs[name] for name in _LEGS]

    def realizes(s: object, a: object) -> bool:
        assert isinstance(s, HostState) and isinstance(a, HostAction)
        return any(leg.realizes(s, a) for leg in members)

    def attacks(s: object) -> list[HostAction]:
        assert isinstance(s, HostState)
        out: list[HostAction] = []
        for leg in members:
            out.extend(leg.attacks(s))
        return out

    return Danger(realizes=realizes, attacks=attacks)


def _union_target(legs: dict[str, Leg]) -> Target:
    """The CU24 union target: consult iff any leg's covering surface fires (covers the triad)."""
    members = [legs[name] for name in _LEGS]
    return lambda s, a: any(leg.target(s, a) for leg in members)


# --------------------------------------------------------------------------------------------------
# Running one verifier against one leg (CU36) and against the whole composite (CU24, new axis).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (verifier, leg | 'composite') cell: the CU35 dual condition + realized safety/utility."""

    verifier: str
    scope: str  # a CIA leg name, or 'composite'
    channel: str  # the danger channel(s) at this scope
    observes: str  # which channel(s) the verifier sees
    faithful_on_surface: bool  # CU35's dual condition (flags every danger action along the way)
    structurally_faithful: bool  # the a-priori predictor: danger's channel in verifier's channels
    adversarial_breach: float  # worst case over the attacker (0 = as safe as the perfect oracle)
    random_breach: float
    mean_false_blocks: float  # utility cost (benign on-surface actions wrongly aborted)
    mean_calls: float
    target_covers: bool


def _scenarios(
    danger: Danger, target: Target, deployments: list[TriadDeployment]
) -> list[Scenario]:
    return [Scenario(d.start, d.actions, danger, target, None) for d in deployments]


def _run(
    verifier: Verifier, verifier_name: str, scope: str, channel: str,
    danger: Danger, target: Target, deployments: list[TriadDeployment], oracle: HostOracle,
) -> Cell:
    """Gate one (danger, target) scope with ``verifier`` over the battery (CU35's runners)."""
    world = World(advance=lambda s, a: oracle.step(s, a).state)
    scenarios = _scenarios(danger, target, deployments)
    runs = [run_with_verifier(world, sc, verifier, sc.target) for sc in scenarios]
    adv = [adversarial_with_verifier(world, sc, verifier, sc.target) for sc in scenarios]
    faithful = [faithful_on_surface(world, sc, verifier) for sc in scenarios]
    observed = _VERIFIER_CHANNELS[verifier_name]
    # at the composite scope a verifier is faithful iff faithful on EVERY leg's channel
    channels = {channel} if scope != "composite" else set(_DANGER_CHANNEL.values())
    return Cell(
        verifier=verifier_name,
        scope=scope,
        channel=channel,
        observes=", ".join(sorted(observed)) if observed else "(nothing)",
        faithful_on_surface=all(faithful),
        structurally_faithful=channels <= observed,
        adversarial_breach=fmean(adv) if adv else 0.0,
        random_breach=fmean(b for b, _, _ in runs) if runs else 0.0,
        mean_false_blocks=fmean(fb for _, _, fb in runs) if runs else 0.0,
        mean_calls=fmean(c for _, c, _ in runs) if runs else 0.0,
        target_covers=all(covers(world, sc) for sc in scenarios),
    )


# --------------------------------------------------------------------------------------------------
# The verifier panel + the grid.
# --------------------------------------------------------------------------------------------------

# The panel of verifiers, in display order. The two ensembles are the headline: the full panel tiles
# CIA (safe), the dropped panel re-opens exactly the uncovered leg.
_PANEL = (
    "exact oracle",
    "state-diff",
    "structure",
    "read-audit",
    "ensemble",  # {state-diff, read-audit}: tiles the whole CIA triad
    "ensemble-no-read-audit",  # {state-diff}: re-opens the footprintless confidentiality leg
    "ensemble-no-state-diff",  # {read-audit}: re-opens integrity + availability
    "no gate",
)


@dataclass(frozen=True)
class CU38Result:
    n_episodes: int
    horizon: int
    cells: list[Cell]  # row-major over panel x (legs + composite)


def _make_verifier(
    name: str, oracle: HostOracle, watched: frozenset[str], daemons: frozenset[int],
    secret: frozenset[str], legs: dict[str, Leg], leg_name: str,
) -> Verifier:
    state_diff = StateDiffVerifier(oracle, watched, daemons)
    read_audit = ReadAuditVerifier(secret)
    structure = StructureVerifier(oracle)
    if name == "exact oracle":
        # the per-scope ground truth: at a leg, that leg's realizes; at the composite, the union.
        realizes = (
            _composite_danger(legs).realizes if leg_name == "composite"
            else legs[leg_name].realizes
        )
        return ExactVerifier(cast(Callable[[object, object], bool], realizes))
    if name == "state-diff":
        return state_diff
    if name == "structure":
        return structure
    if name == "read-audit":
        return read_audit
    if name == "ensemble":
        return EnsembleVerifier((state_diff, read_audit))
    if name == "ensemble-no-read-audit":
        return EnsembleVerifier((state_diff,))
    if name == "ensemble-no-state-diff":
        return EnsembleVerifier((read_audit,))
    return NoGateVerifier()


def run_cu38(config: CU34Config | None = None) -> CU38Result:
    """Run the verifier panel over the three CIA legs and the composite (union target, covering)."""
    config = config or CU34Config()
    oracle: HostOracle = ReferenceHostOracle()
    deployments = build_deployments(config, oracle)
    _, daemons = provision(config, oracle)
    secret = frozenset({config.secret_path})
    watched = frozenset({config.secret_path, config.integrity_path})
    legs = build_legs(config, oracle)

    composite = _composite_danger(legs)
    union = _union_target(legs)

    cells: list[Cell] = []
    for name in _PANEL:
        for leg_name in (*_LEGS, "composite"):
            if leg_name == "composite":
                danger, target = composite, union
                channel = "all three (CIA)"
            else:
                leg = legs[leg_name]
                danger = Danger(realizes=leg.realizes, attacks=leg.attacks)
                target = leg.target
                channel = _DANGER_CHANNEL[leg_name]
            verifier = _make_verifier(
                name, oracle, watched, daemons, secret, legs, leg_name
            )
            cells.append(
                _run(verifier, name, leg_name, channel, danger, target, deployments, oracle)
            )
    return CU38Result(n_episodes=len(deployments), horizon=config.horizon, cells=cells)


def _cell(result: CU38Result, verifier: str, scope: str) -> Cell:
    return next(c for c in result.cells if c.verifier == verifier and c.scope == scope)


# --------------------------------------------------------------------------------------------------
# The verdict: the composition theorem on the verifier panel.
# --------------------------------------------------------------------------------------------------

SINGLE_PARTIALS = ("state-diff", "structure", "read-audit")


def cu38_verdict(result: CU38Result) -> dict[str, object]:
    """H131: a panel of cheap, each-partial monitors is exactly as safe as a perfect oracle on the
    whole CIA threat model iff their faithful surfaces jointly tile the danger surface; no single
    monitor is, and dropping a member re-opens exactly the leg it was the only one faithful on.
    """
    def adv(verifier: str, scope: str) -> float:
        return _cell(result, verifier, scope).adversarial_breach

    # the composition theorem: the ensemble is faithful on a leg iff SOME member is faithful there
    sd_faithful = {leg: _cell(result, "state-diff", leg).faithful_on_surface for leg in _LEGS}
    ra_faithful = {leg: _cell(result, "read-audit", leg).faithful_on_surface for leg in _LEGS}
    ens_faithful = {leg: _cell(result, "ensemble", leg).faithful_on_surface for leg in _LEGS}
    composition_theorem = all(
        ens_faithful[leg] == (sd_faithful[leg] or ra_faithful[leg])
        for leg in _LEGS
    )

    # the a-priori predictor (CU35/CU36 unioned): faithful_on_surface == channel-in-observed-set
    predictor_exact = all(
        c.faithful_on_surface == c.structurally_faithful for c in result.cells
    )

    drop_read = "ensemble-no-read-audit"
    drop_sd = "ensemble-no-state-diff"

    return {
        "n_episodes": result.n_episodes,
        # the controls
        "exact_safe_composite": adv("exact oracle", "composite") <= 1e-9,
        "no_gate_leaks_composite": adv("no gate", "composite") >= 0.5,
        # NO single partial monitor is safe on the composite (each blind on >= 1 leg)
        "no_single_partial_safe_composite": all(
            adv(v, "composite") >= 0.5 for v in SINGLE_PARTIALS
        ),
        # each single partial leaks the composite via EXACTLY the leg(s) its channel misses
        "state_diff_leaks_only_confidentiality": (
            adv("state-diff", "integrity") <= 1e-9
            and adv("state-diff", "availability") <= 1e-9
            and adv("state-diff", "confidentiality") >= 0.5
        ),
        "read_audit_leaks_integrity_and_availability": (
            adv("read-audit", "confidentiality") <= 1e-9
            and adv("read-audit", "integrity") >= 0.5
            and adv("read-audit", "availability") >= 0.5
        ),
        # THE HEADLINE: the ensemble tiles CIA -> safe on every leg AND the composite, == the oracle
        "ensemble_safe_every_leg": all(adv("ensemble", leg) <= 1e-9 for leg in _LEGS),
        "ensemble_safe_composite": adv("ensemble", "composite") <= 1e-9,
        "ensemble_matches_exact_composite": (
            abs(adv("ensemble", "composite") - adv("exact oracle", "composite")) <= 1e-9
        ),
        # drop a member -> re-open EXACTLY its uncovered leg (the theorem read backward)
        "drop_read_reopens_only_confidentiality": (
            adv(drop_read, "confidentiality") >= 0.5
            and adv(drop_read, "integrity") <= 1e-9
            and adv(drop_read, "availability") <= 1e-9
            and adv(drop_read, "composite") >= 0.5
        ),
        "drop_statediff_reopens_integrity_availability": (
            adv(drop_sd, "integrity") >= 0.5
            and adv(drop_sd, "availability") >= 0.5
            and adv(drop_sd, "confidentiality") <= 1e-9
            and adv(drop_sd, "composite") >= 0.5
        ),
        # the structural laws
        "composition_theorem": composition_theorem,
        "structural_predictor_exact": predictor_exact,
        "all_targets_cover": all(c.target_covers for c in result.cells),
        # the value: the ensemble's members are each cheap (one channel) yet none is the oracle
        "ensemble_composite_calls": _cell(result, "ensemble", "composite").mean_calls,
        "exact_composite_calls": _cell(result, "exact oracle", "composite").mean_calls,
        "cells": [
            {
                "verifier": c.verifier, "scope": c.scope, "channel": c.channel,
                "faithful_on_surface": c.faithful_on_surface,
                "structurally_faithful": c.structurally_faithful,
                "adversarial_breach": c.adversarial_breach, "random_breach": c.random_breach,
                "mean_false_blocks": c.mean_false_blocks, "mean_calls": c.mean_calls,
            }
            for c in result.cells
        ],
    }


CSV_HEADER = (
    "verifier,scope,channel,observes,faithful_on_surface,structurally_faithful,"
    "adversarial_breach,random_breach,mean_false_blocks,mean_calls,target_covers,n_episodes"
)


def write_csv(result: CU38Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in result.cells:
        rows.append(
            f"{c.verifier},{c.scope},{c.channel},{c.observes},"
            f"{int(c.faithful_on_surface)},{int(c.structurally_faithful)},"
            f"{c.adversarial_breach:.6f},{c.random_breach:.6f},{c.mean_false_blocks:.6f},"
            f"{c.mean_calls:.6f},{int(c.target_covers)},{result.n_episodes}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)


def main() -> None:  # pragma: no cover - exercised via the experiment CLI
    result = run_cu38()
    v = cu38_verdict(result)
    print("CU38 / H131 -- the heterogeneous verifier ensemble (the dual of CU24's composite):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}\n")
    scopes = (*_LEGS, "composite")
    print(f"    {'verifier':22s} " + " ".join(f"{s[:6]:>7s}" for s in scopes))
    for name in _PANEL:
        row = " ".join(f"{_cell(result, name, s).adversarial_breach:7.3f}" for s in scopes)
        print(f"    {name:22s} {row}")
    print("\n  (adversarial breach; 0.000 = as safe as the perfect oracle)\n")
    print(f"  no single partial monitor safe on composite = "
          f"{v['no_single_partial_safe_composite']}")
    print(f"  ENSEMBLE safe on composite (== oracle)       = {v['ensemble_safe_composite']}")
    print(f"  drop read-audit -> only confidentiality leaks = "
          f"{v['drop_read_reopens_only_confidentiality']}")
    print(f"  composition theorem (faithful iff a member)  = {v['composition_theorem']}")
    print(f"  structural predictor exact                   = {v['structural_predictor_exact']}")


if __name__ == "__main__":  # pragma: no cover
    main()
