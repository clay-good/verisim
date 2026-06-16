"""SPEC-22 CU36 (H129): the grounded verifier -- CU35's verifier-fidelity law, against real models.

CU35 turned the arc's last free variable -- the *verifier* -- and proved the dual coverage law: a
deployed verifier need not be a perfect oracle, it need only be **faithful on the danger surface**
(for every danger action, its consulted verdict equals the oracle's). The non-obvious payoff is
LOCALIZATION: under a covering target the verifier is consulted only on the danger's sparse surface,
so a verifier that is *globally wrong but exact on the danger grammar* is exactly as safe as a
perfect oracle, while a verifier *faithful everywhere except the surface* (an on-surface omitter) is
as blind as no gate. But CU35 demonstrated this with an **abstract** verifier -- a hash-coin
``SurfaceOmitter(phi)`` whose fidelity is a dialed parameter. The skeptic's question stands exactly
as it did before CU28 grounded the *targeting* arc against a real ``/bin/sh``: *is the
verifier-fidelity law a property of real verifiers, or only of a synthetic fidelity dial?*

CU36 grounds it. It exhibits two **real, deployable, structurally-defined** partial verifiers --
each a concrete observation model a defender actually runs, not a coin -- and shows that CU35's
localization law governs them exactly, with the twist that their fidelity profile is **read off the
danger grammar a priori** rather than swept:

  - **the state-diff verifier** (:class:`StateDiffVerifier`) -- the cheap deployed baseline: a
    file-integrity + process monitor (CU34's after-the-fact detector, now reframed as a
    before-commit verifier). It observes the oracle's post-state *delta* and is blind to a read's
    output channel. By the host grammar it is EXACT on the integrity surface (a write flips a
    watched file's content) and the availability surface (a kill ends a daemon), and an ON-SURFACE
    OMITTER on the confidentiality surface (a read mutates nothing -- CU34's footprintless leg).
  - **the structure verifier** (:class:`StructureVerifier`) -- the structure/content boundary
    (SPEC-20) made into a verifier: it observes the process + fd tables (the structure the host
    model learns faithfully) and is blind to file content. By the grammar it is EXACT on
    availability (a kill changes the process table) and an on-surface omitter on BOTH integrity (a
    write changes only content) and confidentiality (a read changes nothing).

So the two real verifiers have *different, grammar-predictable* coverage footprints, and CU35's law
predicts every cell: where the verifier observes the state component the danger mutates, it is
faithful on the surface and **exactly as safe as the perfect oracle**; where it does not, it is an
on-surface omitter and **exactly as blind as no gate** -- even though both verifiers are globally
strict approximations of the oracle. The headline (H129): you can read a cheap verifier's safety off
the danger grammar without ever measuring its fidelity -- ``faithful_on_surface`` (CU35's dual
condition) is decided by whether the danger mutates a channel the verifier observes, and that is the
checkable condition CU35 promised, now grounded against verifiers you can build.

Substrate: the CU34 host CIA battery verbatim (one provisioned state presenting integrity /
availability / confidentiality dangers), each leg's own covering target, run through CU35's
verifier-gated runners. Torch-free, deterministic; the verifiers are the real reference oracle
restricted to one observation channel.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from typing import cast

from verisim.acd.footprintless_targeting import (
    CU34Config,
    TriadDeployment,
    build_deployments,
    build_legs,
    provision,
)
from verisim.acd.footprintless_targeting import state_diff_changed as _state_diff_changed
from verisim.acd.unified_targeting import (
    Danger,
    Scenario,
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

# --------------------------------------------------------------------------------------------------
# Two REAL partial verifiers (concrete observation models, not a hash-coin fidelity dial).
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StateDiffVerifier:
    """The cheap deployed baseline: watch protected resources for a post-state change.

    A file-integrity + process monitor -- exactly CU34's after-the-fact detector, here consulted as
    a BEFORE-COMMIT verifier. It executes the action through the oracle and flags it iff a watched
    file's content-presence flipped (a corruption) or a daemon died (an outage). It observes only
    the *state delta*; a read's disclosure leaves through stdout, not the state, so it is
    structurally blind to it. Exact on the integrity + availability danger surfaces, an on-surface
    omitter on the confidentiality surface (CU34's footprintless leg) -- fixed by the grammar.
    """

    oracle: HostOracle
    watched: frozenset[str]
    daemons: frozenset[int]

    def verdict(self, state: object, action: object) -> bool:
        assert isinstance(state, HostState) and isinstance(action, HostAction)
        after = self.oracle.step(state, action).state
        return _state_diff_changed(state, after, self.watched, self.daemons)


@dataclass(frozen=True)
class StructureVerifier:
    """The structure/content boundary (SPEC-20) as a verifier: observe structure, ignore content.

    It executes the action and flags it iff the process table or the fd table changed -- the host
    *structure*, the dimension the trained host model learns faithfully (SPEC-20). It is blind to
    file content (a write touches only the filesystem, never the proc/fd tables) and to a read
    (which mutates nothing). So it is exact on the availability surface (a kill ends a daemon = a
    proc-table change) and an on-surface omitter on BOTH the integrity and confidentiality surfaces.
    """

    oracle: HostOracle

    def verdict(self, state: object, action: object) -> bool:
        assert isinstance(state, HostState) and isinstance(action, HostAction)
        after = self.oracle.step(state, action).state
        return after.procs != state.procs or after.fds != state.fds


@dataclass(frozen=True)
class NoGateVerifier:
    """The unverified baseline: blocks nothing (the worst-case omitter as a verifier)."""

    def verdict(self, state: object, action: object) -> bool:
        return False


# Which host state channel each danger mutates, and which each verifier observes -- the grammar
# facts that decide ``faithful_on_surface`` a priori (the headline's structural predictor).
_DANGER_MUTATES = {
    "integrity": "file content",
    "availability": "process table",
    "confidentiality": "nothing (footprintless)",
}
_VERIFIER_OBSERVES = {
    "exact oracle": frozenset({"file content", "process table", "nothing (footprintless)"}),
    "state-diff": frozenset({"file content", "process table"}),
    "structure": frozenset({"process table"}),
    "no gate": frozenset(),
}


# --------------------------------------------------------------------------------------------------
# Running one real verifier against one CIA leg over the shared battery.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifierLegResult:
    """One (real verifier, CIA leg) cell: the CU35 dual condition + the realized safety/utility."""

    verifier: str
    leg: str
    danger_mutates: str  # the grammar fact: which channel the danger mutates
    verifier_observes: str  # which channel(s) the verifier sees
    faithful_on_surface: bool  # CU35's dual condition (every scenario flags every attack)
    faithful_fraction: float  # fraction of scenarios that are fully faithful on surface
    structurally_faithful: bool  # the a-priori predictor: danger's channel in verifier's channels
    adversarial_breach: float  # worst case over the attacker (0 = as safe as the oracle)
    random_breach: float
    mean_false_blocks: float  # the utility cost (benign on-target actions wrongly aborted)
    target_covers: bool


def _scenarios(
    leg_name: str, deployments: list[TriadDeployment], config: CU34Config, oracle: HostOracle
) -> list[Scenario]:
    """The leg's covering-target scenarios over the shared provisioned battery (CU34's _leg_arm)."""
    leg = build_legs(config, oracle)[leg_name]
    danger = Danger(realizes=leg.realizes, attacks=leg.attacks)
    return [Scenario(d.start, d.actions, danger, leg.target, None) for d in deployments]


def run_verifier_leg(
    verifier: Verifier, verifier_name: str, leg_name: str,
    deployments: list[TriadDeployment], config: CU34Config, oracle: HostOracle,
) -> VerifierLegResult:
    """Gate one CIA leg's covering target with ``verifier`` over the battery (CU35's runners)."""
    world = World(advance=lambda s, a: oracle.step(s, a).state)
    scenarios = _scenarios(leg_name, deployments, config, oracle)
    runs = [run_with_verifier(world, sc, verifier, sc.target) for sc in scenarios]
    adv = [adversarial_with_verifier(world, sc, verifier, sc.target) for sc in scenarios]
    faithful = [faithful_on_surface(world, sc, verifier) for sc in scenarios]
    mutated = _DANGER_MUTATES[leg_name]
    observed = _VERIFIER_OBSERVES[verifier_name]
    return VerifierLegResult(
        verifier=verifier_name,
        leg=leg_name,
        danger_mutates=mutated,
        verifier_observes=", ".join(sorted(observed)) if observed else "(nothing)",
        faithful_on_surface=all(faithful),
        faithful_fraction=fmean(faithful) if faithful else 0.0,
        structurally_faithful=mutated in observed,
        adversarial_breach=fmean(adv) if adv else 0.0,
        random_breach=fmean(b for b, _, _ in runs) if runs else 0.0,
        mean_false_blocks=fmean(fb for _, _, fb in runs) if runs else 0.0,
        target_covers=all(covers(world, sc) for sc in scenarios),
    )


# --------------------------------------------------------------------------------------------------
# The grid: {real verifiers + controls} x {CIA legs}.
# --------------------------------------------------------------------------------------------------

_LEGS = ("availability", "integrity", "confidentiality")


@dataclass(frozen=True)
class CU36Result:
    n_episodes: int
    horizon: int
    cells: list[VerifierLegResult]  # row-major over verifiers x legs


def run_cu36(config: CU34Config | None = None) -> CU36Result:
    """Run the two real verifiers (+ exact-oracle / no-gate controls) on the three host CIA legs."""
    config = config or CU34Config()
    oracle: HostOracle = ReferenceHostOracle()
    deployments = build_deployments(config, oracle)
    _, daemons = provision(config, oracle)
    watched = frozenset({config.secret_path, config.integrity_path})

    legs = build_legs(config, oracle)

    def verifier_for(name: str, leg_name: str) -> Verifier:
        # the exact oracle is the per-leg ground truth; the real verifiers observe the state
        if name == "exact oracle":
            realizes = cast(Callable[[object, object], bool], legs[leg_name].realizes)
            return ExactVerifier(realizes)
        if name == "state-diff":
            return StateDiffVerifier(oracle, watched, daemons)
        if name == "structure":
            return StructureVerifier(oracle)
        return NoGateVerifier()

    names = ("exact oracle", "state-diff", "structure", "no gate")
    cells = [
        run_verifier_leg(verifier_for(name, leg_name), name, leg_name, deployments, config, oracle)
        for name in names
        for leg_name in _LEGS
    ]
    return CU36Result(n_episodes=len(deployments), horizon=config.horizon, cells=cells)


def _cell(result: CU36Result, verifier: str, leg: str) -> VerifierLegResult:
    return next(c for c in result.cells if c.verifier == verifier and c.leg == leg)


def cu36_verdict(result: CU36Result) -> dict[str, object]:
    """H129: CU35's verifier-fidelity law governs real verifiers; the dual condition
    ``faithful_on_surface`` is decided a priori by the danger grammar (which channel it mutates),
    and a verifier is exactly as safe as the oracle iff it is faithful on the surface, however
    globally wrong it is elsewhere.
    """
    real = [c for c in result.cells if c.verifier in ("state-diff", "structure")]
    exact = [c for c in result.cells if c.verifier == "exact oracle"]
    no_gate = {c.leg: c for c in result.cells if c.verifier == "no gate"}

    # localization: faithful-on-surface <=> as-safe-as-oracle (breach 0); else as-blind-as-no-gate.
    def localized(c: VerifierLegResult) -> bool:
        if c.faithful_on_surface:
            return c.adversarial_breach <= 1e-9
        return c.adversarial_breach >= no_gate[c.leg].adversarial_breach - 1e-9

    # the structural predictor: CU35's empirical dual condition == the a-priori grammar fact.
    predictor_exact = all(c.faithful_on_surface == c.structurally_faithful for c in result.cells)

    # each real verifier is globally partial (blind on at least one leg) yet locally exact.
    def globally_partial_locally_safe(name: str) -> bool:
        cs = [c for c in real if c.verifier == name]
        blind = [c for c in cs if not c.faithful_on_surface]
        safe = [c for c in cs if c.faithful_on_surface]
        return (
            bool(blind) and bool(safe)
            and all(c.adversarial_breach <= 1e-9 for c in safe)
            and all(c.adversarial_breach >= 0.5 for c in blind)
        )

    def adv(verifier: str, leg: str) -> float:
        return _cell(result, verifier, leg).adversarial_breach

    return {
        "n_worlds": 3,
        "exact_safe_everywhere": all(c.adversarial_breach <= 1e-9 for c in exact),
        "no_gate_leaks_everywhere": all(c.adversarial_breach >= 0.5 for c in no_gate.values()),
        "localization_holds_everywhere": all(localized(c) for c in real),
        "structural_predictor_exact": predictor_exact,
        # the two real verifiers' grammar-predicted footprints
        "state_diff_safe_integrity": adv("state-diff", "integrity") <= 1e-9,
        "state_diff_safe_availability": adv("state-diff", "availability") <= 1e-9,
        "state_diff_blind_confidentiality": adv("state-diff", "confidentiality") >= 0.5,
        "structure_safe_availability": adv("structure", "availability") <= 1e-9,
        "structure_blind_integrity": adv("structure", "integrity") >= 0.5,
        "structure_blind_confidentiality": adv("structure", "confidentiality") >= 0.5,
        "state_diff_globally_partial_locally_safe": globally_partial_locally_safe("state-diff"),
        "structure_globally_partial_locally_safe": globally_partial_locally_safe("structure"),
        "all_targets_cover": all(c.target_covers for c in result.cells),
        "cells": [
            {
                "verifier": c.verifier, "leg": c.leg, "danger_mutates": c.danger_mutates,
                "faithful_on_surface": c.faithful_on_surface,
                "structurally_faithful": c.structurally_faithful,
                "adversarial_breach": c.adversarial_breach, "random_breach": c.random_breach,
                "mean_false_blocks": c.mean_false_blocks,
            }
            for c in result.cells
        ],
    }


CSV_HEADER = (
    "verifier,leg,danger_mutates,verifier_observes,faithful_on_surface,structurally_faithful,"
    "faithful_fraction,adversarial_breach,random_breach,mean_false_blocks,target_covers,n_episodes"
)


def write_csv(result: CU36Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in result.cells:
        rows.append(
            f"{c.verifier},{c.leg},{c.danger_mutates},{c.verifier_observes},"
            f"{int(c.faithful_on_surface)},{int(c.structurally_faithful)},"
            f"{c.faithful_fraction:.6f},{c.adversarial_breach:.6f},{c.random_breach:.6f},"
            f"{c.mean_false_blocks:.6f},{int(c.target_covers)},{result.n_episodes}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)


def main() -> None:  # pragma: no cover - exercised via the experiment CLI
    result = run_cu36()
    v = cu36_verdict(result)
    print("CU36 / H129 -- the grounded verifier (CU35's fidelity law against real verifiers):")
    print(f"  {result.n_episodes} deployments, horizon {result.horizon}\n")
    print(f"    {'verifier':14s} {'leg':16s} {'mutates':24s} {'faithful':>9s} {'adv':>6s}")
    for c in result.cells:
        print(f"    {c.verifier:14s} {c.leg:16s} {c.danger_mutates:24s} "
              f"{('yes' if c.faithful_on_surface else 'NO'):>9s} {c.adversarial_breach:6.3f}")
    print(f"\n    localization holds everywhere = {v['localization_holds_everywhere']}")
    print(f"    structural predictor exact (faithful <=> grammar) = "
          f"{v['structural_predictor_exact']}")


if __name__ == "__main__":  # pragma: no cover
    main()
