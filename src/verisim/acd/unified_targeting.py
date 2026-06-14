"""SPEC-22 CU21 (H114): the unified target -- one model-free rule across all three worlds.

The targeting arc shipped four hand-built defenses, one per danger, that each looked bespoke:

  - **CU10/CU11** (network exfil): verify a ``connect`` to a crown jewel;
  - **CU16** (host corruption): verify a ``write`` to an fd bound to a protected path;
  - **CU17** (network segmentation): verify the actions that flip ``can_reach`` to a jewel (the
    reachability *closure*, not a syntactic action class);
  - **CU18** (distributed staleness): verify a ``get`` iff the medium shows the read is stale.

CU21 is the capstone that proves these are **one rule**, not four tricks. Strip each to its parts
and the same three model-free objects appear every time:

  - a danger **D.realizes(state, action)** -- the exact breach event, computed on the observed
    structure via the exact oracle (a flow to a jewel opens / a protected file is corrupted / a
    stale value is consumed / a jewel becomes reachable). It is *never* the drifting world model.
  - a danger **D.attacks(state)** -- every single action that realizes D from here (the adversary's
    arsenal); each one satisfies ``realizes``.
  - a **target(state, action)** -- the consult rule: spend an oracle call iff the action is on D's
    surface. Also model-free.

The single unified schedule is **"consult iff target(state, action)"**, and the whole arc's headline
results follow from one property of the target -- *coverage*:

    COVERAGE:  for every state and action,  D.realizes(s, a)  =>  target(s, a).

THE UN-GAMEABILITY THEOREM (the arc's CU11/CU16/CU18 result, now one proof). Take any defender model
M, any deployment, any attacker timing. Under the target schedule the attacker can win only by
executing an ``a`` with ``realizes(s, a)`` that is not blocked. But ``realizes(s, a)`` implies
``target(s, a)`` by coverage, so the agent consults the oracle, which sees the true ``realizes`` and
blocks. The consult decision never reads M, so the bound is **model-independent**: a covering,
model-free target is un-gameable at a cost of exactly the number of on-surface actions -- a rare,
cheap surface. (See :func:`covers` for the empirical check that the hypothesis holds.)

THE BOUNDARY (CU17/CU18, now one mechanism). A target that *breaks* coverage -- the ``connect``
shortcut carried into the segmentation world (CU17), the genesis-``write`` shortcut carried into
the distributed world (CU18) -- leaks exactly the danger it fails to cover: there is an ``a`` with
``realizes(s, a)`` and ``not shortcut(s, a)``, so it executes uncaught. The cheapness of CU10-CU16
was never magic; it was coverage on a *sparse* surface. Get the surface wrong and you buy false
security.

So the program's most-quoted result is not network-specific, not host-specific, not a
sparse-grammar artifact: it is the statement that **danger in an oracle-grounded world has a
model-free surface, and verifying that surface is cheap, safe, and un-gameable -- provided the
surface covers the danger.**

This module is the worst-case-omitter substrate (the CU16/CU17/CU18 methodology): the schedule
keys on the oracle and the surface, not the model's competence, so the worst-case omitter (foresees
no danger) is the right model to defend against, and the perfect oracle model is the control. The
per-world trained arms already closed the rigor gap (CU5-net/CU8 net, CU19 dist, CU20 host); the
unified claim here is structural. Torch-free, deterministic; danger and surfaces are grounded in the
real reference oracles via the existing per-world helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal, Protocol

from verisim.acd.adversarial_targeting import reachable_exfils as net_reachable_exfils
from verisim.acd.closed_loop_net import _new_flows
from verisim.acd.dist_targeting import (
    CU18Config,
    _is_sensitive_read,
    _is_sensitive_write,
    is_stale,
    reachable_stale_reads,
)
from verisim.acd.dist_targeting import build_deployments as dist_build_deployments
from verisim.acd.host_targeting import (
    CU16Config,
    _writes_protected,
)
from verisim.acd.host_targeting import build_deployments as host_build_deployments
from verisim.acd.host_targeting import (
    reachable_exfils as host_reachable_exfils,
)
from verisim.acd.segmentation_targeting import (
    CU17Config,
    _flips_reachability,
    build_segments,
    reachable_exposures,
)
from verisim.acd.targeted_verification import CU10Config
from verisim.acd.targeted_verification import build_deployments as net_build_deployments
from verisim.distoracle.reference import ReferenceDistOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.netoracle.reference import ReferenceNetworkOracle

Schedule = Literal["uniform", "model", "target", "full_oracle"]

# Heterogeneous across worlds (network / host / distributed states + actions). The generic core is
# typed ``Any`` -- the per-world builders below are concretely grounded in the reference oracles.
State = Any
Action = Any
Target = Callable[[State, Action], bool]


@dataclass(frozen=True)
class Danger:
    """A world's danger ``D``: the exact breach event + the attacker's single-action arsenal.

    Both are MODEL-FREE -- functions of the observed structure and the exact oracle, never the
    drifting world model. ``realizes(s, a)`` is True iff executing ``a`` from ``s`` realizes the
    danger; it is both the ground-truth breach test and the oracle's gate verdict. ``attacks(s)``
    enumerates every single action that realizes the danger now (each ``a`` has ``realizes(s, a)``).
    """

    realizes: Callable[[State, Action], bool]
    attacks: Callable[[State], list[Action]]


@dataclass(frozen=True)
class World:
    """The mission's state evolution -- the exact oracle step (how the world advances)."""

    advance: Callable[[State, Action], State]


class Defender(Protocol):
    """A world model's danger preview: does it foresee this action realizing the danger?"""

    def foresees(self, state: State, action: Action) -> bool: ...


@dataclass(frozen=True)
class OmitterDefender:
    """The worst-case world model: foresees NO danger, ever (the gate's blind spot, CU8/CU16)."""

    def foresees(self, state: State, action: Action) -> bool:
        return False


@dataclass(frozen=True)
class OracleDefender:
    """The perfect-model control (recall 1): foresees danger exactly (the real surface)."""

    realizes: Callable[[State, Action], bool]

    def foresees(self, state: State, action: Action) -> bool:
        return bool(self.realizes(state, action))


@dataclass(frozen=True)
class Scenario:
    """One deployment: a benign mission, plus the danger and target(s) defined over its world.

    ``target`` is the covering unified rule's per-world instance; ``shortcut`` (if any) is a
    non-covering target carried over from another world (the CU17/CU18 boundary demonstration).
    """

    start: State
    actions: tuple[Action, ...]
    danger: Danger
    target: Target
    shortcut: Target | None


def _consult(
    schedule: Schedule, scenario: Scenario, defender: Defender, state: State, action: Action,
    i: int, interval: int, rho: float, *, use_shortcut: bool,
) -> bool:
    """The unified consult rule: when does this schedule spend an oracle call on this action?"""
    if schedule == "uniform":
        return rho >= 1.0 or bool(interval and i % interval == 0)
    if schedule == "target":
        tgt = scenario.shortcut if use_shortcut else scenario.target
        return bool(tgt(state, action)) if tgt is not None else False
    if schedule == "model":
        return defender.foresees(state, action)
    return True  # full_oracle


def run_scenario(
    world: World, scenario: Scenario, defender: Defender, schedule: Schedule, rho: float = 0.0,
    *, use_shortcut: bool = False,
) -> tuple[bool, int]:
    """Run the benign mission under one schedule; return ``(breached, oracle_calls)``.

    At each action the agent previews the danger: it spends an oracle call iff the schedule says to
    (re-anchoring to the truth and gating on the oracle's verdict), else it trusts the model. A
    breach is the first time the agent EXECUTES an action that truly realizes the danger uncaught.
    """
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    danger = scenario.danger
    state = scenario.start
    breached = False
    calls = 0
    for i, action in enumerate(scenario.actions, start=1):
        realizes = danger.realizes(state, action)
        consult = _consult(schedule, scenario, defender, state, action, i, interval, rho,
                            use_shortcut=use_shortcut)
        if consult:
            calls += 1
            block = realizes  # the oracle's true verdict
        else:
            block = defender.foresees(state, action)  # the model's verdict
        if realizes and not block:
            breached = True
        if not (realizes and block):  # advance unless a danger action was blocked (and skipped)
            state = world.advance(state, action)
    return breached, calls


def adversarial_scenario(
    world: World, scenario: Scenario, defender: Defender, schedule: Schedule, rho: float = 0.0,
    *, use_shortcut: bool = False,
) -> bool:
    """Worst-case over the attacker's choice of danger action AND timing (the CU11 probe, unified).

    Replays the benign mission and, before each step, probes whether the attacker could inject a
    single action from the danger's arsenal that the schedule would not catch (not consulted, and
    the model foresees nothing). The deployment is breached if any single placement succeeds.
    """
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    danger = scenario.danger
    state = scenario.start
    for i, action in enumerate(scenario.actions, start=1):
        for atk in danger.attacks(state):  # each atk realizes the danger by construction
            consult = _consult(schedule, scenario, defender, state, atk, i, interval, rho,
                               use_shortcut=use_shortcut)
            block = danger.realizes(state, atk) if consult else defender.foresees(state, atk)
            if not block:
                return True  # an attack the schedule and model both miss -> the attacker wins
        realizes = danger.realizes(state, action)
        consult = _consult(schedule, scenario, defender, state, action, i, interval, rho,
                           use_shortcut=use_shortcut)
        block = realizes if consult else defender.foresees(state, action)
        if not (realizes and block):
            state = world.advance(state, action)
    return False


def covers(world: World, scenario: Scenario, *, use_shortcut: bool = False) -> bool:
    """The coverage invariant (the un-gameability theorem's hypothesis): does the target fire on
    every attack along the benign trajectory? (``realizes`` => ``target``).

    Walks the oracle trajectory and checks that every action in the danger's arsenal is on the
    target's surface. True for a covering target (the theorem applies -> un-gameable for any model);
    False for a non-covering shortcut (it leaks the uncovered danger).
    """
    tgt = scenario.shortcut if use_shortcut else scenario.target
    if tgt is None:
        return False
    state = scenario.start
    for action in scenario.actions:
        for atk in scenario.danger.attacks(state):
            if not tgt(state, atk):
                return False
        state = world.advance(state, action)
    return True


# --------------------------------------------------------------------------------------------------
# The four arms: one (World, Danger, target, shortcut) per danger, on the reference oracles.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One world+danger, its covering target, an optional non-covering shortcut, its deployments."""

    world_name: str
    danger_name: str
    target_name: str
    shortcut_name: str | None
    world: World
    scenarios: list[Scenario]
    horizon: int


def net_flow_arm(config: CU10Config | None = None) -> Arm:
    """Network exfil (CU10/CU11): danger = a flow to a crown jewel; target = connect-to-jewel."""
    config = config or CU10Config()
    oracle = ReferenceNetworkOracle()
    jewels = frozenset(config.protected_servers)
    deployments = net_build_deployments(config, oracle)
    world = World(advance=lambda s, a: oracle.step(s, a).state)

    def realizes(s: State, a: Action) -> bool:
        return bool(_new_flows(s, oracle.step(s, a).state, jewels))

    def attacks(s: State) -> list[Action]:
        return net_reachable_exfils(s, jewels)

    def target(s: State, a: Action) -> bool:
        return a.name == "connect" and len(a.args) > 1 and a.args[1] in jewels

    danger = Danger(realizes=realizes, attacks=attacks)
    scenarios = [Scenario(d.start, d.actions, danger, target, None) for d in deployments]
    return Arm("network", "exfil flow to a crown jewel", "connect-to-jewel", None,
               world, scenarios, config.horizon)


def host_arm(config: CU16Config | None = None) -> Arm:
    """Host corruption (CU16): danger = a protected file corrupted; target = write-to-jewel-fd."""
    config = config or CU16Config()
    oracle = ReferenceHostOracle()
    protected = frozenset(config.protected_paths)
    deployments = host_build_deployments(config, oracle)
    world = World(advance=lambda s, a: oracle.step(s, a).state)

    def realizes(s: State, a: Action) -> bool:
        before = frozenset(p for p in protected if _is_written(s, p))
        after = frozenset(p for p in protected if _is_written(oracle.step(s, a).state, p))
        return bool(after - before)

    def attacks(s: State) -> list[Action]:
        return host_reachable_exfils(s, protected)

    def target(s: State, a: Action) -> bool:
        return _writes_protected(s, a, protected)

    danger = Danger(realizes=realizes, attacks=attacks)
    scenarios = [Scenario(d.start, d.actions, danger, target, None) for d in deployments]
    return Arm("host", "protected file corruption", "write-to-jewel-fd", None,
               world, scenarios, config.horizon)


def _is_written(state: State, path: str) -> bool:
    from verisim.acd.host_integrity import written_files

    return path in written_files(state)


def dist_arm(config: CU18Config | None = None) -> Arm:
    """Distributed staleness (CU18): danger = acting on a stale sensitive read; target = the medium.

    Carries the CU10-CU16 genesis-``write`` rule as the *shortcut* -- the boundary case that breaks
    coverage (a stale ``get`` is never a write, so the write target never consults it).
    """
    config = config or CU18Config()
    oracle = ReferenceDistOracle(config.dist)
    sensitive = frozenset(config.sensitive_keys)
    deployments = dist_build_deployments(config, oracle)
    world = World(advance=lambda s, a: oracle.step(s, a).state)

    def realizes(s: State, a: Action) -> bool:
        return _is_sensitive_read(a, sensitive) and is_stale(s, a.args[0], a.args[1])

    def attacks(s: State) -> list[Action]:
        return reachable_stale_reads(s, sensitive, config.dist)

    def medium_target(s: State, a: Action) -> bool:
        return realizes(s, a)  # consult iff the medium shows the sensitive read is stale

    def write_shortcut(s: State, a: Action) -> bool:
        return _is_sensitive_write(a, sensitive)  # genesis-action rule (does not cover the read)

    danger = Danger(realizes=realizes, attacks=attacks)
    scenarios = [
        Scenario(d.start, d.actions, danger, medium_target, write_shortcut) for d in deployments
    ]
    return Arm("distributed", "agent acts on a stale sensitive value",
               "medium staleness at the read", "write-to-key (CU10 shortcut)",
               world, scenarios, config.horizon)


def net_reach_arm(config: CU17Config | None = None) -> Arm:
    """Network segmentation (CU17): danger = a jewel exposed to the untrusted set; target = closure.

    Carries the CU10-CU16 ``connect`` rule as the *shortcut* -- the boundary case that breaks
    coverage (a config-borne exposure is not a ``connect``, so the connect target never sees it).
    """
    config = config or CU17Config()
    oracle = ReferenceNetworkOracle()
    jewels = frozenset(config.jewels)
    ports = tuple(config.net.ports)
    segments = build_segments(config, oracle)
    world = World(advance=lambda s, a: oracle.step(s, a).state)

    scenarios: list[Scenario] = []
    for seg in segments:
        untrusted = seg.untrusted

        def realizes(s: State, a: Action, _u: tuple[str, ...] = untrusted) -> bool:
            return _flips_reachability(s, a, oracle, _u, jewels, ports)

        def attacks(s: State, _u: tuple[str, ...] = untrusted) -> list[Action]:
            return reachable_exposures(s, _u, jewels, config.net)

        def closure_target(s: State, a: Action, _u: tuple[str, ...] = untrusted) -> bool:
            return _flips_reachability(s, a, oracle, _u, jewels, ports)

        def connect_shortcut(s: State, a: Action) -> bool:
            return a.name == "connect" and len(a.args) > 1 and a.args[1] in jewels

        danger = Danger(realizes=realizes, attacks=attacks)
        scenarios.append(Scenario(seg.start, seg.actions, danger, closure_target, connect_shortcut))
    return Arm("network (segmentation)", "jewel reachable from the untrusted set",
               "reachability closure", "connect-to-jewel (CU10 shortcut)",
               world, scenarios, config.horizon)


# --------------------------------------------------------------------------------------------------
# The unified sweep + verdict.
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One schedule (at one ρ, for uniform): its random + adversarial breach and call cost."""

    schedule: str
    label: str
    rho: float | None
    random_breach: float
    adversarial_breach: float
    mean_calls: float


@dataclass(frozen=True)
class ArmResult:
    world_name: str
    danger_name: str
    target_name: str
    shortcut_name: str | None
    horizon: int
    n_scenarios: int
    uniform: list[Cell]  # the blind clock swept over ρ
    model: Cell  # self-targeting omitter (predicted: fails)
    target: Cell  # the unified covering target (predicted: safe, cheap, un-gameable)
    full_oracle: Cell  # verify every step (the price of total safety)
    shortcut: Cell | None  # a non-covering shortcut (predicted: leaks adversarially)
    oracle_free: Cell  # the perfect-model control at ρ=0 (predicted: self-governs)
    target_covers: bool  # the coverage invariant holds for the covering target
    shortcut_covers: bool | None  # coverage is broken for the shortcut


DEFAULT_RHOS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)


def _cell(
    arm: Arm, defender: Defender, schedule: Schedule, rho: float, label: str,
    *, store_rho: bool = True, use_shortcut: bool = False,
) -> Cell:
    rand = [
        run_scenario(arm.world, sc, defender, schedule, rho, use_shortcut=use_shortcut)
        for sc in arm.scenarios
    ]
    adv = [
        adversarial_scenario(arm.world, sc, defender, schedule, rho, use_shortcut=use_shortcut)
        for sc in arm.scenarios
    ]
    return Cell(
        schedule=schedule,
        label=label,
        rho=rho if store_rho else None,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0,
    )


def _oracle_free_cell(arm: Arm) -> Cell:
    """The perfect-model control at ρ=0: a faithful model self-governs the danger (no oracle)."""
    rand = [
        run_scenario(arm.world, sc, OracleDefender(sc.danger.realizes), "uniform", 0.0)
        for sc in arm.scenarios
    ]
    adv = [
        adversarial_scenario(arm.world, sc, OracleDefender(sc.danger.realizes), "uniform", 0.0)
        for sc in arm.scenarios
    ]
    return Cell(
        schedule="oracle_free",
        label="perfect model (ρ=0)",
        rho=0.0,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0,
    )


def run_arm(arm: Arm, rhos: tuple[float, ...] = DEFAULT_RHOS) -> ArmResult:
    """Run the unified schedule (+ shortcut + controls) on one arm under the worst-case omitter."""
    omitter = OmitterDefender()
    uniform = [_cell(arm, omitter, "uniform", r, f"uniform ρ={r:g}") for r in rhos]
    model = _cell(arm, omitter, "model", 0.0, "model self-targeting", store_rho=False)
    target = _cell(arm, omitter, "target", 0.0, f"target ({arm.target_name})", store_rho=False)
    full = _cell(arm, omitter, "full_oracle", 1.0, "full oracle", store_rho=False)
    shortcut = (
        _cell(arm, omitter, "target", 0.0, f"shortcut ({arm.shortcut_name})",
              store_rho=False, use_shortcut=True)
        if any(sc.shortcut is not None for sc in arm.scenarios) else None
    )
    target_covers = all(covers(arm.world, sc) for sc in arm.scenarios)
    shortcut_covers = (
        all(covers(arm.world, sc, use_shortcut=True) for sc in arm.scenarios)
        if shortcut is not None else None
    )
    return ArmResult(
        world_name=arm.world_name, danger_name=arm.danger_name, target_name=arm.target_name,
        shortcut_name=arm.shortcut_name, horizon=arm.horizon, n_scenarios=len(arm.scenarios),
        uniform=uniform, model=model, target=target, full_oracle=full, shortcut=shortcut,
        oracle_free=_oracle_free_cell(arm), target_covers=target_covers,
        shortcut_covers=shortcut_covers,
    )


@dataclass(frozen=True)
class UnifiedResult:
    arms: list[ArmResult]


def run_cu21(
    arms: list[Arm] | None = None, rhos: tuple[float, ...] = DEFAULT_RHOS
) -> UnifiedResult:
    """Run the unified target on all four arms (net exfil + host + distributed + segmentation)."""
    arms = arms or [net_flow_arm(), host_arm(), dist_arm(), net_reach_arm()]
    return UnifiedResult(arms=[run_arm(a, rhos) for a in arms])


def _uniform_knee(arm: ArmResult) -> Cell:
    """The most favorable sub-oracle uniform budget on the random axis (the apparent knee)."""
    return min(
        (c for c in arm.uniform if c.rho is not None and 0.0 < c.rho < 1.0),
        key=lambda c: c.random_breach, default=arm.uniform[0],
    )


def arm_verdict(arm: ArmResult) -> dict[str, object]:
    """Per-arm: the covering target is safe + cheap + un-gameable; the shortcut (if any) leaks."""
    free = arm.uniform[0]
    knee = _uniform_knee(arm)
    saving = (
        arm.full_oracle.mean_calls / arm.target.mean_calls
        if arm.target.mean_calls > 0 else float("inf")
    )
    out: dict[str, object] = {
        "world": arm.world_name,
        "target_random_breach": arm.target.random_breach,
        "target_adversarial_breach": arm.target.adversarial_breach,
        "target_calls": arm.target.mean_calls,
        "full_oracle_calls": arm.full_oracle.mean_calls,
        "target_call_saving": saving,
        "target_is_safe": arm.target.random_breach <= arm.full_oracle.random_breach + 1e-9,
        "target_is_ungameable": arm.target.adversarial_breach <= 1e-9,
        "target_cheaper_than_full": arm.target.mean_calls < arm.full_oracle.mean_calls,
        "target_covers": arm.target_covers,
        "model_random_breach": arm.model.random_breach,
        "model_self_targeting_fails": arm.model.random_breach >= 0.5 * free.random_breach,
        "uniform_knee_rho": knee.rho,
        "uniform_knee_random_breach": knee.random_breach,
        "uniform_knee_adversarial_breach": knee.adversarial_breach,
        "uniform_is_gameable": knee.adversarial_breach > knee.random_breach + 1e-9,
        "oracle_free_breach": arm.oracle_free.random_breach,
        "oracle_self_governs": arm.oracle_free.random_breach <= 1e-9,
        "free_breach_rate": free.random_breach,
    }
    if arm.shortcut is not None:
        out["shortcut_random_breach"] = arm.shortcut.random_breach
        out["shortcut_adversarial_breach"] = arm.shortcut.adversarial_breach
        out["shortcut_calls"] = arm.shortcut.mean_calls
        out["shortcut_covers"] = arm.shortcut_covers
        out["shortcut_leaks"] = arm.shortcut.adversarial_breach > 1e-9
    return out


def cu21_verdict(result: UnifiedResult) -> dict[str, object]:
    """H114: one model-free covering target is safe + cheap + un-gameable in EVERY world; a
    non-covering shortcut leaks. The un-gameability is a theorem of coverage, not a per-world quirk.
    """
    arms = [arm_verdict(a) for a in result.arms]
    all_safe = all(bool(a["target_is_safe"]) for a in arms)
    all_ungameable = all(bool(a["target_is_ungameable"]) for a in arms)
    all_cheaper = all(bool(a["target_cheaper_than_full"]) for a in arms)
    all_cover = all(bool(a["target_covers"]) for a in arms)
    shortcuts = [a for a in arms if "shortcut_leaks" in a]
    all_shortcuts_leak = bool(shortcuts) and all(bool(a["shortcut_leaks"]) for a in shortcuts)
    all_shortcuts_break_coverage = all(a["shortcut_covers"] is False for a in shortcuts)
    return {
        "n_worlds": len(arms),
        "unified_target_safe_everywhere": all_safe,
        "unified_target_ungameable_everywhere": all_ungameable,
        "unified_target_cheaper_everywhere": all_cheaper,
        "unified_target_covers_everywhere": all_cover,
        "shortcuts_leak_everywhere": all_shortcuts_leak,
        "shortcuts_break_coverage_everywhere": all_shortcuts_break_coverage,
        "arms": arms,
    }


CSV_HEADER = (
    "world,schedule,label,rho,random_breach,adversarial_breach,mean_calls,n_scenarios,horizon"
)


def write_csv(result: UnifiedResult, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for arm in result.arms:
        cells = [*arm.uniform, arm.model, arm.target, arm.full_oracle, arm.oracle_free]
        if arm.shortcut is not None:
            cells.append(arm.shortcut)
        for c in cells:
            rho = f"{c.rho:.3f}" if c.rho is not None else ""
            rows.append(
                f"{arm.world_name},{c.schedule},{c.label},{rho},{c.random_breach:.6f},"
                f"{c.adversarial_breach:.6f},{c.mean_calls:.6f},{arm.n_scenarios},{arm.horizon}"
            )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
