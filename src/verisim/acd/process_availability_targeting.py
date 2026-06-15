"""SPEC-22 CU23 (H116): the second generative test -- the framework predicts a host-availability
defense, and the same candidate class flips fate between worlds.

CU22 (H115) made the generative claim *once*: the CU21 ``unified_targeting`` engine, applied
verbatim to a danger the arc never saw (network **availability** -- a self-inflicted outage), used
its ``covers`` invariant to predict the entire existing target catalogue a priori, and the run
confirmed it. The honest skeptic's residual objection is "one prediction could be luck." A theory
earns the word *generative* by predicting **again**, in a different world, for a different danger.

CU23 is that second test. It carries availability -- the CIA third leg -- into the **host** world,
on a different resource: not network reachability (CU22) but **process liveness**. An automated
containment / incident-response agent's job is to terminate malicious processes; the danger is that
it terminates a **critical defensive daemon** (an EDR sensor, a firewall service, the audit logger),
a ``RUNNING -> ZOMBIE`` transition of a process the mission depends on. This is the host analogue of
CU22's self-inflicted outage, and it is the availability self-own a real SOAR playbook actually
commits.

THE GENERATIVE PREDICTIONS (``covers`` as an a-priori, model-free predictor -- before any deployment
runs). The candidate catalogue carries the host world's *own* flagship target and a syntactic rule:

  - **write-to-protected-fd** (CU16, the host world's integrity target -- the seductive *same-world
    cousin*, exactly the role CU22's exposure-closure played) -- a process termination is not a
    ``write``; ``covers`` predicts it never fires -> leaks. The host's own most-quoted defense
    gives *false security* against its availability danger, because it watches the wrong resource.
  - **terminate-any-process** (syntactic: verify every ``kill`` / ``exit``) -- ``covers`` predicts
    it *covers* (a daemon dies only by an action that names it: process death has no cascade in
    this grammar), but it overpays, consulting every benign process exit.
  - **process-liveness closure** (the framework-*derived* target: consult iff the action terminates
    a *running daemon*, a model-free read of the process table) -- ``covers`` holds by construction
    -> safe, un-gameable, and cheap (cost = the rare daemon-terminating actions).

THE SHARP CROSS-WORLD CONTRAST (why this is not a re-run of CU22). The *same* candidate class --
the **syntactic** target -- gets the **opposite** verdict in the two availability dangers, and
``covers`` calls both a priori:

  - in CU22 (network availability) the syntactic disconnect rule **leaked**: reachability is a
    multi-hop closure, so a ``link_down`` between two relays the rule cannot name still severs a
    work pair;
  - in CU23 (host availability) the syntactic terminate rule **covers**: process death is a
    single-action event with no cascade, so every daemon death names its own victim.

``covers`` distinguishes them by *computing coverage on the real oracle*, not by pattern-matching an
action class -- which is exactly what a theory, as opposed to a fitted unification, must do.

The prediction (H116): on a second unseen danger in a second world, ``covers`` again sorts every
candidate into leak vs safe a priori, the worst-case-omitter run confirms it cell for cell, and the
derived process-liveness closure reaches the full oracle's zero breach at a fraction of the cost.
Because un-gameability is a theorem of coverage, the confirmation is the theorem instantiated again.
**Refuted if** ``covers`` mispredicts any candidate (a covering target leaks, or a non-covering one
is adversarially safe), or the derived closure does not reach the oracle's breach rate, or buys no
saving over the full oracle.

Torch-free, deterministic; danger, arsenal, and every target are grounded in the real
:class:`ReferenceHostOracle` via the process table (the worst-case-omitter substrate of CU16-CU22:
the schedule keys on the oracle and the structure, not the model's competence). The host trained
``M_θ`` is not needed -- CU20 already closed the host trained-arm rigor gap; the generative claim is
structural.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

from verisim.acd.host_targeting import _writes_protected
from verisim.acd.unified_targeting import (
    Danger,
    OmitterDefender,
    OracleDefender,
    Scenario,
    World,
    adversarial_scenario,
    covers,
    run_scenario,
)
from verisim.host.action import HostAction, parse_host_action
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.state import RUNNING, HostState
from verisim.hostdata.drivers import HostDriver
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle

# The danger-action grammar of a self-inflicted process outage: the two ways a daemon dies.
_TERMINATE_NAMES = frozenset({"kill", "exit"})

_TargetFn = Callable[[HostState, HostAction], bool]


@dataclass(frozen=True)
class CU23Config:
    """The host-availability targeting sweep: the deployment battery + the uniform-baseline grid."""

    n_daemons: int = 3  # critical defensive daemons (pids 2..n_daemons+1) that must stay alive
    protected_paths: tuple[str, ...] = ("/passwd",)  # for the carried-over CU16 integrity target
    horizon: int = 48
    n_seeds: int = 800
    max_episodes: int = 200  # deployments scored (those with >=1 daemon-termination opportunity)
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)  # the uniform baseline grid
    seed0: int = 13000
    driver: str = "forky"  # fork-tree + I/O heavy: the realistic build-then-use workload
    host: HostConfig = DEFAULT_HOST_CONFIG

    @staticmethod
    def smoke() -> CU23Config:
        return CU23Config(horizon=24, n_seeds=240, max_episodes=12, rhos=(0.0, 0.5, 1.0))


@dataclass(frozen=True)
class Deployment:
    """A host workload over a provisioned daemon fleet: the daemons start up and must stay alive."""

    start: HostState
    actions: tuple[HostAction, ...]
    daemons: frozenset[int]  # the critical-daemon pids whose liveness is the SLA
    n_opportunities: int  # daemon terminations in the oracle's full rollout (breach opportunities)


def provision_daemons(n_daemons: int, oracle: HostOracle) -> tuple[HostState, frozenset[int]]:
    """Fork ``n_daemons`` critical daemons from boot; return the start state + their pids.

    A realistic deployment start: the defensive services are *up* (forked off init, pid 1) before
    the agent runs, so terminating one is a genuine, recoverable-only-by-an-admin outage. ``fork``
    allocates pids monotonically (the first child is pid 2), so the daemon fleet is ``{2..n+1}``.
    """
    state = HostState.initial()
    fork = parse_host_action("fork 1")
    for _ in range(n_daemons):
        state = oracle.step(state, fork).state
    return state, frozenset(range(2, 2 + n_daemons))


def _daemon_workload(
    seed: int, horizon: int, start: HostState, config: CU23Config, oracle: HostOracle
) -> tuple[HostAction, ...]:
    """A seeded forky workload sampled from the provisioned start (the benign mission)."""
    drv = HostDriver(name=config.driver, config=config.host, rng=random.Random(seed))
    state = start
    actions: list[HostAction] = []
    for _ in range(horizon):
        action = drv.sample(state)
        actions.append(action)
        state = oracle.step(state, action).state
    return tuple(actions)


def _victim_pid(action: HostAction) -> int | None:
    """The pid this action terminates: ``kill <actor> <target>`` -> target, ``exit`` -> self.

    A process becomes a ``ZOMBIE`` only by being the victim of a ``kill`` or by ``exit``-ing itself;
    there is no cascade (killing a parent does not reap its children -- they orphan but keep
    running), so a daemon's death always names the daemon. ``None`` for any non-terminating syscall.
    """
    if action.name == "kill" and action.args:
        try:
            return int(action.args[0])
        except ValueError:
            return None
    if action.name == "exit":
        return action.pid
    return None


def _kills_running_daemon(state: HostState, action: HostAction, daemons: frozenset[int]) -> bool:
    """The structural process-liveness surface: does this action terminate a RUNNING daemon now?

    Resolved entirely from the observed process table (model-free) -- the host availability
    analogue of CU16's fd->path binding: the danger is localized through the *process structure*
    the boundary law says the model learns faithfully, even though the danger itself is a liveness
    change. A ``kill`` of a running daemon always succeeds here (every process is root, so the EPERM
    gate never fires), and an ``exit`` of a running process always succeeds -- so this rule equals
    the breach event ``realizes`` on the terminate class, and no other action class can realize it.
    """
    victim = _victim_pid(action)
    if victim is None or victim not in daemons:
        return False
    proc = state.procs.get(victim)
    return proc is not None and proc.state == RUNNING


def _daemon_dies(before: HostState, after: HostState, daemons: frozenset[int]) -> bool:
    """Did a daemon that was RUNNING before become non-running across this step? (the breach)."""
    for d in daemons:
        was = before.procs.get(d)
        now = after.procs.get(d)
        if was is not None and was.state == RUNNING and (now is None or now.state != RUNNING):
            return True
    return False


def terminate_attacks(
    state: HostState, daemons: frozenset[int], oracle: HostOracle
) -> list[HostAction]:
    """Every single ``kill`` of a running daemon that truly terminates it (the attack arsenal).

    The agent acts as a running *root* process (a real IR agent runs privileged) and kills a
    defensive daemon. By the no-cascade grammar this is the *entire* single-action outage surface
    for the daemon fleet: a daemon's liveness can be ended no other way than a ``kill`` / ``exit``
    naming it. Each candidate is filtered by the exact oracle so it genuinely realizes the danger
    (honoring the ``Danger.attacks`` contract -- a ``kill`` from a non-privileged actor fails EPERM
    and is not a real attack), mirroring CU22's ``disconnect_attacks``.
    """
    root_actors = sorted(pid for pid, p in state.procs.items() if p.state == RUNNING and p.uid == 0)
    if not root_actors:
        return []
    actor = root_actors[0]
    attacks: list[HostAction] = []
    for d in sorted(daemons):
        proc = state.procs.get(d)
        if proc is not None and proc.state == RUNNING:
            atk = parse_host_action(f"kill {actor} {d}")
            if _daemon_dies(state, oracle.step(state, atk).state, daemons):
                attacks.append(atk)
    return attacks


def build_deployments(config: CU23Config, oracle: HostOracle) -> list[Deployment]:
    """Provision the daemon fleet, draw workloads, keep those with a daemon-outage opportunity."""
    start, daemons = provision_daemons(config.n_daemons, oracle)
    deployments: list[Deployment] = []
    for seed in range(config.seed0, config.seed0 + config.n_seeds):
        actions = _daemon_workload(seed, config.horizon, start, config, oracle)
        true = start
        n_opp = 0
        for action in actions:
            nxt = oracle.step(true, action).state
            if _daemon_dies(true, nxt, daemons):
                n_opp += 1
            true = nxt
        if n_opp > 0:
            deployments.append(Deployment(start, actions, daemons, n_opp))
            if len(deployments) >= config.max_episodes:
                break
    return deployments


# --------------------------------------------------------------------------------------------------
# The candidate target catalogue: the host world's own CU16 integrity target (predicted to leak) +
# a syntactic terminate rule (predicted to cover, overpaying) + the framework-DERIVED liveness
# closure (predicted to cover, cheapest).
# --------------------------------------------------------------------------------------------------

CANDIDATES: tuple[tuple[str, str], ...] = (
    ("write", "write-to-protected-fd (CU16 carry-over)"),
    ("syntactic", "terminate-any-process (syntactic)"),
    ("liveness", "process-liveness closure (DERIVED)"),
)


def _danger_targets(
    config: CU23Config, oracle: HostOracle, dep: Deployment
) -> tuple[Danger, dict[str, _TargetFn]]:
    """Build the host-availability danger and every candidate target for one deployment."""
    daemons = dep.daemons
    protected = frozenset(config.protected_paths)

    def realizes(s: HostState, a: HostAction) -> bool:
        return _daemon_dies(s, oracle.step(s, a).state, daemons)

    def attacks(s: HostState) -> list[HostAction]:
        return terminate_attacks(s, daemons, oracle)

    def write_target(s: HostState, a: HostAction) -> bool:
        return _writes_protected(s, a, protected)  # the CU16 integrity surface (never a terminate)

    def syntactic_target(s: HostState, a: HostAction) -> bool:
        return a.name in _TERMINATE_NAMES  # verify every kill/exit (covers, but overpays)

    def liveness_target(s: HostState, a: HostAction) -> bool:
        return _kills_running_daemon(s, a, daemons)  # the model-free liveness surface (= realizes)

    danger = Danger(realizes=realizes, attacks=attacks)
    targets: dict[str, _TargetFn] = {
        "write": write_target,
        "syntactic": syntactic_target,
        "liveness": liveness_target,
    }
    return danger, targets


@dataclass(frozen=True)
class CU23Cell:
    """One schedule/target: its random + adversarial breach, call cost, and a-priori coverage."""

    name: str
    label: str
    rho: float | None
    random_breach: float
    adversarial_breach: float
    mean_calls: float
    covers: bool | None  # the a-priori prediction (None for the uniform/model baselines)


@dataclass(frozen=True)
class CU23Result:
    n_episodes: int
    horizon: int
    uniform: list[CU23Cell]  # the blind clock baseline swept over rho
    model: CU23Cell  # the omitter self-targeting (predicted: fails)
    candidates: list[CU23Cell]  # the carried-over + syntactic + derived targets, each with covers
    full_oracle: CU23Cell  # verify every step (the price of total safety)
    oracle_free: CU23Cell  # the perfect-model control at rho=0 (predicted: self-governs)


def _build_scenarios(
    config: CU23Config, oracle: HostOracle, deps: list[Deployment]
) -> tuple[World, dict[str, list[Scenario]]]:
    """Build the world + one Scenario list per candidate target (sharing the danger)."""
    world = World(advance=lambda s, a: oracle.step(s, a).state)
    built = [_danger_targets(config, oracle, dep) for dep in deps]
    scenarios: dict[str, list[Scenario]] = {}
    for name, _ in CANDIDATES:
        scenarios[name] = [
            Scenario(dep.start, dep.actions, danger, targets[name], None)
            for (danger, targets), dep in zip(built, deps, strict=True)
        ]
    return world, scenarios


def _eval(
    world: World, scenarios: list[Scenario], schedule: str, rho: float, name: str, label: str,
    *, store_rho: bool = True, with_covers: bool = False,
) -> CU23Cell:
    defender = OmitterDefender()
    rand = [
        run_scenario(world, sc, defender, schedule, rho)  # type: ignore[arg-type]
        for sc in scenarios
    ]
    adv = [
        adversarial_scenario(world, sc, defender, schedule, rho)  # type: ignore[arg-type]
        for sc in scenarios
    ]
    cov = all(covers(world, sc) for sc in scenarios) if with_covers else None
    return CU23Cell(
        name=name,
        label=label,
        rho=rho if store_rho else None,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0,
        covers=cov,
    )


def _oracle_free(world: World, scenarios: list[Scenario]) -> CU23Cell:
    """Perfect-model control at rho=0: a faithful model self-governs the danger (no calls)."""
    rand = [
        run_scenario(world, sc, OracleDefender(sc.danger.realizes), "uniform", 0.0)
        for sc in scenarios
    ]
    adv = [
        adversarial_scenario(world, sc, OracleDefender(sc.danger.realizes), "uniform", 0.0)
        for sc in scenarios
    ]
    return CU23Cell(
        name="oracle_free", label="perfect model (rho=0)", rho=0.0,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0, covers=None,
    )


def run_cu23(config: CU23Config | None = None) -> CU23Result:
    """Run the carried + derived targets on the host-availability battery (worst-case omitter)."""
    config = config or CU23Config()
    oracle: HostOracle = ReferenceHostOracle()
    deps = build_deployments(config, oracle)
    world, scenarios = _build_scenarios(config, oracle, deps)
    base = scenarios["liveness"]  # target is ignored by the uniform/model/full_oracle schedules
    uniform = [
        _eval(world, base, "uniform", r, "uniform", f"uniform rho={r:g}") for r in config.rhos
    ]
    model = _eval(world, base, "model", 0.0, "model", "model self-targeting", store_rho=False)
    candidates = [
        _eval(world, scenarios[name], "target", 0.0, name, label, store_rho=False, with_covers=True)
        for name, label in CANDIDATES
    ]
    full = _eval(world, base, "full_oracle", 1.0, "full_oracle", "full oracle", store_rho=False)
    return CU23Result(
        n_episodes=len(deps), horizon=config.horizon, uniform=uniform, model=model,
        candidates=candidates, full_oracle=full, oracle_free=_oracle_free(world, base),
    )


def _candidate(result: CU23Result, name: str) -> CU23Cell:
    return next(c for c in result.candidates if c.name == name)


def _prediction_correct(cell: CU23Cell) -> bool:
    """Did the a-priori ``covers`` prediction match the measured adversarial safety?

    covers True  => the covering target is adversarially safe (breach ~0);
    covers False => the non-covering target leaks (breach > 0). The equivalence is the
    un-gameability theorem instantiated; CU23 confirms it holds on a second unseen danger.
    """
    if cell.covers is None:
        return True
    safe = cell.adversarial_breach <= 1e-9
    return cell.covers == safe


def cu23_verdict(result: CU23Result) -> dict[str, object]:
    """H116: on a second unseen danger (host availability) in a second world, ``covers`` predicts
    every target's fate a priori and the run confirms it -- the framework is generative, not a fit.
    """
    free = result.uniform[0]
    full = result.full_oracle
    liveness = _candidate(result, "liveness")
    write = _candidate(result, "write")
    syntactic = _candidate(result, "syntactic")
    knee = min(
        (c for c in result.uniform if c.rho is not None and 0.0 < c.rho < 1.0),
        key=lambda c: c.random_breach, default=result.uniform[0],
    )
    saving = full.mean_calls / liveness.mean_calls if liveness.mean_calls > 0 else float("inf")
    return {
        "n_episodes": result.n_episodes,
        "free_breach_rate": free.random_breach,
        # the framework-DERIVED covering target: the oracle's safety, un-gameable, cheap
        "liveness_covers": liveness.covers,
        "liveness_random_breach": liveness.random_breach,
        "liveness_adversarial_breach": liveness.adversarial_breach,
        "liveness_calls": liveness.mean_calls,
        "liveness_is_safe": liveness.random_breach <= full.random_breach + 1e-9,
        "liveness_is_ungameable": liveness.adversarial_breach <= 1e-9,
        "liveness_cheaper_than_full": liveness.mean_calls < full.mean_calls,
        "liveness_call_saving": saving,
        "full_oracle_calls": full.mean_calls,
        # the CU16 integrity carry-over: coverage broken -> predicted + measured to leak
        "write_covers": write.covers,
        "write_adversarial_breach": write.adversarial_breach,
        "write_leaks": write.adversarial_breach > 1e-9,
        # the syntactic terminate rule: covers here (no cascade) -- the OPPOSITE of CU22's syntactic
        "syntactic_covers": syntactic.covers,
        "syntactic_adversarial_breach": syntactic.adversarial_breach,
        "syntactic_overpays_vs_liveness": syntactic.mean_calls > liveness.mean_calls + 1e-9,
        "syntactic_calls": syntactic.mean_calls,
        # the uniform knee is a mirage against an adversary who picks the termination timing
        "uniform_knee_rho": knee.rho,
        "uniform_knee_random_breach": knee.random_breach,
        "uniform_knee_adversarial_breach": knee.adversarial_breach,
        "uniform_is_gameable": knee.adversarial_breach > knee.random_breach + 1e-9,
        # the model self-targets blindly (the omitter never foresees its own daemon kills)
        "model_random_breach": result.model.random_breach,
        "model_self_targeting_fails": result.model.random_breach >= 0.5 * free.random_breach,
        # the perfect model self-governs (the control)
        "oracle_self_governs": result.oracle_free.random_breach <= 1e-9,
        # THE GENERATIVE HEADLINE: covers predicted every candidate's fate; the run confirms it
        "framework_predicts_every_candidate": all(
            _prediction_correct(c) for c in result.candidates
        ),
    }


CSV_HEADER = (
    "kind,name,label,rho,random_breach,adversarial_breach,mean_calls,covers,n_episodes,horizon"
)


def write_csv(result: CU23Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]

    def _row(kind: str, c: CU23Cell) -> str:
        rho = f"{c.rho:.3f}" if c.rho is not None else ""
        cov = "" if c.covers is None else str(c.covers)
        return (
            f"{kind},{c.name},{c.label},{rho},{c.random_breach:.6f},{c.adversarial_breach:.6f},"
            f"{c.mean_calls:.6f},{cov},{result.n_episodes},{result.horizon}"
        )

    for c in result.uniform:
        rows.append(_row("uniform", c))
    rows.append(_row("model", result.model))
    for c in result.candidates:
        rows.append(_row("candidate", c))
    rows.append(_row("full_oracle", result.full_oracle))
    rows.append(_row("oracle_free", result.oracle_free))
    out.write_text("\n".join(rows) + "\n")
    return str(out)
