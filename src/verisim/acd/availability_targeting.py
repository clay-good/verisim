"""SPEC-22 CU22 (H115): the generative test -- the unified target framework *predicts* a defense
for a danger it never saw.

CU21 proved that the four hand-built targeting defenses (network exfil, host corruption, network
segmentation, distributed staleness) are one model-free rule: "consult iff the action is on the
danger's surface", whose un-gameability is a theorem of *coverage* (``realizes => target``). But
CU21 *unified results we already had*. The honest skeptic's objection is "you designed the
abstraction to fit four answers you knew" -- a unification, not a theory. A theory must **predict**.

CU22 is that test. It applies the CU21 framework -- verbatim, importing
:mod:`verisim.acd.unified_targeting`'s ``Danger`` / ``World`` / ``Scenario`` / ``run_scenario`` /
``adversarial_scenario`` / ``covers`` engine -- to a danger that the entire CU10-CU21 arc never
studied: **availability**. The CU arc has been confidentiality (exfil) and integrity (corruption);
availability is the third leg of the CIA triad, and it is the one a real automated cyber-defender
breaks by *doing its job too aggressively*. An incident-response agent quarantines hosts and
tightens firewalls; the danger is a **self-inflicted outage** -- the agent executes a containment
action (``link_down`` / ``host_down`` / ``svc_down`` / ``fw_deny``) that disconnects a *critical
work service* from the clients that need it (``can_reach`` flipping ``True -> False`` for a required
pair). Like CU17's exposure, the surface is *semantic* and multi-hop: cutting a link between two
relays that are neither endpoint can still sever the only path, so the danger is the reachability
**closure**, not a syntactic action class.

THE GENERATIVE CLAIM. The framework's ``covers`` invariant is a **model-free, a-priori predictor**:
walk the oracle trajectory and check whether a candidate target fires on every attack in the
danger's arsenal -- *before running a single deployment*. CU22 carries over the existing target
catalogue as candidates and lets ``covers`` predict each one's fate:

  - **connect-to-jewel** (CU10/CU11, the program's most-quoted target) -- a disconnection is not a
    ``connect``; ``covers`` predicts it never fires -> leaks.
  - **exposure-closure** (CU17, the seductive same-world cousin -- *the same reachability
    machinery*) -- it watches reachability flipping *up* to a jewel; availability is reachability
    flipping *down* for a work pair. Opposite polarity, opposite endpoints; ``covers`` predicts it
    never fires -> leaks. The closest relative still leaks, and the framework says so in advance.
  - **syntactic-disconnect** -- verify the down/deny actions that *name* a work host (the CU17
    "grammar" analogue). ``covers`` predicts it leaks exactly when a multi-hop path runs through a
    relay it cannot name.
  - **disconnect-closure** -- the framework-*derived* target: ``target = realizes`` (consult iff the
    action disconnects a required pair, a model-free reachability query). ``covers`` holds by
    construction -> safe, un-gameable, and cheap (cost = the rare on-surface actions).

The prediction (H115): on a danger it never saw, ``covers`` correctly sorts the candidate targets
into leak vs safe *a priori*, and the worst-case-omitter run confirms it cell for cell -- every
non-covering carry-over leaks adversarially (breach 1.000), the derived covering target reaches the
full oracle's zero breach at a fraction of the cost. Because un-gameability is a theorem of
coverage, the run is not a fit; it is the theorem instantiated on an unseen danger. **Refuted if**
``covers``
mispredicts any candidate (a covering target leaks, or a non-covering one is adversarially safe), or
the derived disconnect-closure does not reach the oracle's breach rate, or buys no saving over the
full oracle.

Torch-free, deterministic; danger, arsenal, and every target are grounded in the real
:class:`ReferenceNetworkOracle` via ``can_reach`` (the worst-case-omitter substrate of CU16-CU21:
the schedule keys on the oracle and the surface, not the model's competence).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

from verisim.acd.segmentation_targeting import _flips_reachability
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
from verisim.net.action import NetAction
from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.net.state import NetworkState, can_reach
from verisim.netdelta.apply import apply
from verisim.netdelta.edits import HostUp, NetEdit, SetResult, SvcUp
from verisim.netoracle.base import NetOracle
from verisim.netoracle.reference import ReferenceNetworkOracle

# The danger-action grammar of a self-inflicted outage: every config op that can cut reachability.
_DISCONNECT_NAMES = frozenset({"link_down", "host_down", "svc_down", "fw_deny"})

_TargetFn = Callable[[NetworkState, NetAction], bool]


@dataclass(frozen=True)
class CU22Config:
    """The availability-targeting sweep: the deployment battery and the uniform-baseline grid."""

    work: tuple[str, ...] = ("h1", "h2", "h3")  # the critical work hosts whose service must survive
    jewels: tuple[str, ...] = ("h0", "h4")  # crown jewels (for the carried-over CU10/CU17 targets)
    horizon: int = 48
    n_seeds: int = 600
    max_episodes: int = 200  # deployments scored (those with >=1 disconnection opportunity)
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)  # the uniform baseline grid
    seed0: int = 11000
    driver: str = "weighted"
    net: NetConfig = DEFAULT_NET_CONFIG

    @staticmethod
    def smoke() -> CU22Config:
        return CU22Config(horizon=24, n_seeds=200, max_episodes=24, rhos=(0.0, 0.5, 1.0))


@dataclass(frozen=True)
class Deployment:
    """A workload over a *provisioned* network: the work pairs start reachable and must stay so."""

    start: NetworkState
    actions: tuple[NetAction, ...]
    pairs: frozenset[tuple[str, str, int]]  # required (client, server, port) reachability (the SLA)
    untrusted: tuple[str, ...]  # the non-jewel hosts (for the carried-over exposure target)
    n_opportunities: int  # disconnections in the oracle's full rollout


def provision_work(
    state: NetworkState, work: frozenset[str], ports: tuple[int, ...]
) -> NetworkState:
    """Bring the critical work hosts up + listening on ``ports[0]`` (the running services posture).

    A realistic deployment start: the mission-critical services are *up* and reachable over the
    connected seed topology, so disconnecting one is a genuine, recoverable-only-by-an-admin outage.
    """
    edits: list[NetEdit] = []
    for w in sorted(work):
        if w in state.hosts and not state.hosts[w].up:
            edits.append(HostUp(w))
        if w in state.hosts and ports[0] not in state.hosts[w].services:
            edits.append(SvcUp(w, ports[0]))
    return apply(state, [*edits, SetResult(0)])


def required_pairs(
    state: NetworkState, work: frozenset[str], ports: tuple[int, ...]
) -> frozenset[tuple[str, str, int]]:
    """Every work->work service reachable now on ``ports[0]`` -- the SLA to preserve."""
    p = ports[0]
    return frozenset(
        (a, b, p)
        for a in sorted(work)
        for b in sorted(work)
        if a != b and can_reach(state, a, b, p)
    )


def _connected(
    state: NetworkState, pairs: frozenset[tuple[str, str, int]]
) -> frozenset[tuple[str, str, int]]:
    """The subset of required pairs currently reachable."""
    return frozenset(pr for pr in pairs if can_reach(state, *pr))


def _breaks(
    before: NetworkState, after: NetworkState, pairs: frozenset[tuple[str, str, int]]
) -> bool:
    """Did the step disconnect a required pair that was reachable before? (the breach)."""
    return bool(_connected(before, pairs) - _connected(after, pairs))


def disconnect_attacks(
    state: NetworkState, pairs: frozenset[tuple[str, str, int]], oracle: NetOracle,
    config: CU22Config,
) -> list[NetAction]:
    """Every single config action that severs a required pair from ``state`` (the attack arsenal).

    The over-containment opportunity set at one step: the ``link_down`` / ``host_down`` /
    ``svc_down`` / ``fw_deny`` actions that flip ``can_reach`` to ``False`` for a required pair --
    including the multi-hop cases (a ``link_down`` between two relays that are neither endpoint),
    which is the entire disconnection surface by the reachability relation.
    """
    hosts = sorted(state.hosts)
    p = config.net.ports[0]
    candidates: list[NetAction] = []
    for a in hosts:
        for b in hosts:
            if a < b:
                candidates.append(
                    NetAction(raw=f"link_down {a} {b}", name="link_down", args=(a, b))
                )
    for h in hosts:
        candidates.append(NetAction(raw=f"host_down {h}", name="host_down", args=(h,)))
    for w in sorted(config.work):
        if w in state.hosts:
            candidates.append(
                NetAction(raw=f"svc_down {w} {p}", name="svc_down", args=(w, str(p)))
            )
            for src in hosts:
                if src != w:
                    candidates.append(
                        NetAction(raw=f"fw_deny {w} {src}", name="fw_deny", args=(w, src))
                    )
    return [a for a in candidates if _breaks(state, oracle.step(state, a).state, pairs)]


def build_deployments(config: CU22Config, oracle: NetOracle) -> list[Deployment]:
    """Draw workloads, provision the work services, keep those with an outage opportunity."""
    from verisim.acd.net_integrity import make_net_workload

    work = frozenset(config.work)
    jewels = frozenset(config.jewels)
    ports = tuple(config.net.ports)
    deployments: list[Deployment] = []
    for seed in range(config.seed0, config.seed0 + config.n_seeds):
        raw_start, actions = make_net_workload(
            seed, config.horizon, driver=config.driver, oracle=oracle, config=config.net
        )
        start = provision_work(raw_start, work, ports)
        pairs = required_pairs(start, work, ports)
        if not pairs:
            continue  # no critical connectivity to protect
        untrusted = tuple(h for h in sorted(start.hosts) if h not in jewels)
        true = start
        n_opp = 0
        for action in actions:
            nxt = oracle.step(true, action).state
            if _breaks(true, nxt, pairs):
                n_opp += 1
            true = nxt
        if n_opp > 0:
            deployments.append(Deployment(start, actions, pairs, untrusted, n_opp))
            if len(deployments) >= config.max_episodes:
                break
    return deployments


def _names_work(action: NetAction, work: frozenset[str]) -> bool:
    """The *syntactic* disconnect target: a down/deny action that NAMES a work host (CU17)."""
    return action.name in _DISCONNECT_NAMES and any(arg in work for arg in action.args)


# --------------------------------------------------------------------------------------------------
# The candidate target catalogue: the existing CU10/CU17 targets carried over (predicted to leak) +
# the syntactic disconnect rule + the framework-DERIVED disconnect closure (predicted to cover).
# --------------------------------------------------------------------------------------------------

CANDIDATES: tuple[tuple[str, str], ...] = (
    ("connect", "connect-to-jewel (CU10 carry-over)"),
    ("exposure", "exposure-closure (CU17 carry-over)"),
    ("syntactic", "syntactic disconnect (names a work host)"),
    ("closure", "disconnect-closure (DERIVED: target=realizes)"),
)


def _danger_targets(
    config: CU22Config, oracle: NetOracle, dep: Deployment
) -> tuple[Danger, dict[str, _TargetFn]]:
    """Build the availability danger and every candidate target for one deployment (model-free)."""
    pairs = dep.pairs
    jewels = frozenset(config.jewels)
    work = frozenset(config.work)
    ports = tuple(config.net.ports)
    untrusted = dep.untrusted

    def realizes(s: NetworkState, a: NetAction) -> bool:
        return _breaks(s, oracle.step(s, a).state, pairs)

    def attacks(s: NetworkState) -> list[NetAction]:
        return disconnect_attacks(s, pairs, oracle, config)

    def connect_target(s: NetworkState, a: NetAction) -> bool:
        return a.name == "connect" and len(a.args) > 1 and a.args[1] in jewels

    def exposure_target(s: NetworkState, a: NetAction) -> bool:
        return _flips_reachability(s, a, oracle, untrusted, jewels, ports)

    def syntactic_target(s: NetworkState, a: NetAction) -> bool:
        return _names_work(a, work)

    danger = Danger(realizes=realizes, attacks=attacks)
    targets: dict[str, _TargetFn] = {
        "connect": connect_target,
        "exposure": exposure_target,
        "syntactic": syntactic_target,
        "closure": realizes,  # the derived covering target IS the danger's realizes (model-free)
    }
    return danger, targets


@dataclass(frozen=True)
class CU22Cell:
    """One schedule/target: its random + adversarial breach, call cost, and a-priori coverage."""

    name: str
    label: str
    rho: float | None
    random_breach: float
    adversarial_breach: float
    mean_calls: float
    covers: bool | None  # the a-priori prediction (None for the uniform/model baselines)


@dataclass(frozen=True)
class CU22Result:
    n_episodes: int
    horizon: int
    uniform: list[CU22Cell]  # the blind clock baseline swept over rho
    model: CU22Cell  # the omitter self-targeting (predicted: fails)
    candidates: list[CU22Cell]  # the carried-over targets + the derived closure, each with covers
    full_oracle: CU22Cell  # verify every step (the price of total safety)
    oracle_free: CU22Cell  # the perfect-model control at rho=0 (predicted: self-governs)


def _build_scenarios(
    config: CU22Config, oracle: NetOracle, deps: list[Deployment]
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
) -> CU22Cell:
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
    return CU22Cell(
        name=name,
        label=label,
        rho=rho if store_rho else None,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0,
        covers=cov,
    )


def _oracle_free(world: World, scenarios: list[Scenario]) -> CU22Cell:
    """Perfect-model control at rho=0: a faithful model self-governs the danger (no calls)."""
    rand = [
        run_scenario(world, sc, OracleDefender(sc.danger.realizes), "uniform", 0.0)
        for sc in scenarios
    ]
    adv = [
        adversarial_scenario(world, sc, OracleDefender(sc.danger.realizes), "uniform", 0.0)
        for sc in scenarios
    ]
    return CU22Cell(
        name="oracle_free", label="perfect model (rho=0)", rho=0.0,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0, covers=None,
    )


def run_cu22(config: CU22Config | None = None) -> CU22Result:
    """Run the carried-over + derived targets on the availability battery (worst-case omitter)."""
    config = config or CU22Config()
    oracle: NetOracle = ReferenceNetworkOracle()
    deps = build_deployments(config, oracle)
    world, scenarios = _build_scenarios(config, oracle, deps)
    base = scenarios["closure"]  # target is ignored by the uniform/model/full_oracle schedules
    uniform = [
        _eval(world, base, "uniform", r, "uniform", f"uniform rho={r:g}") for r in config.rhos
    ]
    model = _eval(world, base, "model", 0.0, "model", "model self-targeting", store_rho=False)
    candidates = [
        _eval(world, scenarios[name], "target", 0.0, name, label, store_rho=False, with_covers=True)
        for name, label in CANDIDATES
    ]
    full = _eval(world, base, "full_oracle", 1.0, "full_oracle", "full oracle", store_rho=False)
    return CU22Result(
        n_episodes=len(deps), horizon=config.horizon, uniform=uniform, model=model,
        candidates=candidates, full_oracle=full, oracle_free=_oracle_free(world, base),
    )


def _candidate(result: CU22Result, name: str) -> CU22Cell:
    return next(c for c in result.candidates if c.name == name)


def _prediction_correct(cell: CU22Cell) -> bool:
    """Did the a-priori ``covers`` prediction match the measured adversarial safety?

    covers True  => the covering target is adversarially safe (breach ~0);
    covers False => the non-covering target leaks (breach > 0). The equivalence is the
    un-gameability theorem instantiated; CU22 confirms it holds on a danger the framework never saw.
    """
    if cell.covers is None:
        return True
    safe = cell.adversarial_breach <= 1e-9
    return cell.covers == safe


def cu22_verdict(result: CU22Result) -> dict[str, object]:
    """H115: on an unseen danger (availability), ``covers`` predicts every target's fate a priori
    and the run confirms it -- the framework is generative, not a post-hoc unification.
    """
    free = result.uniform[0]
    full = result.full_oracle
    closure = _candidate(result, "closure")
    connect = _candidate(result, "connect")
    exposure = _candidate(result, "exposure")
    knee = min(
        (c for c in result.uniform if c.rho is not None and 0.0 < c.rho < 1.0),
        key=lambda c: c.random_breach, default=result.uniform[0],
    )
    saving = full.mean_calls / closure.mean_calls if closure.mean_calls > 0 else float("inf")
    carried = [connect, exposure]
    return {
        "n_episodes": result.n_episodes,
        "free_breach_rate": free.random_breach,
        # the framework-DERIVED covering target: the oracle's safety, un-gameable, cheap
        "closure_covers": closure.covers,
        "closure_random_breach": closure.random_breach,
        "closure_adversarial_breach": closure.adversarial_breach,
        "closure_calls": closure.mean_calls,
        "closure_is_safe": closure.random_breach <= full.random_breach + 1e-9,
        "closure_is_ungameable": closure.adversarial_breach <= 1e-9,
        "closure_cheaper_than_full": closure.mean_calls < full.mean_calls,
        "closure_call_saving": saving,
        "full_oracle_calls": full.mean_calls,
        # the carried-over catalogue (CU10 + CU17): coverage broken -> predicted + measured to leak
        "carried_over_all_break_coverage": all(c.covers is False for c in carried),
        "carried_over_all_leak": all(c.adversarial_breach > 1e-9 for c in carried),
        "connect_covers": connect.covers,
        "connect_adversarial_breach": connect.adversarial_breach,
        "exposure_covers": exposure.covers,
        "exposure_adversarial_breach": exposure.adversarial_breach,
        # the uniform knee is a mirage against an adversary who picks the disconnection timing
        "uniform_knee_rho": knee.rho,
        "uniform_knee_random_breach": knee.random_breach,
        "uniform_knee_adversarial_breach": knee.adversarial_breach,
        "uniform_is_gameable": knee.adversarial_breach > knee.random_breach + 1e-9,
        # the model self-targets blindly (the omitter never foresees its own outages)
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


def write_csv(result: CU22Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]

    def _row(kind: str, c: CU22Cell) -> str:
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
