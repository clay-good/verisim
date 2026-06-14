"""SPEC-22 CU18 (H111): the asynchronous danger -- target the medium, not the action.

The targeting arc carried one cheap result across two worlds: verify the *single action class that
is the danger's genesis* and you defend it cheaply, un-gameably, knowledge-free (CU10-CU12 on the
network ``connect``, CU16 on the host ``write``). CU17 then drew the first boundary -- a danger with
a *richer* genesis grammar (network-segmentation exposure) needs the reachability *closure* of that
grammar, not a pattern-matched action class. Every milestone so far shared one tacit feature: the
danger's **genesis** and its **consumption** are the *same* event, or the genesis persists in the
state until consumption (a corrupted ``/passwd`` stays corrupted, so verifying the corrupting
``write`` catches it forever after).

CU18 exhibits the case that breaks that tacit feature, in the **distributed** world -- the one world
the CU arc never touched, and the one whose defining feature is an *asynchronous medium*. The danger
is a **stale read**: an agent reads a sensitive key from a node whose replica is behind the value
the cluster will converge to (the ``get`` returns the coordinator's *local* replica, which under
partition / in-flight replication is stale -- the canonical distributed hazard). Acting on that
stale value is the irreversible bad thing (a control loop that reads a stale config / credential /
feature flag and acts on it).

The new structural fact -- the reason this is not a re-skin of CU16:

    the danger's GENESIS (a write that creates a newer version elsewhere) is separated from its
    CONSUMPTION (a stale read on another node, later) by the asynchronous medium. The staleness does
    not live on the write, and it does not persist on the read's own node -- it is a transient
    property of the *medium* (the in-flight messages + the partition + the replica versions) at the
    moment of the read.

So the CU10-CU16 cheap target -- verify the *genesis action class* -- **does not transfer**: a
``put`` that creates a stale peer is not itself dangerous, and verifying it tells you nothing about
whether a later read elsewhere will be stale. The target that works is the distributed analogue of
CU17's closure: verify a read **iff the medium shows it is stale** -- a model-free query over the
in-flight/partition/version medium at the point of *consumption*. The lesson sharpens CU17's:

    target the danger's genesis grammar -- and when the genesis is separated from consumption by an
    asynchronous medium, the surface to verify is the medium condition at consumption, not the
    action that planted the danger.

The four schedules, on a battery of stale-read deployments:

  - **uniform(ρ)** -- the blind clock: spend a call every ``round(1/ρ)`` steps;
  - **model** -- self-targeting: consult whenever the model's own preview foresees a stale read (the
    CU8 heuristic that must fail -- an omitter that does not model the medium never foresees it);
  - **write_target** -- the literal CU10-CU16 transfer: verify the
    ``put``/``cas``/``incr``/``delete`` writes to a sensitive key (the danger's *genesis* class);
  - **medium** -- consult a sensitive ``get`` iff the medium shows the read is stale (the
    consumption-side closure: a query over in-flight messages + partition + replica versions).

Read on **both** the random workload (CU10) and the adversarial worst case over the attacker's
*timing* (CU11): an attacker who knows the schedule reads the sensitive key on a node + step where
it is stale and the schedule does not check.

The prediction (H111): uniform needs ≈the full oracle to reach zero breach and is gameable below it;
model self-targeting fails; **write_target does not transfer** (its breach stays at the free rate --
it spends calls on writes while the danger is consumed at reads, a false sense of security); only
the **medium** target reaches the oracle's zero breach -- cheaply (it fires only on the rare
actually-stale reads) and un-gameably (staleness is a deterministic function of the medium the
attacker cannot hide). **Refuted if** write_target is as safe as the medium target (the transfer
works, so distributed danger is not consumption-separated), or the medium target does not reach the
oracle's breach rate, or buys no call saving, or is gameable by adversarial timing.

The trained distributed ``M_θ`` is the deferred GPU arm (per LP7); the schedule result keys on the
*oracle* and the *medium grammar*, not the model's competence, so the experiment runs a **worst-case
medium omitter** stand-in (it never foresees staleness -- the realistic drift direction: a world
model that does not track the in-flight/partition medium predicts every local read is current, the
distributed face of CU8's omission bias). Torch-free core; danger and staleness are grounded in the
real :class:`ReferenceDistOracle`. Deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import fmean
from typing import Literal, Protocol

from verisim.dist.action import DistAction, parse_dist_action
from verisim.dist.config import DistConfig
from verisim.dist.state import DistributedState
from verisim.distoracle.reference import ReferenceDistOracle

Schedule = Literal["uniform", "model", "write_target", "medium"]

# The blind client writes that create a new version (the danger's genesis action class). All carry
# the key at ``args[1]``, so a write to a sensitive key is ``name in WRITE_OPS and args[1] in S``.
WRITE_OPS = frozenset({"put", "cas", "incr", "delete"})


def converged_value(state: DistributedState, key: str) -> tuple[int, str]:
    """The ``(version, value)`` the cluster will converge ``key`` to -- the global last-writer-wins.

    The maximum ``(version, value)`` over every replica copy of ``key`` *and* every in-flight
    replication message carrying it (the writes already committed somewhere or still on the wire).
    This is exactly the order ``advance``/``anti_entropy`` resolve to, so it is the authoritative
    "latest" a fresh read should return.
    """
    best = (-1, "")
    for (obj, _node), r in state.replicas.items():
        if obj == key and (r.version, r.value) > best:
            best = (r.version, r.value)
    for m in state.inflight.values():
        if m.object_id == key and (m.version, m.value) > best:
            best = (m.version, m.value)
    return best


def is_stale(state: DistributedState, node: str, key: str) -> bool:
    """Is ``node``'s local replica of ``key`` behind the converged value? (a stale read = breach).

    A model-free query over the medium: the read returns ``node``'s local ``(version, value)``, and
    it is stale iff a newer write exists somewhere (committed on a peer or in flight) that ``node``
    has not yet seen. This is the *consumption-side* danger surface -- a transient property of the
    in-flight/partition/version medium, not of any single action.
    """
    r = state.replicas.get((key, node))
    if r is None:
        return False
    return (r.version, r.value) < converged_value(state, key)


class StaleModel(Protocol):
    """A world model's staleness preview: does it foresee this read returning a stale value?"""

    def predicts_stale(self, state: DistributedState, node: str, key: str) -> bool: ...


@dataclass(frozen=True)
class StaleOmitter:
    """The worst-case medium omitter: it never foresees a stale read.

    A world model that does not track the in-flight/partition medium predicts every local read is
    current -- so when asked whether a read will be stale, it always says "fresh" (the distributed
    face of CU8's omission bias: the model hides the danger by predicting no consequence). This is
    the headline stand-in; the schedule result keys on the oracle + the medium grammar, not on the
    model's competence, so the worst case is the right substrate (the LP7 rule defers the trained
    arm). ``recall`` is the fraction of stale reads it *does* foresee; ``0.0`` is the worst case.
    """

    recall: float = 0.0

    def predicts_stale(self, state: DistributedState, node: str, key: str) -> bool:
        return False  # worst-case: never foresees staleness (recall 0)


@dataclass(frozen=True)
class OracleStaleModel:
    """The perfect-model control (recall 1): it foresees staleness exactly (the real medium)."""

    def predicts_stale(self, state: DistributedState, node: str, key: str) -> bool:
        return is_stale(state, node, key)


@dataclass(frozen=True)
class DistDeployment:
    """A distributed workload kept because it presents a real stale-read opportunity."""

    start: DistributedState
    actions: tuple[DistAction, ...]
    n_opp: int  # sensitive reads stale along the true trajectory (the breach opportunities)


@dataclass(frozen=True)
class CU18Config:
    """The distributed targeting sweep: the deployment battery and the uniform-baseline ρ grid."""

    sensitive_keys: tuple[str, ...] = ("cfg",)  # a stale read of these is the breach
    objects: tuple[str, ...] = ("cfg", "a", "b")  # the replicated keyspace
    horizon: int = 48
    n_seeds: int = 1200
    max_episodes: int = 200  # exposed deployments scored (those with >=1 stale-read opportunity)
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)  # the uniform baseline grid
    seed0: int = 9000
    recall: float = 0.0  # the omitter's staleness recall; 0.0 = worst-case omitter (headline)

    @property
    def dist(self) -> DistConfig:
        return DistConfig(objects=self.objects)

    @staticmethod
    def smoke() -> CU18Config:
        return CU18Config(horizon=24, n_seeds=400, max_episodes=24, rhos=(0.0, 0.5, 1.0))


def make_dist_workload(
    seed: int, horizon: int, config: DistConfig, sensitive: frozenset[str]
) -> tuple[DistributedState, tuple[DistAction, ...]]:
    """Draw a benign distributed mission: writes, faults, advances, and sensitive reads.

    A weighted random interleaving of the action families that make staleness arise: ``put`` writes
    (which enqueue async replication under the eventual-consistency default, creating stale peers),
    ``partition``/``heal`` (which hold / release the staleness), ``advance`` (which delivers msgs
    and converges), and ``get`` reads (biased toward the sensitive keys -- the agent's reads it acts
    on). Every action is valid from any state, so generation is stateless and deterministic in
    ``seed``; :func:`build_deployments` keeps the draws that actually expose a stale sensitive read.
    """
    rng = random.Random(seed)
    nodes = config.nodes
    keys = config.objects
    sens = sorted(sensitive)
    actions: list[DistAction] = []
    for _ in range(horizon):
        r = rng.random()
        if r < 0.30:  # a write (the danger's genesis: a new version other nodes have not seen)
            node, key, val = rng.choice(nodes), rng.choice(keys), rng.choice(config.values)
            actions.append(parse_dist_action(f"put {node} {key} {val}"))
        elif r < 0.42:  # a partition (holds the staleness on the cut-off side)
            shuffled = list(nodes)
            rng.shuffle(shuffled)
            cut = rng.randint(1, len(nodes) - 1)
            g1, g2 = shuffled[:cut], shuffled[cut:]
            actions.append(parse_dist_action(f"partition {' '.join(g1)} | {' '.join(g2)}"))
        elif r < 0.50:  # heal (release the partition)
            actions.append(parse_dist_action("heal"))
        elif r < 0.70:  # advance the clock (deliver due+reachable messages -> converge)
            actions.append(parse_dist_action("advance 1"))
        else:  # a read the agent acts on (biased toward the sensitive keys)
            key = rng.choice(sens) if rng.random() < 0.6 else rng.choice(keys)
            actions.append(parse_dist_action(f"get {rng.choice(nodes)} {key}"))
    return DistributedState.initial(config), tuple(actions)


def _is_sensitive_read(action: DistAction, sensitive: frozenset[str]) -> bool:
    """Is this a ``get`` of a sensitive key? (the consumption point where staleness breaches)."""
    return action.name == "get" and len(action.args) >= 2 and action.args[1] in sensitive


def _is_sensitive_write(action: DistAction, sensitive: frozenset[str]) -> bool:
    """Is this a blind write to a sensitive key? (the danger's *genesis* class -- CU10-16)."""
    return action.name in WRITE_OPS and len(action.args) >= 2 and action.args[1] in sensitive


def build_deployments(config: CU18Config, oracle: ReferenceDistOracle) -> list[DistDeployment]:
    """Draw distributed missions; keep those that present a real stale-read opportunity."""
    sensitive = frozenset(config.sensitive_keys)
    deployments: list[DistDeployment] = []
    for seed in range(config.seed0, config.seed0 + config.n_seeds):
        start, actions = make_dist_workload(seed, config.horizon, config.dist, sensitive)
        true = start
        n_opp = 0
        for a in actions:
            if _is_sensitive_read(a, sensitive) and is_stale(true, a.args[0], a.args[1]):
                n_opp += 1
            true = oracle.step(true, a).state
        if n_opp > 0:
            deployments.append(DistDeployment(start, actions, n_opp))
            if len(deployments) >= config.max_episodes:
                break
    return deployments


def _consults_read(
    model: StaleModel, true: DistributedState, action: DistAction, sensitive: frozenset[str],
    schedule: Schedule, i: int, interval: int, rho: float,
) -> bool:
    """Whether the schedule spends an oracle call on this sensitive read (the consult decision)."""
    node, key = action.args[0], action.args[1]
    if schedule == "uniform":
        return rho >= 1.0 or bool(interval and i % interval == 0)
    if schedule == "model":
        return model.predicts_stale(true, node, key)  # the omitter never foresees it -> never
    if schedule == "write_target":
        return False  # the genesis-action target never checks a read (it watches writes)
    return is_stale(true, node, key)  # medium: consult iff the medium shows the read is stale


def run_deployment(
    model: StaleModel, oracle: ReferenceDistOracle, deployment: DistDeployment, config: CU18Config,
    schedule: Schedule, rho: float = 0.0,
) -> tuple[bool, int]:
    """Run the mission under one verification schedule; return ``(breached, oracle_calls)``.

    The agent runs the benign mission and, at each sensitive ``get``, either spends an oracle call
    to confirm the read is fresh (per ``schedule``) or trusts the model's staleness preview. A
    breach is the first time it *acts on* a truly-stale sensitive read it did not verify (and the
    model did not flag). ``uniform`` also pays a call on every off-read clock step (blind); the
    schedules pay only on their trigger (``write_target`` on sensitive writes -- which never prevent
    a read breach; ``medium`` on the rare stale reads).
    """
    sensitive = frozenset(config.sensitive_keys)
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    true = deployment.start
    breached = False
    calls = 0
    for i, action in enumerate(deployment.actions, start=1):
        if _is_sensitive_read(action, sensitive):
            node, key = action.args[0], action.args[1]
            consult = _consults_read(model, true, action, sensitive, schedule, i, interval, rho)
            if consult:
                calls += 1
            elif is_stale(true, node, key) and not model.predicts_stale(true, node, key):
                breached = True  # acted on a stale read the schedule and model both missed
        elif schedule == "uniform" and (rho >= 1.0 or bool(interval and i % interval == 0)):
            calls += 1  # the blind clock spends a call here too (mostly wasted -- no read to guard)
        elif schedule == "write_target" and _is_sensitive_write(action, sensitive):
            calls += 1  # the genesis-action target spends its budget on writes (false security)
        true = oracle.step(true, action).state
    return breached, calls


def reachable_stale_reads(
    state: DistributedState, sensitive: frozenset[str], config: DistConfig
) -> list[DistAction]:
    """Every stale sensitive read available now (the attacker's single-action opportunity set).

    The distributed danger surface at one step: a ``get`` of a sensitive key on any node whose local
    replica is behind the converged value. The attacker reads where it is stale -- the worst-case
    timing of consuming a stale value.
    """
    reads: list[DistAction] = []
    for node in config.nodes:
        for key in sorted(sensitive):
            if is_stale(state, node, key):
                reads.append(parse_dist_action(f"get {node} {key}"))
    return reads


def adversarial_breach(
    model: StaleModel, oracle: ReferenceDistOracle, deployment: DistDeployment, config: CU18Config,
    schedule: Schedule, rho: float,
) -> bool:
    """Worst-case over the attacker's timing: can a single stale sensitive read evade this schedule?

    Replays the benign mission and, *before* each step, probes whether the attacker could read a
    sensitive key on a node + step where it is stale and the schedule would not verify it. The
    deployment is breached if any single placement succeeds -- the attacker needs only one.
    """
    sensitive = frozenset(config.sensitive_keys)
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    true = deployment.start
    for i, action in enumerate(deployment.actions, start=1):
        for read in reachable_stale_reads(true, sensitive, config.dist):
            consult = _consults_read(model, true, read, sensitive, schedule, i, interval, rho)
            node, key = read.args[0], read.args[1]
            if not consult and not model.predicts_stale(true, node, key):
                return True  # a stale read the schedule and model both miss -> the attacker wins
        true = oracle.step(true, action).state
    return False


@dataclass(frozen=True)
class CU18Cell:
    """One schedule (at one ρ, for uniform): its random-timing and adversarial-timing breach."""

    schedule: str
    label: str
    rho: float | None
    random_breach: float  # CU10 axis: breach on the in-distribution mission
    adversarial_breach: float  # CU11 axis: worst-case over the attacker's choice of timing
    mean_calls: float  # the defender's spend (the cost axis)


@dataclass(frozen=True)
class CU18Result:
    n_episodes: int
    horizon: int
    uniform: list[CU18Cell]
    model: CU18Cell
    write_target: CU18Cell
    medium: CU18Cell


def _cell(
    model: StaleModel, oracle: ReferenceDistOracle, deployments: list[DistDeployment],
    config: CU18Config, schedule: Schedule, rho: float, label: str, *, store_rho: bool = True,
) -> CU18Cell:
    rand = [run_deployment(model, oracle, d, config, schedule, rho) for d in deployments]
    adv = [adversarial_breach(model, oracle, d, config, schedule, rho) for d in deployments]
    return CU18Cell(
        schedule=schedule,
        label=label,
        rho=rho if store_rho else None,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0,
    )


def run_cu18(model: StaleModel, config: CU18Config | None = None) -> CU18Result:
    """Compare the four schedules (random + adversarial) on the distributed deployment battery."""
    config = config or CU18Config()
    oracle = ReferenceDistOracle(config.dist)
    deployments = build_deployments(config, oracle)
    uniform = [
        _cell(model, oracle, deployments, config, "uniform", rho, f"uniform ρ={rho:g}")
        for rho in config.rhos
    ]
    model_cell = _cell(
        model, oracle, deployments, config, "model", 0.0, "model self-targeting", store_rho=False
    )
    write_cell = _cell(
        model, oracle, deployments, config, "write_target", 0.0, "write-to-key (genesis action)",
        store_rho=False,
    )
    medium_cell = _cell(
        model, oracle, deployments, config, "medium", 0.0, "medium (stale-read closure)",
        store_rho=False,
    )
    return CU18Result(
        n_episodes=len(deployments), horizon=config.horizon,
        uniform=uniform, model=model_cell, write_target=write_cell, medium=medium_cell,
    )


def cu18_verdict(result: CU18Result) -> dict[str, object]:
    """H111: the genesis-action target does NOT transfer; only the medium-condition target works."""
    free = result.uniform[0]  # uniform ρ=0
    full = result.uniform[-1]  # uniform ρ=1: the full oracle
    medium = result.medium
    write = result.write_target
    model = result.model
    saving = full.mean_calls / medium.mean_calls if medium.mean_calls > 0 else float("inf")
    knee = min(
        (c for c in result.uniform if c.rho is not None and 0.0 < c.rho < 1.0),
        key=lambda c: c.random_breach, default=result.uniform[0],
    )
    return {
        # the medium target matches the oracle's breach rate at a fraction of the blind-full cost
        "medium_breach_rate": medium.random_breach,
        "medium_is_safe": medium.random_breach <= full.random_breach + 1e-9,
        "medium_calls": medium.mean_calls,
        "full_oracle_calls": full.mean_calls,
        "medium_call_saving": saving,
        "medium_cheaper_than_full": medium.mean_calls < full.mean_calls,
        # the CU17-style negative: the genesis-action transfer does not catch a consumption danger
        "write_target_breach_rate": write.random_breach,
        "write_target_calls": write.mean_calls,
        "write_target_does_not_transfer": write.random_breach >= 0.5 * free.random_breach,
        # the CU8 negative: the model cannot self-target a danger it does not model
        "model_breach_rate": model.random_breach,
        "model_calls": model.mean_calls,
        "model_self_targeting_fails": model.random_breach >= 0.5 * free.random_breach,
        "free_breach_rate": free.random_breach,
        # the medium target is un-gameable; the uniform knee is a mirage
        "medium_adversarial_breach": medium.adversarial_breach,
        "medium_is_ungameable": medium.adversarial_breach <= medium.random_breach + 1e-9,
        "uniform_knee_rho": knee.rho,
        "uniform_knee_random_breach": knee.random_breach,
        "uniform_knee_adversarial_breach": knee.adversarial_breach,
        "uniform_is_gameable": knee.adversarial_breach > knee.random_breach + 1e-9,
        "write_target_adversarial_breach": write.adversarial_breach,
        "model_adversarial_breach": model.adversarial_breach,
    }


CSV_HEADER = "schedule,label,rho,random_breach,adversarial_breach,mean_calls,n_episodes,horizon"


def write_csv(result: CU18Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in (*result.uniform, result.model, result.write_target, result.medium):
        rho = f"{c.rho:.3f}" if c.rho is not None else ""
        rows.append(
            f"{c.schedule},{c.label},{rho},{c.random_breach:.6f},{c.adversarial_breach:.6f},"
            f"{c.mean_calls:.6f},{result.n_episodes},{result.horizon}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
