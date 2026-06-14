"""SPEC-22 CU19 (H112): the trained distributed arm -- does a real learned M_θ track the medium?

CU18 carried the targeting result to the distributed world on a worst-case medium-omitter stand-in
(``StaleOmitter``, recall 0): only the medium-condition target catches the stale-read danger. The
stand-in was justified (LP7 defers the trained arm; the schedule keys on the oracle + the medium
grammar, not the model's competence). CU19 closes the rigor gap exactly as CU5-net/CU8 closed it for
the network world -- it runs the distributed closed loop on the **real trained distributed M_θ** and
asks the two questions a real learned model lets you ask:

  1. (the CU8 analogue -- the drift asymmetry) does the real model's drift *track* the asynchronous
     medium, or is it **omission-biased**? Rolled forward as a belief, does a flat learned world
     model foresee a stale read, or does it predict every local read is current -- the distributed
     face of CU8's omission bias, which would validate the ``StaleOmitter`` substrate empirically?
  2. (the CU5-net analogue -- the closed loop) does the medium target still close on the real model
     -- model self-targeting fail, ``write_target`` not transfer, the uniform knee stay a mirage,
     and only the **medium** target reach zero breach cheaply and un-gameably?

**The model's role -- the belief rollout.** CU18's :class:`~verisim.acd.dist_targeting.StaleModel`
protocol asks ``predicts_stale(state, node, key)``. A real M_θ is a ``predict_delta`` model, not a
staleness oracle, so CU19 derives its staleness preview the honest way a deployed agent would: a
**belief rollout**. The agent maintains a believed cluster state and advances it by applying the
model's *own* predicted delta at each step (``belief = apply(belief, M_θ.predict_delta(belief,
action))``), and answers "is this read stale?" by the model-free medium query on its **belief**
(``is_stale(belief, node, key)``). This is the exact distributed analogue of CU5-net's believed-flow
rollout, and it makes the model's drift structure *emergent* rather than assumed: a belief that
never tracks the medium foresees no staleness (the ``StaleOmitter``), while a belief that drifts the
*other* way -- its replicas falling out of sync with truth over a free-running rollout --
over-predicts staleness (hallucination, the asymmetry the real trained arm actually exhibits). The
two CU18 stand-ins are the recall endpoints of this rollout: a **no-op**
delta model (belief frozen at the boot cluster, where nothing is stale) reproduces ``StaleOmitter``
(recall 0) and an **oracle** delta model (belief == truth) reproduces ``OracleStaleModel`` (recall
1) -- a property the tests assert, bridging CU19 to CU18.

**Why belief-rollout, not teacher-forced.** Staleness is not a one-step property: it is a property
of the medium's accumulated history (a ``put`` enqueues a message, a ``partition`` holds it, an
``advance`` elsewhere delivers a different one). Teacher-forcing from the true state each step
resets the medium and erases the very dynamics under test, so the faithful probe must let the model
accumulate its own belief -- itself a finding about why distributed danger is hard to foresee.

**Cost.** The belief rollout is the only torch cost and it is independent of the schedule, so CU19
traces each deployment once (``horizon`` decodes) and evaluates all four schedules + both timings +
the drift probe off the cached trace -- the trained arm stays tractable on CPU (~``n_episodes ×
horizon`` decodes total). Torch-free core: takes any ``predict_delta`` model (the trained M_θ is
torch-gated in the experiment; stand-in delta models drive the tests). Danger and staleness are
grounded in the real :class:`~verisim.distoracle.reference.ReferenceDistOracle`. Deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Protocol

from verisim.acd.dist_targeting import (
    WRITE_OPS,
    CU18Cell,
    CU18Config,
    DistDeployment,
    Schedule,
    build_deployments,
    is_stale,
    reachable_stale_reads,
)
from verisim.dist.action import DistAction
from verisim.dist.delta import DistDelta, apply
from verisim.dist.state import DistributedState
from verisim.distoracle.reference import ReferenceDistOracle


class DeltaModel(Protocol):
    """A world model that predicts an action's state edit (the M_θ loop interface)."""

    def predict_delta(self, state: DistributedState, action: DistAction) -> DistDelta: ...


def _is_sensitive_read(action: DistAction, sensitive: frozenset[str]) -> bool:
    """Is this a ``get`` of a sensitive key? (the consumption point where staleness breaches)."""
    return action.name == "get" and len(action.args) >= 2 and action.args[1] in sensitive


def _is_sensitive_write(action: DistAction, sensitive: frozenset[str]) -> bool:
    """Is this a blind write to a sensitive key? (the danger's *genesis* class -- CU10-16)."""
    return action.name in WRITE_OPS and len(action.args) >= 2 and action.args[1] in sensitive


@dataclass(frozen=True)
class Trace:
    """One deployment rolled through both the oracle (truth) and the model (belief).

    ``true_states`` / ``belief_states`` are the states *before* each action (so a read's staleness
    is read off the state it sees), aligned with ``actions``. The model's staleness preview at step
    ``i`` is ``is_stale(belief_states[i], node, key)`` -- the belief-rollout's verdict.
    """

    true_states: tuple[DistributedState, ...]
    belief_states: tuple[DistributedState, ...]
    actions: tuple[DistAction, ...]


def trace_deployment(
    model: DeltaModel, oracle: ReferenceDistOracle, deployment: DistDeployment
) -> Trace:
    """Roll the deployment once through the oracle and the model's belief (the only torch cost)."""
    true = deployment.start
    belief = deployment.start
    true_states: list[DistributedState] = []
    belief_states: list[DistributedState] = []
    for action in deployment.actions:
        true_states.append(true)
        belief_states.append(belief)
        belief = apply(belief, model.predict_delta(belief, action))
        true = oracle.step(true, action).state
    return Trace(tuple(true_states), tuple(belief_states), deployment.actions)


def _consults(
    schedule: Schedule, i: int, interval: int, rho: float, model_flags: bool,
    true: DistributedState, node: str, key: str,
) -> bool:
    """Whether the schedule spends an oracle call on this sensitive read (mirrors CU18)."""
    if schedule == "uniform":
        return rho >= 1.0 or bool(interval and i % interval == 0)
    if schedule == "model":
        return model_flags  # the belief-rollout's staleness preview (omitter: never)
    if schedule == "write_target":
        return False  # the genesis-action target never checks a read (it watches writes)
    return is_stale(true, node, key)  # medium: consult iff the medium shows the read is stale


def breach_calls(
    trace: Trace, config: CU18Config, schedule: Schedule, rho: float
) -> tuple[bool, int]:
    """Replay the cached trace under one schedule; return ``(breached, oracle_calls)``.

    A breach is acting on a truly-stale sensitive read the schedule did not consult and the model's
    belief did not flag (``is_stale(true) and not is_stale(belief)``). ``uniform`` also pays on its
    off-read clock steps; ``write_target`` pays on sensitive writes (which never guard a read).
    """
    sensitive = frozenset(config.sensitive_keys)
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    breached = False
    calls = 0
    for i, (true, belief, action) in enumerate(
        zip(trace.true_states, trace.belief_states, trace.actions, strict=True), start=1
    ):
        if _is_sensitive_read(action, sensitive):
            node, key = action.args[0], action.args[1]
            model_flags = is_stale(belief, node, key)
            if _consults(schedule, i, interval, rho, model_flags, true, node, key):
                calls += 1
            elif is_stale(true, node, key) and not model_flags:
                breached = True
        elif (schedule == "uniform" and (rho >= 1.0 or bool(interval and i % interval == 0))) or (
            schedule == "write_target" and _is_sensitive_write(action, sensitive)
        ):
            calls += 1  # uniform's blind off-read clock; write_target's wasted write spend
    return breached, calls


def adversarial_breach(trace: Trace, config: CU18Config, schedule: Schedule, rho: float) -> bool:
    """Worst-case over the attacker's timing: can a single stale sensitive read evade the schedule?

    Before each step, probe whether the attacker could read a sensitive key on a node where it is
    stale and the schedule (and the model's belief) would not catch it (mirrors CU18's worst case).
    """
    sensitive = frozenset(config.sensitive_keys)
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    for i, (true, belief) in enumerate(
        zip(trace.true_states, trace.belief_states, strict=True), start=1
    ):
        for read in reachable_stale_reads(true, sensitive, config.dist):
            node, key = read.args[0], read.args[1]
            model_flags = is_stale(belief, node, key)
            if not _consults(schedule, i, interval, rho, model_flags, true, node, key) \
                    and not model_flags:
                return True
    return False


@dataclass(frozen=True)
class StalenessDrift:
    """The CU8 analogue on the medium: are belief errors omission- or hallucination-biased?

    Counted over every ``(node, sensitive_key)`` at every step of the true/belief traces: a
    **foreseen** stale read (true stale, belief stale) feeds recall; an **omission** (true stale,
    belief fresh) is the gate's missed-danger source; a **hallucination** (true fresh, belief stale)
    is the false-alarm source. The prediction (the distributed CU8): omissions ≫ hallucinations.
    """

    true_stale: int
    foreseen: int
    omissions: int
    hallucinations: int

    @property
    def recall(self) -> float:
        return self.foreseen / self.true_stale if self.true_stale else 1.0

    @property
    def precision(self) -> float:
        """Of the reads the belief calls stale, the fraction truly stale (1.0 if it never flags)."""
        flagged = self.foreseen + self.hallucinations
        return self.foreseen / flagged if flagged else 1.0

    @property
    def omission_ratio(self) -> float:
        """Omissions per hallucination (∞ if the model never hallucinates staleness)."""
        return self.omissions / self.hallucinations if self.hallucinations else float("inf")


def staleness_drift(traces: list[Trace], config: CU18Config) -> StalenessDrift:
    """Classify every belief-vs-truth staleness disagreement on the sensitive keys."""
    sensitive = sorted(config.sensitive_keys)
    nodes = config.dist.nodes
    true_stale = foreseen = omissions = hallucinations = 0
    for trace in traces:
        for true, belief in zip(trace.true_states, trace.belief_states, strict=True):
            for node in nodes:
                for key in sensitive:
                    t = is_stale(true, node, key)
                    b = is_stale(belief, node, key)
                    if t and b:
                        foreseen += 1
                        true_stale += 1
                    elif t and not b:
                        omissions += 1
                        true_stale += 1
                    elif b and not t:
                        hallucinations += 1
    return StalenessDrift(true_stale, foreseen, omissions, hallucinations)


@dataclass(frozen=True)
class CU19Result:
    n_episodes: int
    horizon: int
    drift: StalenessDrift
    uniform: list[CU18Cell]
    model: CU18Cell
    write_target: CU18Cell
    medium: CU18Cell


def _cell(
    traces: list[Trace], config: CU18Config, schedule: Schedule, rho: float, label: str,
    *, store_rho: bool = True,
) -> CU18Cell:
    rand = [breach_calls(t, config, schedule, rho) for t in traces]
    adv = [adversarial_breach(t, config, schedule, rho) for t in traces]
    return CU18Cell(
        schedule=schedule,
        label=label,
        rho=rho if store_rho else None,
        random_breach=fmean(b for b, _ in rand) if rand else 0.0,
        adversarial_breach=fmean(adv) if adv else 0.0,
        mean_calls=fmean(c for _, c in rand) if rand else 0.0,
    )


def run_cu19(model: DeltaModel, config: CU18Config | None = None) -> CU19Result:
    """Trace the battery once on the real model, then score all four schedules + the drift probe."""
    config = config or CU18Config()
    oracle = ReferenceDistOracle(config.dist)
    deployments = build_deployments(config, oracle)
    traces = [trace_deployment(model, oracle, d) for d in deployments]  # the only torch cost
    drift = staleness_drift(traces, config)
    uniform = [
        _cell(traces, config, "uniform", rho, f"uniform ρ={rho:g}") for rho in config.rhos
    ]
    return CU19Result(
        n_episodes=len(deployments),
        horizon=config.horizon,
        drift=drift,
        uniform=uniform,
        model=_cell(traces, config, "model", 0.0, "model self-targeting", store_rho=False),
        write_target=_cell(
            traces, config, "write_target", 0.0, "write-to-key (genesis action)", store_rho=False
        ),
        medium=_cell(traces, config, "medium", 0.0, "medium (stale-read closure)", store_rho=False),
    )


def cu19_verdict(result: CU19Result) -> dict[str, object]:
    """H112: the real M_θ omits the medium, so only the medium-condition target closes the loop."""
    free = result.uniform[0]
    full = result.uniform[-1]
    medium = result.medium
    write = result.write_target
    model = result.model
    d = result.drift
    saving = full.mean_calls / medium.mean_calls if medium.mean_calls > 0 else float("inf")
    return {
        # CU8 analogue: the real model's belief is omission-biased on the medium
        "staleness_recall": d.recall,
        "omissions": d.omissions,
        "hallucinations": d.hallucinations,
        "omission_ratio": d.omission_ratio,
        "drift_is_omission_biased": d.omissions >= d.hallucinations,
        # CU5-net analogue: the closed loop closes on the real model exactly as on the stand-in
        "free_breach_rate": free.random_breach,
        "medium_breach_rate": medium.random_breach,
        "medium_is_safe": medium.random_breach <= full.random_breach + 1e-9,
        "medium_calls": medium.mean_calls,
        "full_oracle_calls": full.mean_calls,
        "medium_call_saving": saving,
        "medium_cheaper_than_full": medium.mean_calls < full.mean_calls,
        "medium_adversarial_breach": medium.adversarial_breach,
        "medium_is_ungameable": medium.adversarial_breach <= medium.random_breach + 1e-9,
        "write_target_breach_rate": write.random_breach,
        "write_target_calls": write.mean_calls,
        "write_target_does_not_transfer": write.random_breach >= 0.5 * free.random_breach,
        "model_breach_rate": model.random_breach,
        "model_calls": model.mean_calls,
        "model_self_targeting_fails": model.random_breach >= 0.5 * free.random_breach,
    }


CSV_HEADER = "schedule,label,rho,random_breach,adversarial_breach,mean_calls,n_episodes,horizon"
DRIFT_HEADER = "true_stale,foreseen,omissions,hallucinations,recall"


def write_csv(result: CU19Result, path: str) -> str:
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
    rows.append("")
    rows.append(DRIFT_HEADER)
    d = result.drift
    rows.append(f"{d.true_stale},{d.foreseen},{d.omissions},{d.hallucinations},{d.recall:.6f}")
    out.write_text("\n".join(rows) + "\n")
    return str(out)
