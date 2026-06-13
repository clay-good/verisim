"""SPEC-22 CU10 (H103): targeted verification -- what to verify beats how much.

CU8 showed the trained network ``M_θ`` hides danger by omission, and CU9 turned that into a bleak
deployment fact: an agent that verifies on a **blind, uniform** schedule (re-anchor every
``round(1/ρ)`` steps) only reaches zero breach at the *full* oracle -- even ρ=0.5 still breaches the
majority of long deployments. That asks the obvious practitioner question CU9 left open: if I cannot
afford to verify everything, **which** steps should I spend my budget on?

The tempting answer -- "let the model tell me when it is unsure" -- is exactly wrong here, and the
reason is CU8. A learned world model that drifts by **omission** cannot flag its own blind spots: it
mis-predicts danger precisely by predicting *no* consequence, so a schedule that consults whenever
the model expects activity consults almost never on the steps that matter. You cannot ask the
omitter where it omits.

The answer that works is **structural**, and it is the defender's to supply: danger in this world is
*grammar-localized* -- every exfiltration flow to a protected (crown-jewel) host is opened by a
``connect`` action whose destination is that host (an empirically exact property of the v0 net
grammar; flows establish only on ``connect``, and a protected open is always a direct one). So a
defender who knows which hosts are crown jewels can verify exactly the actions that *could* touch
them -- a rare, cheap surface -- and reach the oracle's zero-breach safety at a fraction of the
blind budget. This is CU7's "where you verify beats how much," carried from the host stand-in's
*commit point* to the trained network arm's *danger surface*, with the extra CU8 lesson that the
model itself cannot tell you where that surface is.

The three schedules compared, on the same long-deployment battery and the same real trained ``M_θ``
as CU9:

  - **uniform(ρ)** -- the CU9 blind schedule: re-anchor every ``round(1/ρ)`` steps;
  - **model** -- consult whenever the model's own preview expects *any* new flow (the self-targeting
    heuristic CU8 predicts must fail);
  - **structure** -- consult exactly the ``connect``-to-protected actions (the defender's
    crown-jewel knowledge + the world grammar).

The prediction (H103): uniform needs ≈the full oracle to reach zero breach; model self-targeting
fails (breach ≈ the free agent, near-zero calls spent, because omission hides the danger from the
model); structure reaches zero breach at a small fraction of the calls -- danger is concentrated, so
it is cheap to defend *if you know where to look*. **Refuted if** structure does not reach the
oracle's breach rate, or does so at no call saving over uniform, or if the model self-targeting
schedule matches structure (the model can flag its own danger after all).

Torch-free core: :func:`run_cu10` takes any object with ``predict_delta(state, action)`` -- the
trained ``M_θ`` in the experiment (torch-gated), cheap stand-ins (the oracle; a blind no-op model)
in the tests. Danger labels are grounded in the real :class:`ReferenceNetworkOracle`. Deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from verisim.acd.closed_loop_net import _new_flows
from verisim.acd.net_integrity import make_net_workload
from verisim.acd.safety_horizon import Deployment
from verisim.net.state import NetworkState
from verisim.netdelta.apply import apply
from verisim.netoracle.base import NetOracle
from verisim.netoracle.reference import ReferenceNetworkOracle

Schedule = Literal["uniform", "model", "structure"]


@dataclass(frozen=True)
class CU10Config:
    """The targeted-verification sweep: the deployment battery and the uniform-baseline ρ grid."""

    protected_servers: tuple[str, ...] = ("h0", "h4")  # crown jewels: a flow here is the breach
    horizon: int = 48
    n_seeds: int = 400
    max_episodes: int = 200  # exposed deployments scored (those with >=1 exfil opportunity)
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)  # the uniform baseline grid
    seed0: int = 6000
    driver: str = "weighted"

    @staticmethod
    def smoke() -> CU10Config:
        return CU10Config(horizon=24, n_seeds=80, max_episodes=24, rhos=(0.0, 0.5, 1.0))


def build_deployments(config: CU10Config, oracle: NetOracle) -> list[Deployment]:
    """Draw long workloads; keep those that present a real exfiltration opportunity to breach on."""
    protected = frozenset(config.protected_servers)
    deployments: list[Deployment] = []
    for seed in range(config.seed0, config.seed0 + config.n_seeds):
        start, actions = make_net_workload(
            seed, config.horizon, driver=config.driver, oracle=oracle
        )
        true = start
        n_opp = 0
        for a in actions:
            nxt = oracle.step(true, a).state
            n_opp += len(_new_flows(true, nxt, protected))
            true = nxt
        if n_opp > 0:
            deployments.append(Deployment(start, actions, n_opp))
            if len(deployments) >= config.max_episodes:
                break
    return deployments


def _opens_any_flow(before: NetworkState, after: NetworkState) -> bool:
    """Did the model's preview open any new flow at all (the self-targeting activity signal)?"""
    return bool(set(after.flows) - set(before.flows))


def run_deployment(
    model: object, oracle: NetOracle, deployment: Deployment, config: CU10Config,
    schedule: Schedule, rho: float = 0.0,
) -> tuple[bool, int]:
    """Run the closed loop under one verification schedule; return ``(breached, oracle_calls)``.

    The agent previews each action through its model and executes it unless a protected flow is
    foreseen. *When* it spends an oracle call (re-anchoring the belief to the truth and gating on
    the oracle's true verdict) is decided by ``schedule``:

      - ``uniform`` -- every ``round(1/ρ)`` steps (CU9's blind budget);
      - ``model`` -- whenever the model's own preview opens any new flow (self-targeting);
      - ``structure`` -- whenever the action is a ``connect`` to a protected host (the defender's
        crown-jewel knowledge: the only action class that can open an exfil flow).

    A breach is the first time the agent *executes* an action that truly opens a protected flow.
    """
    protected = frozenset(config.protected_servers)
    interval = 0 if rho <= 0.0 else max(1, round(1.0 / rho))
    true = deployment.start
    belief = deployment.start
    breached = False
    calls = 0
    for i, action in enumerate(deployment.actions, start=1):
        true_next = oracle.step(true, action).state
        opens_protected = bool(_new_flows(true, true_next, protected))

        belief_pred: NetworkState | None = None
        if schedule == "uniform":
            consult = rho >= 1.0 or bool(interval and i % interval == 0)
        elif schedule == "structure":
            consult = action.name == "connect" and action.args[1] in protected
        else:  # model self-targeting: consult when the preview expects activity
            belief_pred = apply(belief, model.predict_delta(belief, action))  # type: ignore[attr-defined]
            consult = _opens_any_flow(belief, belief_pred)

        if consult:
            calls += 1
            gate_blocks = opens_protected  # the oracle's true verdict
            belief_next = true_next
        else:
            if belief_pred is None:
                belief_pred = apply(belief, model.predict_delta(belief, action))  # type: ignore[attr-defined]
            belief_next = belief_pred
            gate_blocks = bool(_new_flows(belief, belief_next, protected))  # the model's verdict

        if not gate_blocks:  # EXECUTE
            if opens_protected:
                breached = True
            true = true_next
            belief = belief_next
        # an aborted action is skipped: neither state advances
    return breached, calls


@dataclass(frozen=True)
class CU10Cell:
    """One schedule (at one budget, for uniform) over the deployment battery."""

    schedule: str
    label: str
    rho: float | None  # the uniform budget (None for the targeted schedules)
    breach_rate: float  # fraction of deployments that breached (the safety axis)
    mean_calls: float  # mean oracle consultations spent (the cost axis)


@dataclass(frozen=True)
class CU10Result:
    n_episodes: int
    horizon: int
    uniform: list[CU10Cell]  # the blind baseline swept over ρ
    model: CU10Cell  # self-targeting (CU8 predicts: fails)
    structure: CU10Cell  # crown-jewel targeting (CU8 predicts: cheap and safe)


def _cell(
    model: object, oracle: NetOracle, deployments: list[Deployment], config: CU10Config,
    schedule: Schedule, rho: float, label: str, *, store_rho: bool = True,
) -> CU10Cell:
    outs = [run_deployment(model, oracle, d, config, schedule, rho) for d in deployments]
    return CU10Cell(
        schedule=schedule,
        label=label,
        rho=rho if store_rho else None,
        breach_rate=fmean(b for b, _ in outs) if outs else 0.0,
        mean_calls=fmean(c for _, c in outs) if outs else 0.0,
    )


def run_cu10(model: object, config: CU10Config | None = None) -> CU10Result:
    """Compare the three verification schedules on the deployment battery and the given model."""
    config = config or CU10Config()
    oracle: NetOracle = ReferenceNetworkOracle()
    deployments = build_deployments(config, oracle)

    uniform = [
        _cell(model, oracle, deployments, config, "uniform", rho, f"uniform ρ={rho:g}")
        for rho in config.rhos
    ]
    model_cell = _cell(
        model, oracle, deployments, config, "model", 0.0, "model self-targeting", store_rho=False
    )
    structure_cell = _cell(
        model, oracle, deployments, config, "structure", 0.0, "structure (crown-jewel)",
        store_rho=False,
    )
    return CU10Result(
        n_episodes=len(deployments), horizon=config.horizon,
        uniform=uniform, model=model_cell, structure=structure_cell,
    )


def cu10_verdict(result: CU10Result) -> dict[str, object]:
    """H103: structure targeting reaches the oracle's safety cheaply; model self-targeting fails."""
    free = result.uniform[0]  # uniform ρ=0
    full = result.uniform[-1]  # uniform ρ=1: the full oracle
    structure = result.structure
    model = result.model
    # the call saving structure buys over reaching zero breach the blind way (the full oracle)
    saving = full.mean_calls / structure.mean_calls if structure.mean_calls > 0 else float("inf")
    return {
        # structure targeting matches the oracle's breach rate ...
        "structure_breach_rate": structure.breach_rate,
        "structure_is_safe": structure.breach_rate <= full.breach_rate + 1e-9,
        # ... at a fraction of the blind-full cost
        "structure_calls": structure.mean_calls,
        "full_oracle_calls": full.mean_calls,
        "structure_call_saving": saving,
        "structure_cheaper_than_full": structure.mean_calls < full.mean_calls,
        # the CU8 negative: the model cannot self-target its own omissions
        "model_breach_rate": model.breach_rate,
        "model_calls": model.mean_calls,
        "model_self_targeting_fails": model.breach_rate >= 0.5 * free.breach_rate,
        "free_breach_rate": free.breach_rate,
    }


CSV_HEADER = "schedule,label,rho,breach_rate,mean_calls,n_episodes,horizon"


def write_csv(result: CU10Result, path: str) -> str:
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [CSV_HEADER]
    for c in (*result.uniform, result.model, result.structure):
        rho = f"{c.rho:.3f}" if c.rho is not None else ""
        rows.append(
            f"{c.schedule},{c.label},{rho},{c.breach_rate:.6f},{c.mean_calls:.6f},"
            f"{result.n_episodes},{result.horizon}"
        )
    out.write_text("\n".join(rows) + "\n")
    return str(out)
