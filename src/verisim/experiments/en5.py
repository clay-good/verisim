"""Experiment EN5: online test-time training (self-healing) on the H_ε(ρ) curve (SPEC-5 §12, H7).

EN1-EN7 hold the model's *weights fixed* during a rollout: the oracle corrects the model's drifted
*state* on each consultation, but the model never learns from the correction. EN5 asks the H7
question -- **does correction teach online?** When the loop consults the oracle mid-rollout, the
revealed ``(state, action) -> true-delta`` is a free, exactly-labeled example; an in-rollout step
(:func:`~verisim.netmodel.graph_train.online_update`, the TTT discipline of SPEC-3 §6 / HW-2) lets
the model *adapt to the current trajectory* so its *unaided* steps drift less afterward. This is
the one lever left that could lift the **model-invariant** `H_ε(ρ=0)` floor EN7 just showed every
architecture shares -- because it changes the model *during* the rollout, not just its state.

Two arms, both from the same supervised graph+RSSM arm, run through the **same** full-consult loop
(the EN1 semantics, replicated here so the only difference is the hook):

  - **supervised** -- weights frozen during the rollout (the EN1/EN7 baseline).
  - **+ttt** -- a fresh copy per rollout; each consultation also takes ``ttt_steps`` small gradient
    steps on the oracle-revealed delta, so later unaided steps benefit. *H7 supported* iff ``+ttt``
    lifts `H_ε(ρ)` above ``supervised`` at the same ρ (the correction taught, not just reset state).

*RLVR is deferred (pre-registered):* v0's E4 found REINFORCE on the faithful-horizon reward a clean
null because the reward is sparse *exactly* at the floor (episodes end at the first unfaithful step)
and the network arm still drifts on step 1 -- so RLVR has no horizon to amplify yet. EN5 leads with
the TTT arm, which needs no dense reward. CPU, deterministic.
"""

from __future__ import annotations

import argparse
import copy
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from verisim.loop.policy import StepContext, fixed_interval_for_rho
from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.action import NetAction
from verisim.net.config import scaled_net_config
from verisim.net.state import NetworkState
from verisim.netdelta import apply
from verisim.netloop import PartialNetOracle, budget_for_rho
from verisim.netloop.model import NetModel
from verisim.netloop.runner import ground_truth_rollout
from verisim.netmetrics.divergence import divergence
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetStepResult

from .en1 import eval_actions

ARMS = ("supervised", "+ttt", "+ttt-replay")


@dataclass(frozen=True)
class EN5Config:
    """Small, fast self-healing instance. Scale up (seeds/iters/world) for a publication run."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 48
    graph_mp_rounds: int = 3
    graph_iters: int = 1500
    lr: float = 3e-3
    model_seed: int = 0
    # online TTT (the discipline: few steps, small lr — SPEC-3 HW-2)
    ttt_steps: int = 3
    ttt_lr: float = 1e-3
    # the pre-registered self-healing budget: a replay buffer of recent corrections (SPEC-3 §6)
    replay_steps: int = 5  # minibatch updates per consult, sampled from the buffer
    replay_batch: int = 8
    # evaluation / sweep
    difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "weighted", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    rhos: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
    epsilon: float = 0.05


@dataclass(frozen=True)
class CurvePoint:
    """One (arm, ρ) cell: mean faithful horizon + bootstrap CI over difficulty x seed."""

    arm: str
    rho: float
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return f"{self.arm},{self.rho},{self.mean:.4f},{self.ci_lo:.4f},{self.ci_hi:.4f},{self.n}"


CSV_HEADER = "arm,rho,mean_horizon,ci_lo,ci_hi,n"


def _faithful_horizon(divergences: list[float], epsilon: float) -> int:
    """Steps the rollout stays within ε — first-exceedance horizon (mirrors EN1/EN4/EN7)."""
    for i, d in enumerate(divergences):
        if d > epsilon:
            return i
    return len(divergences)


def _rollout(
    model: NetModel,
    partial: PartialNetOracle,
    s0: NetworkState,
    actions: list[NetAction],
    rho: float,
    epsilon: float,
    *,
    on_consult: Callable[[NetworkState, NetAction, NetStepResult], None] | None,
) -> int:
    """One full-consultation propose-verify-correct rollout; returns the faithful horizon.

    Mirrors :func:`verisim.netloop.runner.run_net_rollout` (full-consult path) so the supervised arm
    is EN1-identical; the only addition is the optional ``on_consult`` hook, called with the exact
    one-step truth on each consultation so an arm can take a self-healing gradient step (TTT) on it.
    """
    n = len(actions)
    truth_states = ground_truth_rollout(partial, s0, actions)
    policy = fixed_interval_for_rho(rho)
    budget = budget_for_rho(rho, n)

    state = s0
    calls = 0
    divergences: list[float] = []
    for t, action in enumerate(actions):
        predicted = apply(state, model.predict_delta(state, action))
        has_budget = budget is None or calls < budget
        must_spend = budget is not None and (budget - calls) >= (n - t)
        consult = has_budget and (must_spend or policy.should_consult(StepContext(step=t)))
        if consult:
            calls += 1
            result = partial.full(state, action)  # VERIFY: the exact one-step truth
            if on_consult is not None:  # self-healing: teach the revealed (state, action) -> truth
                on_consult(state, action, result)
            state = result.state  # CORRECT (hard reset to truth)
        else:
            state = predicted
        divergences.append(divergence(truth_states[t + 1], state))
    return _faithful_horizon(divergences, epsilon)


def run_en5(config: EN5Config | None = None) -> list[CurvePoint]:
    """Train the base arm, then sweep H_ε(ρ) for the three arms; one point per (arm, ρ) cell."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph import build_graph
    from verisim.netmodel.graph_model import GraphRSSMWorldModel, build_graph_model
    from verisim.netmodel.graph_train import (
        GraphExample,
        build_graph_dataset,
        online_update,
        train_graph_model,
    )
    from verisim.netmodel.tokenizer import encode_target

    config = config or EN5Config()
    torch.set_num_threads(1)  # process-reproducibility (the EN1 discipline)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    partial = PartialNetOracle(oracle)

    base = build_graph_model(
        vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds,
        seed=config.model_seed,
    )
    examples = build_graph_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    train_graph_model(base, examples, steps=config.graph_iters, seed=config.model_seed)

    def _make_arm(
        arm: str, seed: int
    ) -> tuple[NetModel, Callable[[NetworkState, NetAction, NetStepResult], None] | None]:
        """Return ``(proposer, on_consult)`` for ``arm``; TTT arms get a fresh adapting copy."""
        if arm == "supervised":
            return base, None
        model: GraphRSSMWorldModel = copy.deepcopy(base)
        opt = torch.optim.AdamW(model.net.parameters(), lr=config.ttt_lr)
        if arm == "+ttt":  # minimal: one update on the single revealed example

            def single(s: NetworkState, a: NetAction, r: NetStepResult) -> None:
                ex = (build_graph(s, a, net), encode_target(r.delta, vocab))
                online_update(model, opt, [ex], steps=config.ttt_steps)

            return model, single
        # +ttt-replay: the pre-registered self-healing budget — a growing replay buffer of
        # corrections, several minibatch updates per consult (SPEC-3 §6; the H7-null next lever).
        buffer: list[GraphExample] = []
        rng = random.Random(seed)

        def replay(s: NetworkState, a: NetAction, r: NetStepResult) -> None:
            buffer.append((build_graph(s, a, net), encode_target(r.delta, vocab)))
            for _ in range(config.replay_steps):
                batch = rng.sample(buffer, min(config.replay_batch, len(buffer)))
                online_update(model, opt, batch, steps=1)

        return model, replay

    points: list[CurvePoint] = []
    for arm in ARMS:
        per_rho: dict[float, list[float]] = {rho: [] for rho in config.rhos}
        for _difficulty, driver in config.difficulties.items():
            for seed in config.eval_seeds:
                actions = eval_actions(oracle, net, driver, seed, config.eval_steps)
                for rho in config.rhos:
                    torch.manual_seed(seed)  # reproducible TTT sampling per rollout
                    model, on_consult = _make_arm(arm, seed)
                    horizon = _rollout(
                        model, partial, NetworkState.initial(net.hosts), actions,
                        rho, config.epsilon, on_consult=on_consult,
                    )
                    per_rho[rho].append(horizon)
        for rho in config.rhos:
            vals = per_rho[rho]
            lo, hi = bootstrap_ci(vals, seed=0)
            points.append(CurvePoint(arm, rho, fmean(vals), lo, hi, len(vals)))
    return points


def _print_summary(points: list[CurvePoint], config: EN5Config) -> None:
    print(f"EN5 self-healing: H_ε(ρ) by arm (ε={config.epsilon}, T={config.eval_steps}):")
    rhos = sorted({p.rho for p in points})
    print("  arm          " + "".join(f"ρ={r:<6}" for r in rhos))
    for arm in ARMS:
        row = {p.rho: p.mean for p in points if p.arm == arm}
        print(f"  {arm:<12} " + "".join(f"{row[r]:<8.1f}" for r in rhos))


def _plot(points: list[CurvePoint], path: Path, config: EN5Config) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    colors = {"supervised": "#9bd", "+ttt": "#c66", "+ttt-replay": "#16a"}
    for arm in ARMS:
        cells = sorted((p for p in points if p.arm == arm), key=lambda p: p.rho)
        xs = [p.rho for p in cells]
        ys = [p.mean for p in cells]
        lo = [p.ci_lo for p in cells]
        hi = [p.ci_hi for p in cells]
        (line,) = ax.plot(xs, ys, marker="o", label=arm, color=colors.get(arm))
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
    ax.set_xlabel("consultation budget ρ")
    ax.set_ylabel(f"faithful horizon H_ε (ε={config.epsilon}, ceiling T={config.eval_steps})")
    ax.set_title("EN5 / H7: does online self-healing (TTT) lift H_ε(ρ)? (95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN5 online self-healing / TTT on H_ε(ρ) (H7).")
    parser.add_argument("--n-hosts", type=int, default=5)
    parser.add_argument("--graph-iters", type=int, default=1500)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument("--ttt-lr", type=float, default=1e-3)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--out", type=str, default="figures/en5_selfheal.csv")
    args = parser.parse_args()
    cfg = EN5Config(
        n_hosts=args.n_hosts, graph_iters=args.graph_iters,
        ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr, eval_seeds=tuple(args.eval_seeds),
    )
    points = run_en5(cfg)
    _print_summary(points, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(p.csv_row() for p in points)]) + "\n")
    print(f"wrote {out}")
    _plot(points, out.with_suffix(".png"), cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
