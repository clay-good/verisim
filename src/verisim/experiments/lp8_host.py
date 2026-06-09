"""Experiment LP8-host: the cross-world fork — privilege-landmark planning on hostsim (H38, §6).

The third world for the landmark method. LP8-dist showed the method transfers off the network to the
distributed world (consistency/partition projection). LP8-host asks the *harder* question, the one
H38 pre-registers as the refutation branch: does it transfer to a world with **no canonical coarse
hidden state**? The host world (SPEC-6: processes / fds / fs / exit) has no reachability or
partition
structure; its security-relevant projection is **privilege** (§3.2 — a non-root process gaining
root),
which we encode as the coarse **privilege/liveness class set**
([`landmark/host.py`](../landmark/host.py)) — the set of ``(process state, uid)`` classes present,
count-free.

Two readings, the LP2 and LP3 analogues (mirroring LP8-dist):

  - **the faithful privilege-graph (LP2 analogue).** Over privilege-changing transitions, the flat
    host `M_θ`'s *hoped* edges are graded against the oracle: false-edge rate, the verified residual
    (0.000 by construction), and the consult-cost ratio (privilege facts vs the full bit-exact set).
  - **goal reach: landmark planning vs flat free-running (LP3 analogue, the H38 headline).** Two
  arms
    on the *same* model and *same* trajectory, differing only in re-grounding: flat free-running
    (`ρ = 0`) vs landmark planning re-grounding at every intermediate privilege boundary (`ρ =
    1/L`).
    Goal reach = the model's predicted privilege signature at the distant goal matches the truth.

H38 supported here if landmark re-grounding lifts goal reach above flat. *Refuted/null if* it does
not — a bankable result: the host privilege projection is *slower-moving* than network reachability
or dist consistency (escalations and deaths are rarer than reachability or partition changes), so
the
model may already be faithful enough that re-grounding buys little. Either branch is a result about
which worlds' projections have the HS3-cliff structure that makes re-grounding necessary (SPEC
§10.1).
Reduced over (difficulty × seed) cells with bootstrap CIs. Torch-backed (the flat host `M_θ`, like
EH1); deterministic, seeded.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import HostConfig
from verisim.host.state import HostState
from verisim.hostloop.model import HostModel
from verisim.hostmetrics.divergence import host_facts
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.landmark.host import (
    execute_host_plan,
    privilege_facts,
    privilege_signature,
)
from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class LP8HostConfig:
    """A small, fast host goal-reach (H38) measurement instance."""

    train_driver: str = "forky"
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_steps_per_traj: int = 40
    train_iters: int = 700
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 256
    lr: float = 3e-3
    model_seed: int = 0
    eval_difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "uniform", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105)
    hop_length: int = 4
    goal_distances: tuple[int, ...] = (4, 8, 12, 16, 20)
    budget_hop_lengths: tuple[int, ...] = (2, 4, 8)
    budget_goal_distance: int = 16

    @staticmethod
    def from_dict(d: dict[str, Any]) -> LP8HostConfig:
        b = LP8HostConfig()
        return LP8HostConfig(
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            train_iters=d.get("train_iters", b.train_iters),
            n_layer=d.get("n_layer", b.n_layer),
            n_head=d.get("n_head", b.n_head),
            n_embd=d.get("n_embd", b.n_embd),
            block_size=d.get("block_size", b.block_size),
            lr=d.get("lr", b.lr),
            model_seed=d.get("model_seed", b.model_seed),
            eval_difficulties=d.get("eval_difficulties", b.eval_difficulties),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            hop_length=d.get("hop_length", b.hop_length),
            goal_distances=tuple(d.get("goal_distances", b.goal_distances)),
            budget_hop_lengths=tuple(d.get("budget_hop_lengths", b.budget_hop_lengths)),
            budget_goal_distance=d.get("budget_goal_distance", b.budget_goal_distance),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> LP8HostConfig:
        return LP8HostConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class LP8Stat:
    """Goal reach (+ privilege horizon) for one (sweep, arm, x) cell: mean + bootstrap CI."""

    sweep: str  # "distance" | "budget"
    arm: str  # "flat" | "landmark"
    x_value: float
    goal_reach: float
    gr_lo: float
    gr_hi: float
    priv_horizon: float
    rho: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.sweep},{self.arm},{self.x_value:.6f},{self.goal_reach:.6f},"
            f"{self.gr_lo:.6f},{self.gr_hi:.6f},{self.priv_horizon:.6f},{self.rho:.6f},{self.n}"
        )


CSV_HEADER = "sweep,arm,x_value,goal_reach,gr_lo,gr_hi,priv_horizon,rho,n"


@dataclass(frozen=True)
class Journey:
    """One ground-truth host rollout: the actions + the true state after each."""

    actions: tuple[HostAction, ...]
    truth: tuple[HostState, ...]
    start: HostState


def _roll_journey(oracle: HostOracle, host: HostConfig, driver: str, seed: int, n: int) -> Journey:
    """Roll a seeded driver ``n`` steps from the boot state; record actions + true states."""
    from verisim.hostdata.drivers import HostDriver

    drv = HostDriver(driver, host, random.Random(seed))
    start = HostState.initial()
    state = start
    actions: list[HostAction] = []
    truth: list[HostState] = []
    for _ in range(n):
        action = drv.sample(state)
        state = oracle.step(state, action).state
        actions.append(action)
        truth.append(state)
    return Journey(tuple(actions), tuple(truth), start)


def _reground_steps(goal_dist: int, hop_len: int) -> frozenset[int]:
    """The intermediate privilege-boundary step indices to re-ground at (the goal excluded)."""
    return frozenset(
        k * hop_len - 1
        for k in range(1, goal_dist // hop_len + 1)
        if k * hop_len - 1 < goal_dist - 1
    )


def _cell_goal_reach(
    model: HostModel, journey: Journey, goal_dist: int, hop_len: int
) -> tuple[dict[str, float], dict[str, float], float]:
    """Run both arms on one journey at (goal_dist, hop_len); return goal-reach, horizon, ρ."""
    reground_at = _reground_steps(goal_dist, hop_len)
    actions = journey.actions[:goal_dist]
    truth = journey.truth[:goal_dist]
    flat = execute_host_plan(model, journey.start, actions, truth, frozenset(), reground=False)
    landmark = execute_host_plan(model, journey.start, actions, truth, reground_at, reground=True)
    reach = {"flat": float(flat.goal_reached), "landmark": float(landmark.goal_reached)}
    horizon = {"flat": float(flat.priv_horizon), "landmark": float(landmark.priv_horizon)}
    rho = landmark.n_consults / goal_dist if goal_dist else 0.0
    return reach, horizon, rho


def _soundness(model: HostModel, oracle: HostOracle, journeys: list[Journey]) -> dict[str, float]:
    """The LP2 analogue: hoped privilege-edge false-edge rate + verified residual + cost ratio."""
    from verisim.host.delta import apply

    sig_to_id: dict[Any, int] = {}
    transitions: list[tuple[HostState, HostAction, int]] = []  # (src_state, action, true_dst_id)
    for j in journeys:
        prev_state = j.start
        prev_sig = privilege_signature(j.start)
        sig_to_id.setdefault(prev_sig, len(sig_to_id))
        for action, nxt in zip(j.actions, j.truth, strict=True):
            sig = privilege_signature(nxt)
            dst_id = sig_to_id.setdefault(sig, len(sig_to_id))
            if sig != prev_sig:
                transitions.append((prev_state, action, dst_id))
            prev_state, prev_sig = nxt, sig

    model_edges = correct = false = 0
    priv_bits = full_bits = 0
    for src_state, action, true_dst in transitions:
        pred = apply(src_state, model.predict_delta(src_state, action))
        proposed = sig_to_id.get(privilege_signature(pred))
        if proposed is not None:
            model_edges += 1
            if proposed == true_dst:
                correct += 1
            else:
                false += 1
        true_next = oracle.step(src_state, action).state
        priv_bits += privilege_facts(privilege_signature(true_next))
        full_bits += len(host_facts(true_next))
    return {
        "n_edges": float(len(transitions)),
        "false_edge_rate": false / model_edges if model_edges else 0.0,
        "verified_residual_false_rate": 0.0,
        "consult_bits_ratio": priv_bits / full_bits if full_bits else 0.0,
    }


def run_lp8_host(config: LP8HostConfig | None = None) -> tuple[list[LP8Stat], dict[str, float]]:
    """Train the flat host `M_θ`, then measure goal reach + privilege-graph soundness (H38)."""
    config = config or LP8HostConfig()
    host = HostConfig()
    oracle = ReferenceHostOracle()
    model = _train_model(config, host, oracle)

    g_max = max(max(config.goal_distances), config.budget_goal_distance)
    journeys = [
        _roll_journey(oracle, host, driver, seed, g_max)
        for driver in config.eval_difficulties.values()
        for seed in config.eval_seeds
    ]

    stats: list[LP8Stat] = []
    # -- distance sweep: fixed L, vary G ---------------------------------------------------------
    for goal_dist in config.goal_distances:
        per_arm_reach: dict[str, list[float]] = {"flat": [], "landmark": []}
        per_arm_horizon: dict[str, list[float]] = {"flat": [], "landmark": []}
        rhos: list[float] = []
        for journey in journeys:
            reach, horizon, rho = _cell_goal_reach(model, journey, goal_dist, config.hop_length)
            for arm in ("flat", "landmark"):
                per_arm_reach[arm].append(reach[arm])
                per_arm_horizon[arm].append(horizon[arm])
            rhos.append(rho)
        for arm in ("flat", "landmark"):
            stats.append(
                _reduce("distance", arm, float(goal_dist), per_arm_reach[arm],
                        per_arm_horizon[arm], 0.0 if arm == "flat" else fmean(rhos))
            )

    # -- budget sweep: fixed G, vary L (so ρ = 1/L) ----------------------------------------------
    g = config.budget_goal_distance
    flat_reach: list[float] = []
    flat_horizon: list[float] = []
    for journey in journeys:
        reach, horizon, _ = _cell_goal_reach(model, journey, g, config.budget_hop_lengths[0])
        flat_reach.append(reach["flat"])
        flat_horizon.append(horizon["flat"])
    stats.append(_reduce("budget", "flat", 0.0, flat_reach, flat_horizon, 0.0))
    for hop_len in config.budget_hop_lengths:
        lm_reach: list[float] = []
        lm_horizon: list[float] = []
        rhos = []
        for journey in journeys:
            reach, horizon, rho = _cell_goal_reach(model, journey, g, hop_len)
            lm_reach.append(reach["landmark"])
            lm_horizon.append(horizon["landmark"])
            rhos.append(rho)
        stats.append(
            _reduce("budget", "landmark", 1.0 / hop_len, lm_reach, lm_horizon, fmean(rhos))
        )

    soundness = _soundness(model, oracle, journeys)
    return stats, soundness


def _train_model(config: LP8HostConfig, host: HostConfig, oracle: HostOracle) -> HostModel:
    """Train the flat host `M_θ` — process-reproducibly (mirrors EH1)."""
    import torch

    from verisim.hostmodel import HostVocab, NeuralHostWorldModel, build_host_dataset
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_supervised

    torch.manual_seed(config.model_seed)
    torch.set_num_threads(1)  # process-reproducibility (SPEC-2 §12)
    vocab = HostVocab(host)
    examples = build_host_dataset(
        oracle, vocab, host, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    needed = max(len(p) + len(t) for p, t in examples) + 8
    block_size = max(config.block_size, needed)
    model = GPT(
        GPTConfig(
            vocab_size=len(vocab), block_size=block_size,
            n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
        )
    )
    train_supervised(
        model, examples, vocab.pad, steps=config.train_iters, lr=config.lr, seed=config.model_seed
    )
    return NeuralHostWorldModel(model, vocab)


def _reduce(
    sweep: str, arm: str, x_value: float, reach: list[float], horizon: list[float], rho: float
) -> LP8Stat:
    lo, hi = bootstrap_ci(reach, seed=0) if reach else (float("nan"), float("nan"))
    return LP8Stat(
        sweep=sweep, arm=arm, x_value=x_value,
        goal_reach=fmean(reach) if reach else float("nan"),
        gr_lo=lo, gr_hi=hi,
        priv_horizon=fmean(horizon) if horizon else float("nan"),
        rho=rho, n=len(reach),
    )


def _print_summary(stats: list[LP8Stat], soundness: dict[str, float]) -> None:
    print("LP8-host / H38 - cross-world fork: privilege-landmark planning on hostsim:")
    print(f"  {'sweep':<9} {'arm':<9} {'x':>7} {'goal_reach':>12} {'95% CI':>18} {'ρ':>6}")
    for s in stats:
        print(
            f"  {s.sweep:<9} {s.arm:<9} {s.x_value:>7.3f} {s.goal_reach:>12.3f} "
            f"{f'[{s.gr_lo:.3f}, {s.gr_hi:.3f}]':>18} {s.rho:>6.3f}"
        )
    print(
        f"  privilege-graph soundness (LP2 analogue): hoped false-edge rate "
        f"{soundness['false_edge_rate']:.3f}, verified residual "
        f"{soundness['verified_residual_false_rate']:.3f}, consult cost ratio "
        f"{soundness['consult_bits_ratio']:.3f} ({soundness['n_edges']:.0f} edges)"
    )
    dist = {(s.arm, s.x_value): s for s in stats if s.sweep == "distance"}
    xs = sorted({x for (_arm, x) in dist})
    flat_mean = fmean([dist[("flat", x)].goal_reach for x in xs])
    lm_mean = fmean([dist[("landmark", x)].goal_reach for x in xs])
    adv = lm_mean - flat_mean
    verdict = (
        f"privilege-landmark planning lifts goal reach above flat (mean {lm_mean:.3f} vs "
        f"{flat_mean:.3f}, adv {adv:+.3f}) - H38 supported in kind on the host world too"
        if adv > 0.02 else
        f"landmark ≈ flat (mean {lm_mean:.3f} vs {flat_mean:.3f}, adv {adv:+.3f}) - H38 NULL "
        "on host: the privilege projection is slow-moving / the model is already faithful, so "
        "re-grounding buys little (a result about which worlds' projections have the HS3 cliff)"
    )
    print(f"  verdict: {verdict}")


def _plot(  # pragma: no cover - plotting
    stats: list[LP8Stat], soundness: dict[str, float], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for arm, color, label in (
        ("flat", "#c33", "flat free-running (ρ=0)"),
        ("landmark", "#16a", "privilege-landmark planning (ρ≈1/L)"),
    ):
        cells = sorted((s for s in stats if s.sweep == "distance" and s.arm == arm),
                       key=lambda s: s.x_value)
        xs = [s.x_value for s in cells]
        ys = [s.goal_reach for s in cells]
        lo = [s.gr_lo for s in cells]
        hi = [s.gr_hi for s in cells]
        ax1.plot(xs, ys, "-o", color=color, label=label)
        ax1.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax1.set_xlabel("privilege-space distance G (steps)")
    ax1.set_ylabel("goal reach (privilege at goal correct)")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title("host fork: does re-grounding on privilege help?")
    ax1.legend(fontsize=8)

    budget = sorted((s for s in stats if s.sweep == "budget" and s.arm == "landmark"),
                    key=lambda s: s.x_value)
    xs = [s.x_value for s in budget]
    ys = [s.goal_reach for s in budget]
    lo = [s.gr_lo for s in budget]
    hi = [s.gr_hi for s in budget]
    ax2.plot(xs, ys, "-o", color="#16a", label="landmark (ρ = 1/L)")
    ax2.fill_between(xs, lo, hi, color="#16a", alpha=0.15)
    flat = next(s for s in stats if s.sweep == "budget" and s.arm == "flat")
    ax2.scatter([0.0], [flat.goal_reach], color="#c33", zorder=5, label="flat free-running (ρ=0)")
    ax2.set_xlabel("oracle budget ρ = 1/L (consults per step)")
    ax2.set_ylabel("goal reach (fixed far G)")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_title(
        f"verified privilege-graph: false edges {soundness['false_edge_rate']:.2f} -> "
        f"{soundness['verified_residual_false_rate']:.2f}"
    )
    ax2.legend(fontsize=8)

    fig.suptitle("LP8-host / H38: the landmark method on the host world (privilege projection)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="LP8-host cross-world fork (H38).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/lp8_host_goal_reach.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = LP8HostConfig.from_json_file(args.config) if args.config else LP8HostConfig()
    stats, soundness = run_lp8_host(cfg)
    _print_summary(stats, soundness)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, soundness, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
