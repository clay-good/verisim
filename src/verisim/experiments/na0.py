"""Experiment NA0: does the graph processor *execute* the reachability propagation? (H45, SPEC-14).

The NAR diagnosis, and the gate that licenses NA1. SPEC-10's HS3 found the program's one genuine
wall: the structured GNN+RSSM arm free-runs to a faithful horizon of *zero* at exact tolerance even
though it beats the flat transformer ~6.6x on one-step delta-exact (EN4/H11) -- great one-step, zero
horizon, the textbook end-to-end-NAR symptom. SPEC-14 attacks that wall with hint supervision, but
only *after* a diagnosis: is the failure in the **processor** (the ``mp_rounds`` message passing
does not run the multi-hop BFS) or in the **decoder/rollout** (it runs the algorithm and the
zero-horizon is pure error accumulation)? The fix differs by branch, so NA0 decides it first.

The diagnostic is a linear probe on the trained arm's intermediate computation. The oracle's
reachability is a BFS / min-hop propagation; the free, exact **hint** at round ``r`` is the
``<= r``-hop reachability frontier (which hosts are reachable from each host in at most ``r`` link
hops over up-links between up-hosts). If the processor executes the propagation, its per-round node
embeddings ``h_r`` (:meth:`GraphRSSMNet.message_pass_trace`) should linearly decode the matched
round-``r`` frontier at every hop. **H45** predicts they do *not*: the probe recovers the 1-hop
frontier (immediate neighbors -- a one-step model has those) but its lift over the marginal-rate
baseline collapses at hops ``>= 2``, the multi-step propagation the arm never learned.

We report, per round ``r``, the probe's per-bit test accuracy and its lift over the best-constant
(per-bit majority) baseline, multi-seed with bootstrap CIs, against the arm's one-step next-state
exact ``p``. A collapse of the lift at ``r >= 2`` (with ``p`` at EN4 level) **supports H45** -- the
wall is in the processor, hint supervision (NA1) is the right fix. A lift that holds through round
``R`` would **refute H45** and redirect the spec to the decoder side. Pure diagnostics on a freshly
trained HS3-level arm: CPU, deterministic, seeded; the ridge probe is closed-form NumPy.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.metrics.aggregate import bootstrap_ci

if TYPE_CHECKING:
    from verisim.net.state import NetworkState


@dataclass(frozen=True)
class NA0Config:
    """A small, fast NAR-diagnosis (H45) instance: train an HS3-level arm, then probe per round."""

    n_hosts: int = 8
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 600
    model_seeds: tuple[int, ...] = (0, 1, 2)
    eval_driver: str = "weighted"
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    eval_steps: int = 40
    ridge_lambda: float = 1.0
    test_frac: float = 0.4
    split_seed: int = 7

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NA0Config:
        b = NA0Config()
        return NA0Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            eval_driver=d.get("eval_driver", b.eval_driver),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            ridge_lambda=d.get("ridge_lambda", b.ridge_lambda),
            test_frac=d.get("test_frac", b.test_frac),
            split_seed=d.get("split_seed", b.split_seed),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> NA0Config:
        return NA0Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class RoundStat:
    """The H45 diagnostic at one round, reduced over model seeds (bootstrap CI).

    ``lift`` is the *processed* probe ``h_r -> F_r`` minus baseline; ``control_lift`` is the
    *pre-propagation* probe ``h_0 -> F_r`` minus baseline (the input projection, which has no link
    adjacency, against the same target). ``lift - control_lift`` is what the ``r`` rounds of message
    passing *add* -- the load-bearing number: if message passing executes the propagation, the
    processed lift must exceed the control lift at hops ``>= 2`` by a clear margin.
    """

    round: int
    probe_acc: float  # per-bit test accuracy of the linear probe h_r -> <= r-hop frontier
    probe_lo: float
    probe_hi: float
    base_acc: float  # best-constant (per-bit majority) baseline at the same round
    lift: float  # probe_acc - base_acc: representation info beyond the marginal rate
    lift_lo: float
    lift_hi: float
    control_lift: float  # h_0 -> F_r lift: what the pre-propagation input embedding alone decodes
    control_lo: float
    control_hi: float
    n_seeds: int

    def csv_row(self) -> str:
        return (
            f"{self.round},{self.probe_acc:.6f},{self.probe_lo:.6f},{self.probe_hi:.6f},"
            f"{self.base_acc:.6f},{self.lift:.6f},{self.lift_lo:.6f},{self.lift_hi:.6f},"
            f"{self.control_lift:.6f},{self.control_lo:.6f},{self.control_hi:.6f},{self.n_seeds}"
        )


CSV_HEADER = (
    "round,probe_acc,probe_lo,probe_hi,base_acc,lift,lift_lo,lift_hi,"
    "control_lift,control_lo,control_hi,n_seeds"
)


def reach_frontiers(state: NetworkState, hosts: tuple[str, ...], max_r: int) -> list[Any]:
    """The oracle's free, exact ``<= r``-hop reachability frontier for ``r = 0 .. max_r``.

    Returns ``[F_0, ..., F_{max_r}]``, each an ``N x N`` 0/1 matrix with ``F_r[v, u] = 1`` iff host
    ``u`` is reachable from host ``v`` in at most ``r`` link hops over **up** links between **up**
    hosts -- exactly the propagation the ``mp_rounds`` message passing is the natural processor for.
    ``F_0`` is the identity over up hosts (a host reaches itself in zero hops iff it is up).
    """
    import numpy as np

    n = len(hosts)
    idx = {h: i for i, h in enumerate(hosts)}
    up = [bool(state.hosts.get(h) and state.hosts[h].up) for h in hosts]
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in state.links:
        if a in idx and b in idx and up[idx[a]] and up[idx[b]]:
            adj[idx[a]].append(idx[b])
            adj[idx[b]].append(idx[a])

    # Hop-bounded BFS from every source gives the full frontier stack in one pass.
    frontiers = [np.zeros((n, n), dtype=np.float64) for _ in range(max_r + 1)]
    for src in range(n):
        if not up[src]:
            continue
        dist = [-1] * n
        dist[src] = 0
        q: deque[int] = deque([src])
        while q:
            cur = q.popleft()
            if dist[cur] >= max_r:
                continue
            for nxt in adj[cur]:
                if dist[nxt] == -1:
                    dist[nxt] = dist[cur] + 1
                    q.append(nxt)
        for dst in range(n):
            if dist[dst] != -1:
                for r in range(dist[dst], max_r + 1):
                    frontiers[r][src, dst] = 1.0
    return frontiers


def _ridge_probe(
    x_tr: Any, y_tr: Any, x_te: Any, y_te: Any, lam: float
) -> tuple[float, float]:
    """Closed-form multi-output ridge probe. Returns ``(probe_acc, baseline_acc)`` on the test set.

    ``probe_acc`` is mean per-bit accuracy of ``X W > 0.5`` vs the 0/1 frontier; ``baseline_acc`` is
    the best-constant predictor (per-bit train majority) -- the marginal reachability rate a probe
    must beat to show the representation *carries* the frontier rather than the base rate.
    """
    import numpy as np

    xb_tr = np.hstack([x_tr, np.ones((x_tr.shape[0], 1))])
    xb_te = np.hstack([x_te, np.ones((x_te.shape[0], 1))])
    d = xb_tr.shape[1]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0  # do not penalize the bias
    w = np.linalg.solve(xb_tr.T @ xb_tr + reg, xb_tr.T @ y_tr)
    y_te_bin = y_te > 0.5
    pred = (xb_te @ w) > 0.5
    probe_acc = float((pred == y_te_bin).mean())
    majority = y_tr.mean(axis=0) > 0.5  # per-bit train majority, broadcast over the test rows
    base_acc = float((majority[None, :] == y_te_bin).mean())
    return probe_acc, base_acc


def _next_state_exact(model: Any, oracle: Any, states_actions: list[tuple[Any, Any]]) -> float:
    """One-step next-state-exact ``p`` (EN4 axis): predicted delta lands on the true next state."""
    from verisim.netdelta import apply

    if not states_actions:
        return float("nan")
    correct = 0
    for state, action in states_actions:
        pred_next = apply(state, model.predict_delta(state, action))
        true_next = oracle.step(state, action).state
        correct += int(pred_next == true_next)
    return correct / len(states_actions)


def run_na0(config: NA0Config | None = None) -> tuple[list[RoundStat], float, float, float]:
    """Train HS3-level arms, probe per-round embeddings vs the oracle frontier (H45).

    Returns ``(round_stats, p_mean, p_lo, p_hi)`` -- the per-round probe diagnostic and the arm's
    one-step next-state-exact accuracy (mean + bootstrap CI over model seeds), the EN4 comparator.
    """
    import random

    import numpy as np
    import torch

    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph import build_graph
    from verisim.netmodel.graph_model import build_graph_model, graphs_to_tensors
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or NA0Config()
    torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    hosts = net.hosts
    rounds = config.graph_mp_rounds
    device = torch.device("cpu")

    # Held-out rollouts: collect states (for the probe) and (state, action) (for one-step p).
    eval_states: list[NetworkState] = []
    eval_sa: list[tuple[Any, Any]] = []
    for seed in config.eval_seeds:
        driver = NetDriver(name=config.eval_driver, config=net, rng=random.Random(seed))
        state = NetworkState.initial(hosts)
        for _ in range(config.eval_steps):
            action = driver.sample(state)
            eval_states.append(state)
            eval_sa.append((state, action))
            state = oracle.step(state, action).state
    frontiers = [reach_frontiers(s, hosts, rounds) for s in eval_states]

    # Train/test split by held-out state (no node-level leakage across the split).
    n_states = len(eval_states)
    order = list(range(n_states))
    random.Random(config.split_seed).shuffle(order)
    n_test = max(1, round(config.test_frac * n_states))
    test_idx = set(order[:n_test])
    tr_states = [i for i in range(n_states) if i not in test_idx]
    te_states = [i for i in range(n_states) if i in test_idx]

    per_round_probe: dict[int, list[float]] = {r: [] for r in range(rounds + 1)}
    per_round_base: dict[int, list[float]] = {r: [] for r in range(rounds + 1)}
    per_round_lift: dict[int, list[float]] = {r: [] for r in range(rounds + 1)}
    per_round_control: dict[int, list[float]] = {r: [] for r in range(rounds + 1)}
    p_vals: list[float] = []

    for model_seed in config.model_seeds:
        model = build_graph_model(
            vocab, net, d_model=config.graph_d_model, mp_rounds=rounds, seed=model_seed,
        )
        examples = build_graph_dataset(
            oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
            n_steps=config.train_steps_per_traj,
        )
        train_graph_model(model, examples, steps=config.graph_iters, seed=model_seed)
        model.net.eval()

        # Per-round node embeddings on every held-out state (one forward pass each).
        traces: list[list[Any]] = []
        with torch.no_grad():
            for state in eval_states:
                g = build_graph(state, None, net)
                node, gfeat, a_link, a_flow = graphs_to_tensors([g], device)
                trace = model.net.message_pass_trace(node, gfeat, a_link, a_flow)
                traces.append([h[0].cpu().numpy() for h in trace])  # each [N, d]

        # Pre-propagation control embedding (round 0: node_in only, no link adjacency).
        x0_tr = np.vstack([traces[i][0] for i in tr_states])
        x0_te = np.vstack([traces[i][0] for i in te_states])
        for r in range(rounds + 1):
            x_tr = np.vstack([traces[i][r] for i in tr_states])
            x_te = np.vstack([traces[i][r] for i in te_states])
            y_tr = np.vstack([frontiers[i][r] for i in tr_states])
            y_te = np.vstack([frontiers[i][r] for i in te_states])
            probe_acc, base_acc = _ridge_probe(x_tr, y_tr, x_te, y_te, config.ridge_lambda)
            per_round_probe[r].append(probe_acc)
            per_round_base[r].append(base_acc)
            per_round_lift[r].append(probe_acc - base_acc)
            # Same target F_r, but decoded from the pre-propagation embedding h_0.
            ctrl_acc, _ = _ridge_probe(x0_tr, y_tr, x0_te, y_te, config.ridge_lambda)
            per_round_control[r].append(ctrl_acc - base_acc)

        p_vals.append(_next_state_exact(model, oracle, eval_sa))

    stats: list[RoundStat] = []
    for r in range(rounds + 1):
        pa = fmean(per_round_probe[r])
        plo, phi = bootstrap_ci(per_round_probe[r], seed=0)
        ba = fmean(per_round_base[r])
        lf = fmean(per_round_lift[r])
        llo, lhi = bootstrap_ci(per_round_lift[r], seed=0)
        cl = fmean(per_round_control[r])
        clo, chi = bootstrap_ci(per_round_control[r], seed=0)
        stats.append(
            RoundStat(r, pa, plo, phi, ba, lf, llo, lhi, cl, clo, chi, len(config.model_seeds))
        )

    p_mean = fmean(p_vals)
    p_lo, p_hi = bootstrap_ci(p_vals, seed=0)
    return stats, p_mean, p_lo, p_hi


def _verdict(stats: list[RoundStat]) -> str:
    """H45: does message passing supply the multi-hop reachability the input embedding lacks?

    H45 (processor fails to propagate) is *supported* if the processed lift collapses at hops >= 2:
    the embeddings carry no more multi-hop frontier than the marginal. It is *refuted* if the
    processed lift not only holds but exceeds the pre-propagation control by a clear margin: message
    passing is *executing* the propagation, so the H_free=0 wall is downstream (decoder/rollout).
    """
    deep = [s for s in stats if s.round >= 2]
    if not deep:
        return "inconclusive (need a round >= 2)"
    proc = max(s.lift for s in deep)
    ctrl = max(s.control_lift for s in deep)
    if proc < 0.05:
        return (
            f"H45 SUPPORTED: deep (>=2-hop) processed lift {proc:.3f} ~ 0 -- the processor does "
            "not execute multi-step propagation (NA1 hint supervision is the right fix)."
        )
    if proc > 2.0 * max(ctrl, 1e-6):
        return (
            f"H45 REFUTED: deep processed lift {proc:.3f} >> pre-propagation control {ctrl:.3f} "
            "-- message passing supplies the multi-hop reachability the input lacks; the processor "
            "executes the propagation, so the H_free=0 wall is downstream (decoder/rollout side)."
        )
    return (
        f"H45 MIXED: deep processed lift {proc:.3f} vs control {ctrl:.3f} -- the message-passing "
        "margin is present but modest; alignment (NA2) may be the lever."
    )


def _print_summary(stats: list[RoundStat], p_mean: float, p_lo: float, p_hi: float) -> None:
    print("NA0 / H45 - does the graph processor execute the reachability propagation?")
    print(f"  one-step next-state-exact p = {p_mean:.3f} [{p_lo:.3f}, {p_hi:.3f}] (EN4 level)")
    print(f"  {'round':<8} {'probe acc':>11} {'baseline':>9} {'lift':>8} {'ctrl(h0)':>9}")
    for s in stats:
        print(
            f"  r={s.round:<14} {s.probe_acc:>11.3f} {s.base_acc:>9.3f} "
            f"{s.lift:>8.3f} {s.control_lift:>9.3f}"
        )
    print("  " + _verdict(stats))


def _plot(stats: list[RoundStat], p_mean: float, path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [s.round for s in stats]
    probe = [s.probe_acc for s in stats]
    probe_err = [
        [s.probe_acc - s.probe_lo for s in stats],
        [s.probe_hi - s.probe_acc for s in stats],
    ]
    base = [s.base_acc for s in stats]
    lift = [s.lift for s in stats]
    lift_err = [
        [s.lift - s.lift_lo for s in stats],
        [s.lift_hi - s.lift for s in stats],
    ]
    control = [s.control_lift for s in stats]
    control_err = [
        [s.control_lift - s.control_lo for s in stats],
        [s.control_hi - s.control_lift for s in stats],
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.errorbar(rounds, probe, yerr=probe_err, marker="o", color="#16a", capsize=4,
                 label="linear probe acc")
    ax1.plot(rounds, base, marker="s", color="#999", linestyle="--", label="marginal baseline")
    ax1.set_xlabel("message-passing round r  (probe vs <= r-hop frontier)")
    ax1.set_ylabel("per-bit reachability accuracy")
    ax1.set_xticks(rounds)
    ax1.set_ylim(0, 1.02)
    ax1.set_title(f"per-round frontier decodability (one-step p = {p_mean:.2f})")
    ax1.legend(fontsize=8, loc="lower left")

    width = 0.38
    rr = [r - width / 2 for r in rounds]
    rc = [r + width / 2 for r in rounds]
    ax2.bar(rr, lift, width, yerr=lift_err, color="#16a", capsize=3,
            label="processed  h_r -> F_r")
    ax2.bar(rc, control, width, yerr=control_err, color="#c99", capsize=3,
            label="control  h_0 -> F_r (no propagation)")
    ax2.axhline(0.0, color="#333", linewidth=0.8)
    ax2.set_xlabel("message-passing round r")
    ax2.set_ylabel("lift over marginal baseline")
    ax2.set_xticks(rounds)
    ax2.set_title("message passing supplies multi-hop reachability (H45 refuted)")
    ax2.legend(fontsize=8, loc="upper left")
    fig.suptitle("NA0 / H45: does the graph processor execute the oracle's reachability BFS?")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="NA0 NAR diagnosis: processor propagation (H45).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/na0_hint_probe.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = NA0Config.from_json_file(args.config) if args.config else NA0Config()
    stats, p_mean, p_lo, p_hi = run_na0(cfg)
    _print_summary(stats, p_mean, p_lo, p_hi)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, p_mean, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
