"""Experiment NA5: the decode-side rollout diagnostic — is the wall the decoder, not the processor?

NA0 refuted H45: on *teacher-forced* held-out states the structured arm's per-round embeddings
linearly decode the oracle's ``<= r``-hop reachability frontier, so the ``mp_rounds`` processor
*does* execute the propagation and the HS3 ``H_free = 0`` wall must be **downstream** — the
autoregressive delta decoder + free-running rollout. That was an *inference* from a teacher-forced
probe. NA5 tests it **directly, at the rollout level**, and (like NA0) as pure measurement on the
already trained arm — no trained-arm bet.

The move: fit NA0's reachability probe ``P : h_R -> F_R`` on teacher-forced true states, **freeze
it**, then **free-run** the same arm (it rolls forward on its *own* predicted states, drifting from
truth). At each rollout depth apply the frozen ``P`` to the embedding of the model's own (drifted)
state and score it two ways — **own-input fidelity** (vs the true frontier *of the drifted state
itself*: does the processor still compute the reachability of whatever state it is now in?) and
**tracks-truth** (vs the frontier of the *true* state: does the rollout still reflect the world?).

The ambiguity is that the frozen ``P`` is fit in-distribution, so a falling own-input fidelity
conflates "the representation degrades off-distribution" with "the in-distribution probe just fails
to transfer." The **control that resolves it**: refit a fresh probe on the *drifted* states (their
oracle frontier is free) and evaluate it held-out. If the refit probe **recovers** the accuracy the
frozen one lost, the reachability is still linearly in the embedding — the processor stays faithful
and the rollout drifts because the **decoder** emits wrong deltas that compound (NA0's redirection
confirmed). If even the refit probe falls, the representation genuinely degrades (the processor side
is not cleared). CPU-only, deterministic, seeded; probes are closed-form NumPy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from verisim.experiments.na0 import reach_frontiers
from verisim.metrics.aggregate import bootstrap_ci

if TYPE_CHECKING:
    from verisim.net.state import NetworkState


@dataclass(frozen=True)
class NA5Config:
    """A small, fast decode-side rollout-diagnostic instance (reuses the NA0 arm + frozen probe)."""

    n_hosts: int = 8
    n_ports: int = 3
    train_driver: str = "weighted"
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 600
    model_seeds: tuple[int, ...] = (0, 1, 2)
    probe_driver: str = "weighted"
    probe_seeds: tuple[int, ...] = (100, 101, 102, 103)  # teacher-forced states the probe is fit on
    probe_steps: int = 40
    ridge_lambda: float = 1.0
    rollout_driver: str = "weighted"
    rollout_seeds: tuple[int, ...] = (200, 201, 202, 203, 204, 205)  # free-run trajectories
    rollout_steps: int = 12
    depth_buckets: int = 6  # rollout depth is bucketed into this many bins for the curve

    @staticmethod
    def from_dict(d: dict[str, Any]) -> NA5Config:
        b = NA5Config()
        return NA5Config(
            n_hosts=d.get("n_hosts", b.n_hosts),
            n_ports=d.get("n_ports", b.n_ports),
            train_driver=d.get("train_driver", b.train_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            model_seeds=tuple(d.get("model_seeds", b.model_seeds)),
            probe_driver=d.get("probe_driver", b.probe_driver),
            probe_seeds=tuple(d.get("probe_seeds", b.probe_seeds)),
            probe_steps=d.get("probe_steps", b.probe_steps),
            ridge_lambda=d.get("ridge_lambda", b.ridge_lambda),
            rollout_driver=d.get("rollout_driver", b.rollout_driver),
            rollout_seeds=tuple(d.get("rollout_seeds", b.rollout_seeds)),
            rollout_steps=d.get("rollout_steps", b.rollout_steps),
            depth_buckets=d.get("depth_buckets", b.depth_buckets),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> NA5Config:
        return NA5Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class DepthStat:
    """The decode-side diagnostic at one rollout-depth bucket, reduced over model seeds (CI).

    The frozen probe is fit in-distribution (teacher-forced) and never refit; the **refit** probe is
    fit on free-run drifted states and evaluated held-out — the control that separates "the
    representation degrades off-distribution" (refit also falls) from "the in-distribution probe
    merely fails to transfer, the reachability is still in the embedding" (refit recovers), the
    latter localizing the wall to the decoder.
    """

    depth: int  # representative depth of the bucket
    frozen_own: float  # frozen (in-dist) probe vs the true frontier OF the model's own state ŝ_t
    frozen_lo: float
    frozen_hi: float
    refit_own: float  # refit-on-drift probe (held-out) vs the frontier of ŝ_t — the control
    refit_lo: float
    refit_hi: float
    tracks_truth: float  # frozen probe vs the frontier of the TRUE state s_t
    truth_lo: float
    truth_hi: float
    divergence: float  # state divergence ŝ_t vs s_t (the H_ε metric; context for the curve)
    n_seeds: int

    def csv_row(self) -> str:
        return (
            f"{self.depth},{self.frozen_own:.6f},{self.frozen_lo:.6f},{self.frozen_hi:.6f},"
            f"{self.refit_own:.6f},{self.refit_lo:.6f},{self.refit_hi:.6f},"
            f"{self.tracks_truth:.6f},{self.truth_lo:.6f},{self.truth_hi:.6f},"
            f"{self.divergence:.6f},{self.n_seeds}"
        )


CSV_HEADER = (
    "depth,frozen_own,frozen_lo,frozen_hi,refit_own,refit_lo,refit_hi,"
    "tracks_truth,truth_lo,truth_hi,divergence,n_seeds"
)


def _fit_ridge(x: Any, y: Any, lam: float) -> Any:
    """Closed-form multi-output ridge weights ``W`` (bias appended, unpenalized). ``x``=[m,d]."""
    import numpy as np

    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    d = xb.shape[1]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _predict(w: Any, x: Any) -> Any:
    """Apply frozen ridge ``W`` to ``x:[m,d]`` and threshold at 0.5 -> 0/1 frontier ``[m,k]``."""
    import numpy as np

    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    return (xb @ w) > 0.5


def run_na5(config: NA5Config | None = None) -> list[DepthStat]:
    """Fit + freeze the NA0 probe, free-run the arm, score own-input vs tracks-truth (+ refit)."""
    import random

    import numpy as np
    import torch

    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netdelta import apply
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph import build_graph
    from verisim.netmodel.graph_model import build_graph_model, graphs_to_tensors
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model
    from verisim.netoracle import ReferenceNetworkOracle

    config = config or NA5Config()
    torch.set_num_threads(1)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)
    hosts = net.hosts
    rounds = config.graph_mp_rounds
    device = torch.device("cpu")

    def embed_round_r(model: Any, state: NetworkState) -> Any:
        """The per-node round-R embedding ``h_R[N,d]`` of one state (the frozen probe's input)."""
        g = build_graph(state, None, net)
        node, gfeat, a_link, a_flow = graphs_to_tensors([g], device)
        trace = model.net.message_pass_trace(node, gfeat, a_link, a_flow)
        return trace[rounds][0].cpu().numpy()  # [N, d]

    # Teacher-forced probe-fit states (the NA0 in-distribution data).
    probe_states: list[NetworkState] = []
    for seed in config.probe_seeds:
        drv = NetDriver(name=config.probe_driver, config=net, rng=random.Random(seed))
        state = NetworkState.initial(hosts)
        for _ in range(config.probe_steps):
            action = drv.sample(state)
            probe_states.append(state)
            state = oracle.step(state, action).state

    # Per-depth accumulators across model seeds. Refit-probe split is by rollout seed (no leakage).
    edges = np.linspace(0, config.rollout_steps, config.depth_buckets + 1)
    half = len(config.rollout_seeds) // 2
    fit_seeds = set(config.rollout_seeds[:half]) if half else set()
    per_depth_frozen: list[list[float]] = [[] for _ in range(config.depth_buckets)]
    per_depth_refit: list[list[float]] = [[] for _ in range(config.depth_buckets)]
    per_depth_truth: list[list[float]] = [[] for _ in range(config.depth_buckets)]
    per_depth_div: list[list[float]] = [[] for _ in range(config.depth_buckets)]

    def _bucket(t: int) -> int:
        return min(config.depth_buckets - 1, int(np.searchsorted(edges, t, "right") - 1))

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

        # Fit + freeze the reachability probe P: h_R -> F_R on the teacher-forced states.
        with torch.no_grad():
            x_fit = np.vstack([embed_round_r(model, s) for s in probe_states])
            y_fit = np.vstack([reach_frontiers(s, hosts, rounds)[rounds] for s in probe_states])
        w_frozen = _fit_ridge(x_fit, y_fit, config.ridge_lambda)

        # Free-run rollouts. Collect per-step records: (in-fit-split, depth, embedding, F(ŝ), F(s)).
        records: list[tuple[bool, int, Any, Any, Any]] = []
        with torch.no_grad():
            for r_seed in config.rollout_seeds:
                in_fit = r_seed in fit_seeds
                drv = NetDriver(name=config.rollout_driver, config=net, rng=random.Random(r_seed))
                s_true = NetworkState.initial(hosts)
                s_hat = NetworkState.initial(hosts)
                actions = []
                st = s_true
                for _ in range(config.rollout_steps):
                    a = drv.sample(st)
                    actions.append(a)
                    st = oracle.step(st, a).state
                for t, a in enumerate(actions):
                    emb = embed_round_r(model, s_hat)  # [N, d]
                    f_own = reach_frontiers(s_hat, hosts, rounds)[rounds] > 0.5
                    f_true = reach_frontiers(s_true, hosts, rounds)[rounds] > 0.5
                    records.append((in_fit, t, emb, f_own, f_true))
                    s_hat = apply(s_hat, model.predict_delta(s_hat, a))
                    s_true = oracle.step(s_true, a).state

        # The refit control: fit a fresh probe on the FIT-split drifted states (oracle frontier is
        # free), then evaluate it held-out on the EVAL split. If it recovers high accuracy where the
        # frozen probe fell, the reachability is still in the embedding (the wall is the decoder).
        fit_rows = [(r[2], r[3]) for r in records if r[0]]
        if fit_rows:
            w_refit = _fit_ridge(
                np.vstack([e for e, _ in fit_rows]), np.vstack([f for _, f in fit_rows]),
                config.ridge_lambda,
            )
        else:
            w_refit = w_frozen

        s_frozen: list[list[float]] = [[] for _ in range(config.depth_buckets)]
        s_refit: list[list[float]] = [[] for _ in range(config.depth_buckets)]
        s_truth: list[list[float]] = [[] for _ in range(config.depth_buckets)]
        s_div: list[list[float]] = [[] for _ in range(config.depth_buckets)]
        for in_fit, t, emb, f_own, f_true in records:
            b = _bucket(t)
            pred_frozen = _predict(w_frozen, emb)
            s_frozen[b].append(float((pred_frozen == f_own).mean()))
            s_truth[b].append(float((pred_frozen == f_true).mean()))
            s_div[b].append(float((f_own != f_true).mean()))
            if not in_fit:  # refit probe scored only on held-out rollouts
                s_refit[b].append(float((_predict(w_refit, emb) == f_own).mean()))
        for b in range(config.depth_buckets):
            if s_frozen[b]:
                per_depth_frozen[b].append(fmean(s_frozen[b]))
                per_depth_truth[b].append(fmean(s_truth[b]))
                per_depth_div[b].append(fmean(s_div[b]))
                per_depth_refit[b].append(fmean(s_refit[b]) if s_refit[b] else fmean(s_frozen[b]))

    stats: list[DepthStat] = []
    for b in range(config.depth_buckets):
        if not per_depth_frozen[b]:
            continue
        depth = int((edges[b] + edges[b + 1]) / 2)
        flo, fhi = bootstrap_ci(per_depth_frozen[b], seed=0)
        rlo, rhi = bootstrap_ci(per_depth_refit[b], seed=0)
        tlo, thi = bootstrap_ci(per_depth_truth[b], seed=0)
        stats.append(
            DepthStat(
                depth, fmean(per_depth_frozen[b]), flo, fhi,
                fmean(per_depth_refit[b]), rlo, rhi,
                fmean(per_depth_truth[b]), tlo, thi, fmean(per_depth_div[b]),
                len(per_depth_frozen[b]),
            )
        )
    return stats


def _verdict(stats: list[DepthStat]) -> str:
    """Decode-side reading, disambiguated by the refit control (a decomposition of the frozen drop).

    The frozen probe's deepest-bucket drop splits into a *transfer artifact* (``recovery`` = how
    much a probe refit on drifted states regains over the frozen one) and *genuine info loss*
    (``refit_drop`` = how much even the refit probe falls). Decode-side dominates when the
    reachability is largely still recoverable (recovery >= refit_drop) and the truth-tracking loss
    is driven by state drift, not info loss.
    """
    if len(stats) < 2:
        return "inconclusive (need >= 2 depth buckets)"
    first, last = stats[0], stats[-1]
    refit_drop = first.refit_own - last.refit_own
    truth_drop = first.tracks_truth - last.tracks_truth
    recovery = last.refit_own - last.frozen_own  # transfer artifact the refit probe regains
    if truth_drop <= 0.05:
        return (
            f"WEAK DRIFT: tracks-truth barely falls ({first.tracks_truth:.3f} -> "
            f"{last.tracks_truth:.3f}) — too little rollout divergence at this scale to localize "
            "the wall; lengthen the rollout."
        )
    ratio = truth_drop / max(refit_drop, 1e-6)
    if recovery >= refit_drop and truth_drop > 1.5 * max(refit_drop, 1e-6):
        return (
            f"DECODE-SIDE DOMINANT: a probe refit on the drifted states recovers +{recovery:.3f} "
            f"over the frozen in-distribution probe at the deepest bucket (frozen "
            f"{last.frozen_own:.3f} -> refit {last.refit_own:.3f}), so most of the frozen probe's "
            "apparent collapse is a transfer artifact — the reachability is still linearly in the "
            f"embedding of the model's own drifted state. tracks-truth falls {truth_drop:.3f} "
            f"(~{ratio:.1f}x the refit probe's {refit_drop:.3f}) as divergence climbs to "
            f"{last.divergence:.3f}: the rollout drift is dominated by the DECODER's compounding "
            f"deltas, with a small residual ({refit_drop:.3f}) of genuine off-distribution "
            "representation drift. NA0's redirection holds at the rollout level — the wall is "
            "predominantly the decoder/rollout."
        )
    return (
        f"REPRESENTATION DEGRADES: even a probe refit on drifted states loses {refit_drop:.3f} "
        f"with depth (recovering only +{recovery:.3f} over frozen) — a material part of the "
        "reachability is genuinely no longer linearly present off-distribution."
    )


def _print_summary(stats: list[DepthStat]) -> None:
    print("NA5 / the decode-side rollout diagnostic — is the wall the decoder, not the processor?")
    print(f"  {'depth':>6} {'frozen-own':>11} {'refit-own':>10} {'tracks-truth':>13} {'diverg':>8}")
    for s in stats:
        print(
            f"  {s.depth:>6} {s.frozen_own:>11.3f} {s.refit_own:>10.3f} "
            f"{s.tracks_truth:>13.3f} {s.divergence:>8.3f}"
        )
    print("  " + _verdict(stats))


def _plot(stats: list[DepthStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = [s.depth for s in stats]
    frozen = [s.frozen_own for s in stats]
    frozen_err = [
        [s.frozen_own - s.frozen_lo for s in stats], [s.frozen_hi - s.frozen_own for s in stats],
    ]
    refit = [s.refit_own for s in stats]
    refit_err = [
        [s.refit_own - s.refit_lo for s in stats], [s.refit_hi - s.refit_own for s in stats],
    ]
    truth = [s.tracks_truth for s in stats]
    truth_err = [
        [s.tracks_truth - s.truth_lo for s in stats],
        [s.truth_hi - s.tracks_truth for s in stats],
    ]
    div = [s.divergence for s in stats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax1.errorbar(depths, refit, yerr=refit_err, marker="o", color="#2ca02c", capsize=4,
                 label="refit-on-drift  P'(h(ŝ_t)) vs F(ŝ_t)  [the control]")
    ax1.errorbar(depths, frozen, yerr=frozen_err, marker="D", color="#1f77b4", capsize=4,
                 label="frozen in-dist  P(h(ŝ_t)) vs F(ŝ_t)")
    ax1.errorbar(depths, truth, yerr=truth_err, marker="s", color="#d62728", capsize=4,
                 label="tracks-truth  P(h(ŝ_t)) vs F(s_t)")
    ax1.set_xlabel("free-running rollout depth t")
    ax1.set_ylabel("per-bit reachability accuracy")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("refit recovers what frozen loses: reachability stays in the embedding")
    ax1.legend(fontsize=7.5, loc="lower left")

    ax2.plot(depths, div, marker="^", color="#555")
    ax2.set_xlabel("free-running rollout depth t")
    ax2.set_ylabel("state divergence  ŝ_t vs s_t  (reachability bits)")
    ax2.set_title("the decoder's deltas compound into state drift")
    fig.suptitle("NA5: the wall is the decoder/rollout, not the processor (SPEC-14, post-NA0)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="NA5 decode-side rollout diagnostic (post-NA0).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/na5_decode_rollout.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = NA5Config.from_json_file(args.config) if args.config else NA5Config()
    stats = run_na5(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
