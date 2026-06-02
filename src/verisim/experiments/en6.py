"""Experiment EN6: counterfactual grounding for change-safety (SPEC-5 §12, H5).

The oracle is total, so it generates **counterfactual branches for free**: from any visited state
``s`` it returns the exact next state of *alternative* actions ``a'`` the trajectory did not take,
``O(s, a')``. EN6 asks the H5 question for the *predictive* model (EN9 asked it for the contrastive
representation): does *training* the delta predictor on these free counterfactual branches improve
its prediction of **interventions** at test time -- the change-safety question a network-defense
simulator is actually asked ("if I deny this flow / drop this link, what happens to reachability?").

To separate the counterfactual signal from mere data volume, three arms train the **same**
graph+RSSM arm for the **same** number of steps, differing only in the training set's *composition*
(matched example count):

  - **trajectory** -- the base trajectory dataset (one action per visited state).
  - **trajectory-more** -- *more* trajectory data (extra seeds) to the same count: a volume control.
  - **+counterfactual** -- the base trajectory plus oracle counterfactual branches (``k`` sampled
    alternative actions per visited state), to the same count.

Evaluation is on **held-out** states: for each, sample intervention actions ``a'`` and score the
model against the oracle's truth two ways -- **intervention delta-exact** (did it predict the exact
edit set?) and **change-safety** = reachability-faithfulness of the predicted next state (did it get
the *reachability change* right, the metric a defender cares about?). *H5 supported* iff
``+counterfactual`` beats *both* trajectory arms -- the lift being structure, not volume.

*The two-oracle axis (H12) is deferred:* it needs a second, control-plane (Batfish-style) oracle
(SPEC-5 §5.1) -- a substantial new checker -- pre-registered for when that ships. EN6 here is the
single-oracle counterfactual-grounding slice, self-contained and local. CPU, deterministic.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from verisim.metrics.aggregate import bootstrap_ci
from verisim.net.action import NetAction
from verisim.net.config import NetConfig, scaled_net_config
from verisim.net.state import NetworkState
from verisim.netdelta import apply
from verisim.netdelta.edits import NetDelta
from verisim.netmetrics.divergence import reachability_faithfulness
from verisim.netmetrics.exact import delta_exact
from verisim.netoracle import ReferenceNetworkOracle

ARMS = ("trajectory", "trajectory-more", "+counterfactual")
METRICS = ("intervention_delta_exact", "change_safety")


@dataclass(frozen=True)
class EN6Config:
    """Small, fast counterfactual-grounding instance. Scale up for a publication run."""

    n_hosts: int = 5
    n_ports: int = 3
    train_driver: str = "weighted"
    intervention_driver: str = "uniform"  # diverse counterfactual / test interventions
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    k_counterfactual: int = 4  # counterfactual branches sampled per visited state
    graph_d_model: int = 48
    graph_mp_rounds: int = 3
    graph_iters: int = 1500
    lr: float = 3e-3
    model_seed: int = 0
    # evaluation: held-out states x sampled interventions
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    m_interventions: int = 6  # test interventions sampled per held-out state


@dataclass(frozen=True)
class ArmStat:
    """One (arm, metric) cell: mean + bootstrap CI over eval seeds."""

    arm: str
    metric: str
    mean: float
    ci_lo: float
    ci_hi: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.metric},{self.mean:.4f},{self.ci_lo:.4f},{self.ci_hi:.4f},{self.n}"
        )


CSV_HEADER = "arm,metric,mean,ci_lo,ci_hi,n"

# An intervention test case: (state, action, true delta, true next state).
EvalCase = tuple[NetworkState, NetAction, NetDelta, NetworkState]


def _intervention_cases(
    oracle: ReferenceNetworkOracle, config: NetConfig, driver: str, seed: int,
    n_steps: int, m: int,
) -> list[EvalCase]:
    """Roll a held-out trajectory; sample ``m`` interventions per state -> labeled test cases."""
    from verisim.netdata import NetDriver

    traj = NetDriver(name="weighted", config=config, rng=random.Random(seed))
    iv = NetDriver(name=driver, config=config, rng=random.Random(seed ^ 0x1217E))
    state = NetworkState.initial(config.hosts)
    cases: list[EvalCase] = []
    for _ in range(n_steps):
        for _k in range(m):
            a = iv.sample(state)
            r = oracle.step(state, a)
            cases.append((state, a, r.delta, r.state))
        state = oracle.step(state, traj.sample(state)).state  # advance on the true trajectory
    return cases


def run_en6(config: EN6Config | None = None) -> list[ArmStat]:
    """Train the three arms, then score intervention delta-exact + change-safety, held out."""
    import torch

    from verisim.netdata import NetDriver
    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph import build_graph
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import GraphExample, build_graph_dataset, train_graph_model
    from verisim.netmodel.tokenizer import encode_target

    config = config or EN6Config()
    torch.set_num_threads(1)  # process-reproducibility (the EN1 discipline)
    oracle = ReferenceNetworkOracle()
    net = scaled_net_config(config.n_hosts, config.n_ports)
    vocab = NetVocab(net)

    base_n = len(config.train_seeds) * config.train_steps_per_traj
    target_n = base_n * (1 + config.k_counterfactual)  # the matched example-count budget

    # --- the three training sets, all of size ~target_n ----------------------------
    traj = build_graph_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=config.train_seeds,
        n_steps=config.train_steps_per_traj,
    )
    more_seeds = tuple(range(-(-target_n // config.train_steps_per_traj)))  # ceil division
    traj_more = build_graph_dataset(
        oracle, vocab, net, driver=config.train_driver, seeds=more_seeds,
        n_steps=config.train_steps_per_traj,
    )[:target_n]

    cf: list[GraphExample] = []
    for seed in config.train_seeds:
        traj_drv = NetDriver(name=config.train_driver, config=net, rng=random.Random(seed))
        cf_rng = random.Random(seed ^ 0xC0FFEE)
        cf_drv = NetDriver(name=config.intervention_driver, config=net, rng=cf_rng)
        state = NetworkState.initial(net.hosts)
        for _ in range(config.train_steps_per_traj):
            for _k in range(config.k_counterfactual):
                a_cf = cf_drv.sample(state)
                delta = oracle.step(state, a_cf).delta
                cf.append((build_graph(state, a_cf, net), encode_target(delta, vocab)))
            state = oracle.step(state, traj_drv.sample(state)).state
    counterfactual = (traj + cf)[:target_n]

    datasets = {"trajectory": traj, "trajectory-more": traj_more, "+counterfactual": counterfactual}

    # --- held-out intervention test cases (shared across arms) ---------------------
    cases_by_seed = {
        seed: _intervention_cases(
            oracle, net, config.intervention_driver, seed,
            config.eval_steps, config.m_interventions,
        )
        for seed in config.eval_seeds
    }

    stats: list[ArmStat] = []
    for arm in ARMS:
        model = build_graph_model(
            vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds,
            seed=config.model_seed,
        )
        train_graph_model(model, datasets[arm], steps=config.graph_iters, seed=config.model_seed)
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in config.eval_seeds:
            cases = cases_by_seed[seed]
            exact = 0
            safety_sum = 0.0
            for state, action, true_delta, true_next in cases:
                pred = model.predict_delta(state, action)
                exact += int(delta_exact(pred, true_delta))
                safety_sum += reachability_faithfulness(apply(state, pred), true_next)
            total = len(cases)
            per_metric["intervention_delta_exact"].append(exact / total)
            per_metric["change_safety"].append(safety_sum / total)
        for metric, vals in per_metric.items():
            lo, hi = bootstrap_ci(vals, seed=0)
            stats.append(ArmStat(arm, metric, fmean(vals), lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[ArmStat]) -> None:
    print("EN6 counterfactual grounding — held-out intervention prediction (bootstrap CIs):")
    print(f"  {'arm':<16} {'metric':<26} {'mean':>8} {'95% CI':>18}")
    for s in stats:
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        print(f"  {s.arm:<16} {s.metric:<26} {s.mean:>8.3f} {ci:>18}")


def _plot(stats: list[ArmStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.4 * len(METRICS), 4.2), squeeze=False)
    colors = {"trajectory": "#9bd", "trajectory-more": "#c66", "+counterfactual": "#16a"}
    for ax, metric in zip(axes[0], METRICS, strict=True):
        cells = [s for s in stats if s.metric == metric]
        xs = range(len(cells))
        ax.bar(
            list(xs), [c.mean for c in cells],
            yerr=[[c.mean - c.ci_lo for c in cells], [c.ci_hi - c.mean for c in cells]],
            color=[colors.get(c.arm, "#999") for c in cells], capsize=4,
        )
        ax.set_xticks(list(xs))
        ax.set_xticklabels([c.arm for c in cells], rotation=15, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(metric)
    fig.suptitle("EN6 / H5: does counterfactual grounding improve intervention prediction?")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EN6 counterfactual grounding (H5).")
    parser.add_argument("--n-hosts", type=int, default=5)
    parser.add_argument("--graph-iters", type=int, default=1500)
    parser.add_argument("--k-counterfactual", type=int, default=4)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102])
    parser.add_argument("--out", type=str, default="figures/en6_counterfactual.csv")
    args = parser.parse_args()
    cfg = EN6Config(
        n_hosts=args.n_hosts, graph_iters=args.graph_iters,
        k_counterfactual=args.k_counterfactual, eval_seeds=tuple(args.eval_seeds),
    )
    stats = run_en6(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
