"""Experiment EH6 (counterfactual arm) -- counterfactual grounding for the host (SPEC-6 §2.8, H16).

The host oracle is *total*, so from any visited bundle state ``s`` it returns the exact next state
of **alternative** syscalls ``a'`` the trajectory did not take, ``O(s, a')`` -- a free
counterfactual branch. This is the capability physical-domain causal world models structurally lack
(§2.8): you can re-run the process tree from step ``t`` with one syscall changed and read back the
*true* alternative state. EH6's counterfactual arm asks the **H16** question (the two-oracle/H12
slice ships separately in :mod:`.eh6_two_oracle`): does *training* the host delta predictor on these
free counterfactual branches improve its prediction of **interventions** at test time?

The host makes the question security-relevant in a way the network world could not. The
interventions that matter for a defender are **near-miss privilege mistakes** (§17.7): "what if this
process had called ``setuid`` instead?", "what if it wrote a closed fd?" -- the syscalls whose
oracle outcome is an **EPERM/EBADF denial**. So the counterfactual driver is ``adversarial``
(setuid/exit/close-heavy: the denial-rich near-misses).

To separate the counterfactual *signal* from mere data volume, three arms train the **same**
factored graph+RSSM arm for the **same** number of steps, differing only in the training set's
*composition* (matched example count):

  - **trajectory** -- the base ``forky`` trajectory dataset (one action per visited state).
  - **trajectory-more** -- *more* trajectory data (extra seeds) to the same count: a volume control.
  - **+counterfactual** -- the base trajectory plus oracle counterfactual branches (``k`` sampled
    alternative ``adversarial`` syscalls per visited state), to the same count.

Evaluation is on **held-out** states: for each, sample intervention syscalls ``a'`` and score the
model against the oracle two ways -- **intervention exact** (the robust headline: did
``apply(s, Δ̂)`` equal the true next bundle state on every subsystem?) and **intervention denied
recall** (the security-relevant readout: of the interventions truth *denies*, does the model
predict the failure?). *H16 supported* iff ``+counterfactual`` beats *both* trajectory arms -- the
lift being structure, not volume. CPU, deterministic, torch-gated like every learned-arm experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.action import HostAction
from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.delta import HostDelta, apply
from verisim.host.state import HostState
from verisim.hostdata.drivers import HostDriver
from verisim.hostmetrics.divergence import divergence
from verisim.hostoracle.base import EXIT_OK, HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.metrics.aggregate import bootstrap_ci

from .eh8_privilege import predicted_exit

ARMS = ("trajectory", "trajectory-more", "+counterfactual")
METRICS = ("intervention_exact", "intervention_denied_recall")


@dataclass(frozen=True)
class EH6CFConfig:
    """Small, fast counterfactual-grounding instance. Scale up for a publication run."""

    train_driver: str = "forky"
    intervention_driver: str = "adversarial"  # denial-rich near-miss interventions (§17.7)
    train_seeds: tuple[int, ...] = (0, 1, 2)
    train_steps_per_traj: int = 40
    k_counterfactual: int = 4  # counterfactual branches sampled per visited state
    max_pid: int = 64
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 1200
    graph_batch: int = 32
    lr: float = 3e-3
    model_seed: int = 0
    # evaluation: held-out states x sampled interventions
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    m_interventions: int = 6  # test interventions sampled per held-out state

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH6CFConfig:
        b = EH6CFConfig()
        return EH6CFConfig(
            train_driver=d.get("train_driver", b.train_driver),
            intervention_driver=d.get("intervention_driver", b.intervention_driver),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            k_counterfactual=d.get("k_counterfactual", b.k_counterfactual),
            max_pid=d.get("max_pid", b.max_pid),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_batch=d.get("graph_batch", b.graph_batch),
            lr=d.get("lr", b.lr),
            model_seed=d.get("model_seed", b.model_seed),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            m_interventions=d.get("m_interventions", b.m_interventions),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH6CFConfig:
        return EH6CFConfig.from_dict(json.loads(Path(path).read_text()))


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

# An intervention test case: (state, action, true next state, true exit code).
EvalCase = tuple[HostState, HostAction, HostState, int]


def _intervention_cases(
    oracle: HostOracle, config: HostConfig, base_driver: str, iv_driver: str,
    seed: int, n_steps: int, m: int,
) -> list[EvalCase]:
    """Roll a held-out trajectory; sample ``m`` interventions per state -> labeled test cases."""
    traj = HostDriver(name=base_driver, config=config, rng=random.Random(seed))
    iv = HostDriver(name=iv_driver, config=config, rng=random.Random(seed ^ 0x1217E))
    state = HostState.initial()
    cases: list[EvalCase] = []
    for _ in range(n_steps):
        for _k in range(m):
            a = iv.sample(state)
            r = oracle.step(state, a)
            cases.append((state, a, r.state, r.exit_code))
        state = oracle.step(state, traj.sample(state)).state  # advance on the true trajectory
    return cases


def _score(model: Any, cases: list[EvalCase]) -> dict[str, float]:
    """Intervention exact + denied recall for one arm over one held-out seed's cases."""
    exact = 0
    denied_total = 0
    denied_hit = 0
    for state, action, true_next, true_exit in cases:
        delta: HostDelta = model.predict_delta(state, action)
        if divergence(apply(state, delta), true_next) == 0.0:
            exact += 1
        if true_exit != EXIT_OK:
            denied_total += 1
            if predicted_exit(delta) != EXIT_OK:
                denied_hit += 1
    return {
        "intervention_exact": exact / len(cases) if cases else 1.0,
        "intervention_denied_recall": denied_hit / denied_total if denied_total else 1.0,
    }


def run_eh6_counterfactual(config: EH6CFConfig | None = None) -> list[ArmStat]:
    """Train the three arms, then score intervention exact + denied recall, held out (H16)."""
    import torch

    from verisim.hostmodel import HostVocab, build_host_graph, encode_target
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import GraphExample, train_host_graph_model

    config = config or EH6CFConfig()
    torch.set_num_threads(1)  # process-reproducibility (the E1/EN1 discipline)
    oracle = ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    def graph_example(state: HostState, action: HostAction) -> GraphExample:
        delta = oracle.step(state, action).delta
        return (build_host_graph(state, action, host, vocab.max_pid), encode_target(delta, vocab))

    def trajectory_examples(driver: str, seeds: tuple[int, ...]) -> list[GraphExample]:
        out: list[GraphExample] = []
        for seed in seeds:
            drv = HostDriver(name=driver, config=host, rng=random.Random(seed))
            state = HostState.initial()
            for _ in range(config.train_steps_per_traj):
                action = drv.sample(state)
                out.append(graph_example(state, action))
                state = oracle.step(state, action).state
        return out

    base_n = len(config.train_seeds) * config.train_steps_per_traj
    target_n = base_n * (1 + config.k_counterfactual)  # the matched example-count budget

    # --- the three training sets, all of size ~target_n ----------------------------
    traj = trajectory_examples(config.train_driver, config.train_seeds)
    more_seeds = tuple(range(-(-target_n // config.train_steps_per_traj)))  # ceil division
    traj_more = trajectory_examples(config.train_driver, more_seeds)[:target_n]

    cf: list[GraphExample] = []
    for seed in config.train_seeds:
        traj_drv = HostDriver(name=config.train_driver, config=host, rng=random.Random(seed))
        cf_drv = HostDriver(
            name=config.intervention_driver, config=host, rng=random.Random(seed ^ 0xC0FFEE)
        )
        state = HostState.initial()
        for _ in range(config.train_steps_per_traj):
            for _k in range(config.k_counterfactual):
                cf.append(graph_example(state, cf_drv.sample(state)))  # free counterfactual branch
            state = oracle.step(state, traj_drv.sample(state)).state
    counterfactual = (traj + cf)[:target_n]

    datasets = {"trajectory": traj, "trajectory-more": traj_more, "+counterfactual": counterfactual}

    # --- held-out intervention test cases (shared across arms) ---------------------
    cases_by_seed = {
        seed: _intervention_cases(
            oracle, host, config.train_driver, config.intervention_driver,
            seed, config.eval_steps, config.m_interventions,
        )
        for seed in config.eval_seeds
    }

    stats: list[ArmStat] = []
    for arm in ARMS:
        model = build_host_graph_model(
            vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
            mp_rounds=config.graph_mp_rounds, seed=config.model_seed,
        )
        train_host_graph_model(
            model, datasets[arm], steps=config.graph_iters, lr=config.lr,
            batch_size=config.graph_batch, seed=config.model_seed,
        )
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in config.eval_seeds:
            scored = _score(model, cases_by_seed[seed])
            for metric in METRICS:
                per_metric[metric].append(scored[metric])
        for metric, vals in per_metric.items():
            lo, hi = bootstrap_ci(vals, seed=0)
            stats.append(ArmStat(arm, metric, fmean(vals), lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[ArmStat]) -> None:
    print("EH6 counterfactual grounding — held-out intervention prediction (bootstrap CIs):")
    print(f"  {'arm':<16} {'metric':<28} {'mean':>8} {'95% CI':>18}")
    for s in stats:
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        print(f"  {s.arm:<16} {s.metric:<28} {s.mean:>8.3f} {ci:>18}")


def _plot(stats: list[ArmStat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "intervention_exact": "intervention exact\n(full bundle next-state)",
        "intervention_denied_recall": "intervention denied recall\n(predicts the EPERM/EBADF)",
    }
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
        ax.set_ylim(0, 1.05)
        ax.set_title(titles[metric])
    fig.suptitle("Verisim EH6 / H16 — does free oracle counterfactual replay train intervention "
                 "fidelity?")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH6 counterfactual grounding (H16).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh6_counterfactual.csv")
    args = parser.parse_args()
    config = EH6CFConfig.from_json_file(args.config) if args.config else EH6CFConfig()
    stats = run_eh6_counterfactual(config)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
