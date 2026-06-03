"""Experiment EH-stream -- the experience stream vs the batch, with a plasticity probe (SPEC-6 §8.5,
H15 / HW-4).

The host world is long and generative enough that the propose-verify-correct loop need never stop: a
continuous sandboxed stream of host activity from which the model predicts, the oracle verifies for
free, and the model heals from a replay buffer -- indefinitely (the "Era of Experience" loop, §2.4 /
§8.5). **H15** asks whether that stream *beats the batch* at **equal total compute**; **HW-4** asks
whether a model that learns forever keeps learning or **ossifies** (loss of plasticity, §2.5). This
experiment makes both falsifiable on the factored host arm, at smoke scale.

Three arms see the **same** experience stream (the same ordered oracle-labeled transitions) and take
the **same number of gradient steps** ``G`` (the stream length -- one pass) on the same tiny graph
arm, differing only in *how* they consume it:

  - **batch** -- the offline baseline: shuffle the whole stream and take ``G`` minibatch steps (the
    standard supervised regime; the data is an i.i.d. pool).
  - **stream+replay** -- the Era-of-Experience recipe: walk the stream **in order**, append each
    arriving transition to a growing replay buffer, and take one gradient step on a minibatch
    *sampled from the whole buffer* per arrival (the stream-x + experience-replay fix for forgetting
    *and* plasticity loss, §2.5).
  - **stream-no-replay** -- the forgetting-prone control: walk the stream in order but train only on
    the **most recent window** of transitions (no sampling from the past) -- isolates *replay* as
    the lever at equal batch size and equal compute.

Two readouts on held-out rollouts: **one-step exact** (teacher-forced delta-exactness, the stable
accuracy) and the free-running **faithful horizon** ``H_ε`` (the headline, noisier at smoke scale).
Plus the **plasticity probe** (HW-4, §9.4): after training, clone each arm's final model and measure
how much loss it can still shed on a **frozen, never-seen probe batch** in ``K`` fresh gradient
steps -- ``(L0 - L_K)/L0``. A model that has lost plasticity sheds less. *H15 supported* iff
``stream+replay`` matches or beats ``batch`` at equal compute; *HW-4 localized* iff the no-replay
stream's
plasticity decays below the others. Either way it is a datum (negatives are first-class). CPU,
deterministic, torch-gated.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.config import DEFAULT_HOST_CONFIG, HostConfig
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostmetrics.divergence import divergence
from verisim.hostoracle.base import HostOracle
from verisim.hostoracle.reference import ReferenceHostOracle
from verisim.metrics.aggregate import bootstrap_ci

from .eh1 import eval_actions

ARMS = ("batch", "stream+replay", "stream-no-replay")
METRICS = ("one_step_exact", "free_horizon", "plasticity")


@dataclass(frozen=True)
class EHStreamConfig:
    """Small, fast experience-stream instance. Scale up (stream length, world) for a real run."""

    # the experience stream: ordered oracle-labeled transitions across seeds (one pass = G steps)
    stream_driver: str = "forky"
    stream_seeds: tuple[int, ...] = tuple(range(16))
    stream_steps_per_traj: int = 30
    replay_batch: int = 16  # minibatch size (shared by all arms, so compute is matched)
    max_pid: int = 64
    graph_d_model: int = 48
    graph_mp_rounds: int = 3
    lr: float = 3e-3
    model_seed: int = 0
    # evaluation
    difficulties: dict[str, str] = field(
        default_factory=lambda: {"low": "forky", "high": "adversarial"}
    )
    eval_seeds: tuple[int, ...] = (100, 101, 102)
    eval_steps: int = 24
    epsilon: float = 0.05
    # the HW-4 plasticity probe: a frozen, never-trained-on batch + K fresh steps
    probe_seed: int = 900
    probe_size: int = 48
    probe_steps: int = 20
    probe_lr: float = 3e-3

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EHStreamConfig:
        b = EHStreamConfig()
        return EHStreamConfig(
            stream_driver=d.get("stream_driver", b.stream_driver),
            stream_seeds=tuple(d.get("stream_seeds", b.stream_seeds)),
            stream_steps_per_traj=d.get("stream_steps_per_traj", b.stream_steps_per_traj),
            replay_batch=d.get("replay_batch", b.replay_batch),
            max_pid=d.get("max_pid", b.max_pid),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            lr=d.get("lr", b.lr),
            model_seed=d.get("model_seed", b.model_seed),
            difficulties=dict(d.get("difficulties", b.difficulties)),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            epsilon=d.get("epsilon", b.epsilon),
            probe_seed=d.get("probe_seed", b.probe_seed),
            probe_size=d.get("probe_size", b.probe_size),
            probe_steps=d.get("probe_steps", b.probe_steps),
            probe_lr=d.get("probe_lr", b.probe_lr),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EHStreamConfig:
        return EHStreamConfig.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ArmStat:
    """One (arm, metric) cell: mean + bootstrap CI over (difficulty x eval seed)."""

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


def _eval_rollout(
    model: Any, oracle: HostOracle, host: HostConfig, driver: str, seed: int,
    n_steps: int, epsilon: float,
) -> tuple[float, int]:
    """Return ``(one_step_exact_fraction, free_running_horizon)`` for one held-out rollout.

    One-step exact is teacher-forced (predict from the *true* state each step); the free horizon
    rolls the model on its *own* predictions until composed divergence first exceeds ``epsilon``.
    """
    actions = eval_actions(oracle, host, driver, seed, n_steps)
    # teacher-forced one-step exactness
    state = HostState.initial()
    exact = 0
    truth_states: list[HostState] = []
    for action in actions:
        truth = oracle.step(state, action).state
        truth_states.append(truth)
        if divergence(apply(state, model.predict_delta(state, action)), truth) == 0.0:
            exact += 1
        state = truth
    one_step = exact / len(actions) if actions else 1.0
    # free-running horizon
    free = HostState.initial()
    horizon = len(actions)
    for i, action in enumerate(actions):
        free = apply(free, model.predict_delta(free, action))
        if divergence(free, truth_states[i]) > epsilon:
            horizon = i
            break
    return one_step, horizon


def run_eh_stream(config: EHStreamConfig | None = None) -> list[ArmStat]:
    """Train the 3 arms on one stream at equal compute; score + probe plasticity (H15/HW-4)."""
    import copy

    import torch

    from verisim.hostmodel import HostVocab, build_host_graph, encode_target
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import (
        GraphExample,
        online_update,
        train_host_graph_model,
    )

    config = config or EHStreamConfig()
    torch.set_num_threads(1)  # process-reproducibility (the E1/EN1 discipline)
    oracle = ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)

    def example(state: HostState, action: Any) -> GraphExample:
        delta = oracle.step(state, action).delta
        return (build_host_graph(state, action, host, vocab.max_pid), encode_target(delta, vocab))

    def roll_examples(driver: str, seeds: tuple[int, ...], n_steps: int) -> list[GraphExample]:
        from verisim.hostdata.drivers import HostDriver

        out: list[GraphExample] = []
        for seed in seeds:
            drv = HostDriver(name=driver, config=host, rng=random.Random(seed))
            state = HostState.initial()
            for _ in range(n_steps):
                action = drv.sample(state)
                out.append(example(state, action))
                state = oracle.step(state, action).state
        return out

    # The experience stream (ordered) and the frozen, never-trained-on plasticity probe batch.
    stream = roll_examples(config.stream_driver, config.stream_seeds, config.stream_steps_per_traj)
    g_steps = len(stream)  # one pass -> equal compute across arms
    probe = roll_examples(config.stream_driver, (config.probe_seed,), config.probe_size)[
        : config.probe_size
    ]

    def fresh_model() -> Any:
        return build_host_graph_model(
            vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
            mp_rounds=config.graph_mp_rounds, seed=config.model_seed,
        )

    def plasticity(model: Any) -> float:
        """Fraction of probe-batch loss a *clone* can still shed in K fresh steps (HW-4, §9.4)."""
        clone = copy.deepcopy(model)
        opt = torch.optim.AdamW(clone.net.parameters(), lr=config.probe_lr)
        from verisim.hostmodel.graph_train import _batch_loss

        clone.net.train()
        l0 = float(_batch_loss(clone, probe, sample=False).item())
        online_update(clone, opt, probe, steps=config.probe_steps)
        lk = float(_batch_loss(clone, probe, sample=False).item())
        return max(0.0, (l0 - lk) / l0) if l0 > 0 else 0.0

    def train_arm(arm: str) -> Any:
        torch.manual_seed(config.model_seed)  # reproducible RSSM sampling per arm
        model = fresh_model()
        if arm == "batch":
            train_host_graph_model(
                model, stream, steps=g_steps, lr=config.lr,
                batch_size=config.replay_batch, seed=config.model_seed,
            )
            return model
        opt = torch.optim.AdamW(model.net.parameters(), lr=config.lr)
        rng = random.Random(config.model_seed)
        buffer: list[GraphExample] = []
        b = config.replay_batch
        for ex in stream:  # one pass, in order -> g_steps gradient steps
            buffer.append(ex)
            if arm == "stream+replay":  # sample a minibatch from the whole history
                batch = rng.sample(buffer, min(b, len(buffer)))
            else:  # stream-no-replay: only the most recent window
                batch = buffer[-b:]
            online_update(model, opt, batch, steps=1)
        return model

    stats: list[ArmStat] = []
    for arm in ARMS:
        model = train_arm(arm)
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for _difficulty, driver in config.difficulties.items():
            for seed in config.eval_seeds:
                one_step, horizon = _eval_rollout(
                    model, oracle, host, driver, seed, config.eval_steps, config.epsilon
                )
                per_metric["one_step_exact"].append(one_step)
                per_metric["free_horizon"].append(float(horizon))
        plast = plasticity(model)  # one probe per arm (deterministic given the model)
        per_metric["plasticity"] = [plast]
        for metric in METRICS:
            vals = per_metric[metric]
            lo, hi = bootstrap_ci(vals, seed=0)
            stats.append(ArmStat(arm, metric, fmean(vals), lo, hi, len(vals)))
    return stats


def _print_summary(stats: list[ArmStat], config: EHStreamConfig) -> None:
    print(f"EH-stream (H15 / HW-4): experience stream vs batch at equal compute "
          f"(ε={config.epsilon}, T={config.eval_steps}):")
    print(f"  {'arm':<18} {'metric':<16} {'mean':>8} {'95% CI':>18}")
    for s in stats:
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        print(f"  {s.arm:<18} {s.metric:<16} {s.mean:>8.3f} {ci:>18}")


def _plot(stats: list[ArmStat], path: Path, config: EHStreamConfig) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "one_step_exact": "one-step exact\n(teacher-forced)",
        "free_horizon": f"free-running H_ε\n(ceiling T={config.eval_steps})",
        "plasticity": "plasticity (HW-4)\n(loss shed on fresh batch)",
    }
    colors = {"batch": "#9bd", "stream+replay": "#16a", "stream-no-replay": "#c66"}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.8 * len(METRICS), 4.2), squeeze=False)
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
        if metric != "free_horizon":
            ax.set_ylim(0, 1.05)
        ax.set_title(titles[metric])
    fig.suptitle("Verisim EH-stream / H15 — does the experience stream beat the batch at equal "
                 "compute? (+ HW-4 plasticity)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH-stream: stream vs batch (H15/HW-4).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh_stream.csv")
    args = parser.parse_args()
    config = EHStreamConfig.from_json_file(args.config) if args.config else EHStreamConfig()
    stats = run_eh_stream(config)
    _print_summary(stats, config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, out.with_suffix(".png"), config)


if __name__ == "__main__":  # pragma: no cover
    main()
