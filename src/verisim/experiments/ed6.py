"""ED6 -- counterfactual grounding for the distributed world (H5, SPEC-7 §10.1, DS8).

The distributed oracle is a *total* deterministic DES, so from any visited cluster state ``s`` it
returns the exact next state of an **alternative** admin/fault action ``a'`` the trajectory did not
take, ``O(s, a')`` -- a free counterfactual branch (re-run from ``(seed, t)`` with one fault
flipped, §10.1). This is the capability physical-domain causal world models structurally lack: you
can re-run the cluster from step ``t`` with one *fault* changed and read back the **true**
alternative state. ED6 asks the **H5** question for the distributed world (the host's
:mod:`.eh6_counterfactual` analogue): does *training* the flat DS4 `M_θ` on these free
counterfactual branches improve its prediction of **interventions** at test time?

The distributed world makes the question operationally sharp. The interventions that matter for an
SRE/defender are the **near-miss partitions and split-brain scenarios** (§17 Q7): "what if this node
*had* been partitioned at step ``t``?", "what if this coordinator had crashed mid-write?" -- the
fault ops (``partition``/``crash``/``heal``/``restart``) that set the distributed world's *medium*,
the hidden state every prior experiment (ED3's ``ReplicasOnly`` collapse, ED5's consistency-vs-bit
gap) turns on. A model trained on a light-fault factual workload rarely sees the medium change; the
counterfactual branches are nothing but medium changes. So the counterfactual driver is fault-heavy
(``adversarial`` with the fault dial turned up: the denial-rich near-misses).

To separate the counterfactual *signal* from mere data volume, three arms train the **same** flat
`M_θ` for the **same** number of steps, differing only in the training set's *composition* (matched
example count):

  - **trajectory** -- the base ``uniform`` trajectory dataset (one action per visited state).
  - **trajectory-more** -- *more* trajectory data (extra seeds) to the same count: a volume control.
  - **+counterfactual** -- the base trajectory plus oracle counterfactual fault branches (``k``
    sampled alternative fault interventions per visited state), to the same count.

Evaluation is on **held-out** states: for each, sample intervention fault actions ``a'`` and score
the model against the oracle two ways -- **intervention exact** (the robust headline: did
``apply(s, Δ̂)`` reproduce the true next cluster state, bit-for-bit, ``divergence == 0``?) and
**medium recall** (the operationally-decisive readout: of the interventions truth *changes the
medium* -- a new partition split or a crashed node, the split-brain precondition -- does the model
predict the resulting partition/down structure exactly?). *H5 supported* iff ``+counterfactual``
beats *both* trajectory arms on the headline -- the lift being structure, not volume. The honest
negative (§10.1): counterfactual data adds nothing over factual. Reported whichever way it falls.

CPU, deterministic, torch-gated like every learned-arm experiment (the ``[model]`` extra). Unlike
the small-dataset flat-arm experiments (ED1-learned / ED4-fault) that fit on the full-batch
:func:`~verisim.train.supervised.train_supervised`, ED6 is the first distributed experiment with a
matched-*volume* dataset (the ``+counterfactual`` arm is ``1+k`` times the base), so all three arms
train on the minibatched :func:`~verisim.train.supervised.train_batched` (the K2 loop, SPEC-2.1 §6)
-- the same trainer the host EH6 uses for the analogue matched-count comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.dist.action import DistAction
from verisim.dist.config import DEFAULT_DIST_CONFIG, DistConfig
from verisim.dist.delta import DistDelta, apply
from verisim.dist.state import DistributedState
from verisim.distdata.drivers import DistDriver
from verisim.distmetrics.divergence import divergence
from verisim.distoracle import ReferenceDistOracle
from verisim.distoracle.base import DistOracle
from verisim.metrics.aggregate import bootstrap_ci

ARMS = ("trajectory", "trajectory-more", "+counterfactual")
METRICS = ("intervention_exact", "medium_recall")

#: The medium: the (partition structure, crashed-node set) the fault ops change -- the distributed
#: world's hidden state (ED3/ED5). ``object`` so the tuple is hashable/comparable across states.
Medium = tuple[frozenset[frozenset[str]], frozenset[str]]


def _medium(state: DistributedState) -> Medium:
    return (frozenset(frozenset(g) for g in state.partitions), frozenset(state.down))


@dataclass(frozen=True)
class ED6Config:
    """Small, fast counterfactual-grounding instance. Scale up via ``configs/ed6.json``."""

    dist: DistConfig = DEFAULT_DIST_CONFIG
    train_driver: str = "uniform"
    intervention_driver: str = "adversarial"  # fault-heavy near-miss interventions (§17 Q7)
    intervention_fault_prob: float = 0.85  # turn the fault dial up: the medium-changing near-misses
    intervention_partition_bias: float = 0.75
    train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    train_steps_per_traj: int = 40
    k_counterfactual: int = 4  # counterfactual fault branches sampled per visited state
    # model (mirrors ED1-learned / ED4-fault)
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 512
    train_iters: int = 700
    batch_size: int = 64  # the matched-volume datasets need minibatching (the K2 loop, SPEC-2.1 §6)
    lr: float = 3e-3
    model_seed: int = 0
    max_int: int = 256
    # evaluation: held-out states x sampled interventions
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103)
    eval_steps: int = 24
    m_interventions: int = 6  # test interventions sampled per held-out state

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ED6Config:
        b = ED6Config()
        return ED6Config(
            train_driver=d.get("train_driver", b.train_driver),
            intervention_driver=d.get("intervention_driver", b.intervention_driver),
            intervention_fault_prob=d.get("intervention_fault_prob", b.intervention_fault_prob),
            intervention_partition_bias=d.get(
                "intervention_partition_bias", b.intervention_partition_bias
            ),
            train_seeds=tuple(d.get("train_seeds", b.train_seeds)),
            train_steps_per_traj=d.get("train_steps_per_traj", b.train_steps_per_traj),
            k_counterfactual=d.get("k_counterfactual", b.k_counterfactual),
            n_layer=d.get("n_layer", b.n_layer),
            n_head=d.get("n_head", b.n_head),
            n_embd=d.get("n_embd", b.n_embd),
            block_size=d.get("block_size", b.block_size),
            train_iters=d.get("train_iters", b.train_iters),
            batch_size=d.get("batch_size", b.batch_size),
            lr=d.get("lr", b.lr),
            model_seed=d.get("model_seed", b.model_seed),
            max_int=d.get("max_int", b.max_int),
            eval_seeds=tuple(d.get("eval_seeds", b.eval_seeds)),
            eval_steps=d.get("eval_steps", b.eval_steps),
            m_interventions=d.get("m_interventions", b.m_interventions),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> ED6Config:
        return ED6Config.from_dict(json.loads(Path(path).read_text()))


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

# An intervention test case: (state, action, true next state).
EvalCase = tuple[DistributedState, DistAction, DistributedState]


def _intervention_driver(config: ED6Config, seed: int) -> DistDriver:
    return DistDriver(
        name=config.intervention_driver, config=config.dist, rng=random.Random(seed),
        fault_prob=config.intervention_fault_prob,
        partition_bias=config.intervention_partition_bias,
    )


def _intervention_cases(
    oracle: DistOracle, config: ED6Config, seed: int
) -> list[EvalCase]:
    """Roll a held-out trajectory; sample ``m`` interventions per state -> labeled test cases."""
    traj = DistDriver(name=config.train_driver, config=config.dist, rng=random.Random(seed))
    iv = _intervention_driver(config, seed ^ 0x1217E)
    state = DistributedState.initial(config.dist)
    cases: list[EvalCase] = []
    for _ in range(config.eval_steps):
        for _k in range(config.m_interventions):
            a = iv.sample(state)
            cases.append((state, a, oracle.step(state, a).state))
        state = oracle.step(state, traj.sample(state)).state  # advance on the true trajectory
    return cases


def _score(model: Any, cases: list[EvalCase]) -> dict[str, float]:
    """Intervention exact + medium recall for one arm over one held-out seed's cases."""
    exact = 0
    medium_total = 0
    medium_hit = 0
    for state, action, true_next in cases:
        delta: DistDelta = model.predict_delta(state, action)
        pred_next = apply(state, delta)
        if divergence(pred_next, true_next) == 0.0:
            exact += 1
        # the decisive class: interventions that actually change the medium (a split/crash/heal)
        if _medium(true_next) != _medium(state):
            medium_total += 1
            if _medium(pred_next) == _medium(true_next):
                medium_hit += 1
    return {
        "intervention_exact": exact / len(cases) if cases else 1.0,
        "medium_recall": medium_hit / medium_total if medium_total else 1.0,
    }


def _build_datasets(config: ED6Config, oracle: DistOracle, vocab: Any) -> dict[str, list[Any]]:
    """The three matched-count training sets (trajectory / trajectory-more / +counterfactual)."""
    from verisim.distmodel import encode_prompt, encode_target

    def example(state: DistributedState, action: DistAction) -> Any:
        delta = oracle.step(state, action).delta
        return (encode_prompt(state, action, vocab), encode_target(delta, vocab))

    def trajectory(driver: str, seeds: tuple[int, ...]) -> list[Any]:
        out: list[Any] = []
        for seed in seeds:
            drv = DistDriver(name=driver, config=config.dist, rng=random.Random(seed))
            state = DistributedState.initial(config.dist)
            for _ in range(config.train_steps_per_traj):
                action = drv.sample(state)
                out.append(example(state, action))
                state = oracle.step(state, action).state
        return out

    base_n = len(config.train_seeds) * config.train_steps_per_traj
    target_n = base_n * (1 + config.k_counterfactual)  # the matched example-count budget

    traj = trajectory(config.train_driver, config.train_seeds)
    more_seeds = tuple(range(-(-target_n // config.train_steps_per_traj)))  # ceil division
    traj_more = trajectory(config.train_driver, more_seeds)[:target_n]

    cf: list[Any] = []
    for seed in config.train_seeds:
        traj_drv = DistDriver(name=config.train_driver, config=config.dist, rng=random.Random(seed))
        cf_drv = _intervention_driver(config, seed ^ 0xC0FFEE)
        state = DistributedState.initial(config.dist)
        for _ in range(config.train_steps_per_traj):
            for _k in range(config.k_counterfactual):
                cf.append(example(state, cf_drv.sample(state)))  # free counterfactual fault branch
            state = oracle.step(state, traj_drv.sample(state)).state
    counterfactual = (traj + cf)[:target_n]

    return {"trajectory": traj, "trajectory-more": traj_more, "+counterfactual": counterfactual}


def run_ed6(config: ED6Config | None = None, *, oracle: DistOracle | None = None) -> list[ArmStat]:
    """Train the three arms, then score intervention exact + medium recall, held out (H5)."""
    import torch

    from verisim.distmodel import DistVocab, NeuralDistWorldModel
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.train.supervised import train_batched

    config = config or ED6Config()
    oracle = oracle or ReferenceDistOracle(config.dist)
    torch.set_num_threads(1)  # process-reproducibility (the E1/EN1 discipline)

    vocab = DistVocab(config.dist, max_int=config.max_int)
    datasets = _build_datasets(config, oracle, vocab)

    # held-out intervention test cases (shared across arms)
    cases_by_seed = {s: _intervention_cases(oracle, config, s) for s in config.eval_seeds}

    # the model's block_size must cover the longest example across every arm
    needed = max(len(p) + len(t) for ds in datasets.values() for p, t in ds) + 8
    block_size = max(config.block_size, needed)

    stats: list[ArmStat] = []
    for arm in ARMS:
        torch.manual_seed(config.model_seed)
        model = GPT(
            GPTConfig(
                vocab_size=len(vocab), block_size=block_size,
                n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
            )
        )
        train_batched(
            model, datasets[arm], vocab.pad, steps=config.train_iters, lr=config.lr,
            batch_size=config.batch_size, seed=config.model_seed,
        )
        world_model = NeuralDistWorldModel(model, vocab)
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in config.eval_seeds:
            scored = _score(world_model, cases_by_seed[seed])
            for metric in METRICS:
                per_metric[metric].append(scored[metric])
        for metric, vals in per_metric.items():
            lo, hi = bootstrap_ci(vals, seed=0)
            stats.append(ArmStat(arm, metric, fmean(vals), lo, hi, len(vals)))
    return stats


def h5_supported(stats: list[ArmStat], metric: str = "intervention_exact") -> bool:
    """H5 holds iff ``+counterfactual`` beats *both* trajectory arms on ``metric`` (structure)."""
    by_arm = {s.arm: s.mean for s in stats if s.metric == metric}
    cf = by_arm.get("+counterfactual", 0.0)
    return cf > by_arm.get("trajectory", 1.0) and cf > by_arm.get("trajectory-more", 1.0)


def _print_summary(stats: list[ArmStat]) -> None:
    print("ED6 counterfactual grounding — held-out intervention prediction (bootstrap CIs):")
    print(f"  {'arm':<16} {'metric':<22} {'mean':>8} {'95% CI':>18}")
    for s in stats:
        ci = f"[{s.ci_lo:.3f}, {s.ci_hi:.3f}]"
        print(f"  {s.arm:<16} {s.metric:<22} {s.mean:>8.3f} {ci:>18}")
    for metric in METRICS:
        verdict = "H5 SUPPORTED" if h5_supported(stats, metric) else "no lift over factual"
        print(f"  [{metric}] {verdict}")


def write_csv(stats: list[ArmStat], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="ED6 distributed counterfactual grounding (H5).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/ed6.csv")
    parser.add_argument("--plot", type=str, default="figures/ed6.png")
    args = parser.parse_args()
    config = ED6Config.from_json_file(args.config) if args.config else ED6Config()
    stats = run_ed6(config)
    _print_summary(stats)
    path = write_csv(stats, args.out)
    print(f"wrote {path}")
    try:
        from figures.plot_ed6 import plot_ed6

        plot_ed6(stats, args.plot)
        print(f"wrote {args.plot}")
    except Exception as exc:  # pragma: no cover - plotting is optional/local
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":  # pragma: no cover
    main()
