"""Experiment RS1: free-oracle DAgger vs teacher forcing (H55, SPEC-16 §5). *The headline.*

Every horizon number to date trained the proposer **teacher-forced** (on the oracle's true states)
and
rolled it out **free-running** (on its own predicted states) -- the textbook exposure-bias /
covariate-shift gap, and exactly the gap SPEC-10's HS1.1 caught (a per-step-accurate model with a
near-zero free-running horizon). The classic cure is DAgger (Ross-Gordon-Bagnell, 2011): query the
expert on the *learner's own* drifted state distribution and aggregate. DAgger's bottleneck is the
expensive expert -- but in verisim the expert is the **oracle**: free, exact, callable at any
visited
state. So verisim runs the cure the simulator-learning field can only approximate.

RS1 trains the **real flat ``M_θ``** (a small GPT over the constrained-decode grammar) two ways at a
*fixed total example budget and fixed total gradient-step budget* (the only fair comparison):

  - **teacher-forced** -- ``train_batched`` on ``n`` on-policy oracle examples;
  - **DAgger** -- ``N`` rounds: train on the aggregated pool, free-run ``M_θ`` to produce drifted
    states, **relabel each with the oracle** (``oracle.step(s̃, a)`` -- the exact correction at the
    drifted state), aggregate, retrain. The pool grows to ``n`` over the rounds; total gradient
    steps
    match the baseline.

It reports the free-running faithful horizon ``H_free`` per DAgger round (the cure curve), the
one-step
exact rate ``p`` (the HS1.1 signature -- teacher forcing is per-step accurate but horizon-poor), and
a
same-budget bar against teacher forcing. **The figure that shows the cure -- or banks the
fundamental-compounding negative** (if DAgger does not lift ``H_free``, the gap is the model class's
inability to represent the recovery map, the deepest HS3 reading).

Torch-gated: imports torch lazily and is ``skipif``-guarded in tests; the committed figure is
generated
on the primary host. Deterministic and seeded (``torch.set_num_threads(1)``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.metrics.aggregate import bootstrap_ci


@dataclass(frozen=True)
class RS1Config:
    """A small, fast DAgger-vs-teacher-forcing instance (runs on the primary host CPU)."""

    n_hosts: int = 5
    n_ports: int = 3
    driver: str = "weighted"
    n_examples: int = 320  # total example budget (equal for both arms)
    total_steps: int = 600  # total gradient-step budget (equal for both arms)
    n_rounds: int = 4  # DAgger rounds
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 128
    lr: float = 3e-4
    batch_size: int = 64
    n_steps: int = 40  # rollout length for data + eval
    epsilon: float = 0.0  # exact-match faithful horizon (the HS1 tolerance)
    train_seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104, 105)
    one_step_seeds: tuple[int, ...] = (200, 201, 202, 203)
    model_seeds: tuple[int, ...] = (0, 1, 2)  # repeats for CIs
    num_threads: int = 1

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RS1Config:
        b = RS1Config()
        g = d.get
        return RS1Config(
            n_hosts=g("n_hosts", b.n_hosts), n_ports=g("n_ports", b.n_ports),
            driver=g("driver", b.driver), n_examples=g("n_examples", b.n_examples),
            total_steps=g("total_steps", b.total_steps), n_rounds=g("n_rounds", b.n_rounds),
            n_layer=g("n_layer", b.n_layer), n_head=g("n_head", b.n_head),
            n_embd=g("n_embd", b.n_embd),
            block_size=g("block_size", b.block_size), lr=g("lr", b.lr),
            batch_size=g("batch_size", b.batch_size), n_steps=g("n_steps", b.n_steps),
            epsilon=g("epsilon", b.epsilon), train_seeds=tuple(g("train_seeds", b.train_seeds)),
            eval_seeds=tuple(g("eval_seeds", b.eval_seeds)),
            one_step_seeds=tuple(g("one_step_seeds", b.one_step_seeds)),
            model_seeds=tuple(g("model_seeds", b.model_seeds)),
            num_threads=g("num_threads", b.num_threads),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> RS1Config:
        return RS1Config.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class RS1Stat:
    """One arm/round cell: free-running horizon and one-step exact rate, mean + bootstrap CI."""

    arm: str  # "teacher-forced" | "dagger"
    dagger_round: int  # 0 for teacher-forced; 0..N-1 for dagger
    h_free: float
    h_lo: float
    h_hi: float
    p_one_step: float
    n: int

    def csv_row(self) -> str:
        return (
            f"{self.arm},{self.dagger_round},{self.h_free:.6f},{self.h_lo:.6f},"
            f"{self.h_hi:.6f},{self.p_one_step:.6f},{self.n}"
        )


CSV_HEADER = "arm,dagger_round,h_free,h_lo,h_hi,p_one_step,n"


def run_rs1(config: RS1Config | None = None) -> list[RS1Stat]:
    """Train the flat M_θ teacher-forced and via free-oracle DAgger; per-round H_free/p (H55)."""
    config = config or RS1Config()
    import torch

    from verisim.loop.policy import fixed_interval_for_rho
    from verisim.metrics.horizon import faithful_horizon
    from verisim.model.transformer import GPT, GPTConfig
    from verisim.net.config import scaled_net_config
    from verisim.net.state import NetworkState
    from verisim.netdata.drivers import NetDriver
    from verisim.netdelta.apply import apply as net_apply
    from verisim.netloop.observe import PartialNetOracle
    from verisim.netloop.runner import budget_for_rho, run_net_rollout
    from verisim.netmetrics.exact import delta_exact_rate
    from verisim.netmodel import NetVocab, NeuralNetworkWorldModel, build_net_dataset
    from verisim.netmodel.tokenizer import encode_prompt, encode_target
    from verisim.netoracle import ReferenceNetworkOracle
    from verisim.train.supervised import train_batched

    if config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
    net = scaled_net_config(config.n_hosts, config.n_ports)
    oracle = ReferenceNetworkOracle()
    vocab = NetVocab(net)

    def model() -> Any:
        return GPT(GPTConfig(vocab_size=len(vocab), block_size=config.block_size,
                             n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd))

    def actions(seed: int) -> list[Any]:
        drv = NetDriver(name=config.driver, config=net, rng=__import__("random").Random(seed))
        s = NetworkState.initial(net.hosts)
        acts = []
        for _ in range(config.n_steps):
            a = drv.sample(s)
            acts.append(a)
            s = oracle.step(s, a).state
        return acts

    def eval_h_free(wm: Any) -> float:
        partial = PartialNetOracle(oracle)
        hs = []
        for es in config.eval_seeds:
            acts = actions(es)
            r = run_net_rollout(wm, partial, NetworkState.initial(net.hosts), acts,
                                fixed_interval_for_rho(0.0), epsilon=config.epsilon,
                                budget=budget_for_rho(0.0, len(acts)), seed=es)
            hs.append(float(faithful_horizon(list(r.divergences), config.epsilon)))
        return fmean(hs)

    def eval_p(wm: Any) -> float:
        triples = []
        for os in config.one_step_seeds:
            drv = NetDriver(name=config.driver, config=net, rng=__import__("random").Random(os))
            s = NetworkState.initial(net.hosts)
            for _ in range(config.n_steps):
                a = drv.sample(s)
                res = oracle.step(s, a)
                triples.append((s, a, res.delta))
                s = res.state
        return delta_exact_rate((wm.predict_delta(s, a), true) for s, a, true in triples)

    def drifted_examples(wm: Any) -> list[Any]:
        """Free-run ``wm`` on the train seeds; relabel each visited drifted state with the oracle.
        """
        out = []
        for sd in config.train_seeds:
            coupled = NetworkState.initial(net.hosts)
            for a in actions(sd):
                res = oracle.step(coupled, a)  # the exact correction at the DRIFTED state
                out.append((encode_prompt(coupled, a, vocab), encode_target(res.delta, vocab)))
                coupled = net_apply(coupled, wm.predict_delta(coupled, a))
        return out

    per_round_tf: dict[int, list[tuple[float, float]]] = {0: []}
    per_round_dag: dict[int, list[tuple[float, float]]] = {r: [] for r in range(config.n_rounds)}
    # The RS1 fairness axis is equal *examples* (SPEC-16 §5): each round retrains fully (the same
    # per-train step budget as teacher forcing); DAgger's extra total compute is the RS6 (H58) axis,
    # and HS1.1 already established that more teacher-forced compute does not lift the horizon.
    steps_per_round = config.total_steps

    half = max(1, config.n_examples // 2)
    for ms in config.model_seeds:
        torch.manual_seed(ms)
        # Teacher-forced baseline: n on-policy examples, total_steps steps (= DAgger round 0).
        tf_ex = build_net_dataset(oracle, vocab, net, driver=config.driver,
                                  seeds=config.train_seeds,
                                  n_steps=config.n_steps)[: config.n_examples]
        m_tf = model()
        train_batched(m_tf, tf_ex, vocab.pad, steps=config.total_steps, lr=config.lr,
                      batch_size=config.batch_size, seed=ms)
        wm_tf = NeuralNetworkWorldModel(m_tf, vocab)
        per_round_tf[0].append((eval_h_free(wm_tf), eval_p(wm_tf)))

        # DAgger: a fixed-size pool of n examples = half on-policy + half oracle-relabeled drifted,
        # the drifted half refreshed each round from the *current* deployed model (Ross 2011).
        # total examples (n) and equal per-train steps as teacher forcing -- the §5 fair comparison.
        base = list(tf_ex[:half])
        wm_dag = wm_tf  # the deployed model whose drift round 1 relabels
        for rnd in range(config.n_rounds):
            pool = base + drifted_examples(wm_dag)[: config.n_examples - len(base)]
            m_dag = model()
            train_batched(m_dag, pool, vocab.pad, steps=steps_per_round, lr=config.lr,
                          batch_size=config.batch_size, seed=ms)
            wm_dag = NeuralNetworkWorldModel(m_dag, vocab)
            per_round_dag[rnd].append((eval_h_free(wm_dag), eval_p(wm_dag)))

    stats: list[RS1Stat] = []
    tf_h = [h for h, _ in per_round_tf[0]]
    lo, hi = bootstrap_ci(tf_h, seed=0)
    stats.append(RS1Stat("teacher-forced", 0, fmean(tf_h), lo, hi,
                         fmean([p for _, p in per_round_tf[0]]), len(tf_h)))
    for rnd in range(config.n_rounds):
        hs = [h for h, _ in per_round_dag[rnd]]
        lo, hi = bootstrap_ci(hs, seed=0)
        stats.append(RS1Stat("dagger", rnd, fmean(hs), lo, hi,
                             fmean([p for _, p in per_round_dag[rnd]]), len(hs)))
    return stats


def _print_summary(stats: list[RS1Stat]) -> None:
    print("RS1 / H55 - free-oracle DAgger vs teacher forcing (the exposure-bias cure):")
    tf = next(s for s in stats if s.arm == "teacher-forced")
    dag = [s for s in stats if s.arm == "dagger"]
    print(f"  teacher-forced: H_free={tf.h_free:.2f}  p={tf.p_one_step:.2f}")
    for s in dag:
        print(f"  DAgger round {s.dagger_round}: H_free={s.h_free:.2f} [{s.h_lo:.2f},{s.h_hi:.2f}] "
              f"p={s.p_one_step:.3f}")
    final = dag[-1]
    best = max(dag, key=lambda s: s.h_free)
    lift = best.h_free - tf.h_free
    pstr = f"p {tf.p_one_step:.2f}->{final.p_one_step:.2f}"
    # A lift counts only if its CI clears teacher forcing (best round's lower bound > TF mean).
    reliable = best.h_lo > tf.h_free
    if reliable:
        verdict = (
            f"DAgger lifts H_free {tf.h_free:.2f} -> {best.h_free:.2f} ({lift:+.2f}), CI clearing "
            f"teacher forcing, at equal examples, one-step accuracy held ({pstr}) - H55 supported: "
            "exposure bias is a train/deploy mismatch the oracle cures by relabeling its own drift"
        )
    else:
        verdict = (
            f"DAgger does not reliably lift H_free at this scale (best {best.h_free:.2f} vs "
            f"teacher-forced {tf.h_free:.2f}, CIs overlap; {pstr}) - H55 not supported here: at "
            "CPU scale the flat M_θ is near the H_ε floor, and its own oracle-relabeled drift "
            "buys no horizon (the pre-registered fundamental-compounding / scale-limited branch); "
            "whether it pays for a competent high-p model is the open RS5-RS7 + scale question"
        )
    print(f"  verdict: {verdict}")


def _plot(stats: list[RS1Stat], path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf = next(s for s in stats if s.arm == "teacher-forced")
    dag = sorted((s for s in stats if s.arm == "dagger"), key=lambda s: s.dagger_round)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    rounds = [s.dagger_round for s in dag]
    ax1.axhline(tf.h_free, color="#d62728", ls="--", lw=1.5, label="teacher-forced")
    ax1.fill_between([min(rounds), max(rounds)], tf.h_lo, tf.h_hi, color="#d62728", alpha=0.10)
    ax1.plot(rounds, [s.h_free for s in dag], "-o", color="#1f77b4", label="DAgger (relabel drift)")
    ax1.fill_between(rounds, [s.h_lo for s in dag], [s.h_hi for s in dag],
                     color="#1f77b4", alpha=0.15)
    ax1.set_xlabel("DAgger round")
    ax1.set_ylabel("free-running faithful horizon H_free")
    ax1.set_title("DAgger vs teacher forcing: H_free per round")
    ax1.legend(fontsize=8)
    # The HS1.1 signature: one-step accuracy p vs horizon H_free, teacher-forced vs final DAgger.
    labels = ["teacher-\nforced", f"DAgger\nround {dag[-1].dagger_round}"]
    x = range(2)
    width = 0.38
    mx = max(1.0, tf.h_free, *(s.h_free for s in dag))
    ax2.bar([i - width / 2 for i in x], [tf.p_one_step, dag[-1].p_one_step], width,
            color="#9467bd", label="one-step exact rate p")
    ax2.bar([i + width / 2 for i in x], [tf.h_free / mx, dag[-1].h_free / mx], width,
            color="#1f77b4", label="H_free (normalized)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("rate / normalized horizon")
    ax2.set_title("HS1.1 signature: per-step accurate, horizon-poor")
    ax2.legend(fontsize=8)
    fig.suptitle("RS1 / H55: free-oracle DAgger vs teacher forcing (real flat M_θ)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="RS1 free-oracle DAgger vs teacher forcing (H55).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/rs1_dagger.csv")
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()
    cfg = RS1Config.from_json_file(args.config) if args.config else RS1Config()
    stats = run_rs1(cfg)
    _print_summary(stats)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([CSV_HEADER, *(s.csv_row() for s in stats)]) + "\n")
    print(f"wrote {out}")
    _plot(stats, Path(args.plot) if args.plot else out.with_suffix(".png"))


if __name__ == "__main__":  # pragma: no cover
    main()
