"""Experiment EH6 -- two-oracle grounding: the privilege invariant as a cheap second oracle (H12
host).

The host analogue of the network EN10/H12. The full **state oracle** (§5) verifies the predicted
bundle bit-for-bit; the symbolic **privilege second-oracle** (:mod:`verisim.hostoracle.invariant`)
answers only one formally-checkable security question -- "does any non-root process hold a protected
fd?" -- far more cheaply. EH6 asks the same three things EN10 did, teacher-forced over a
denial-heavy
workload:

  - **non-redundant rate** -- steps where the invariant flags an error the *full* prediction got
  right
    (full bits-to-correct == 0 but invariant bits-to-correct > 0). **0 by construction** -- the
    invariant is a pure function of the state, so it catches nothing a bit-exact prediction misses.
    Redundant *for verification*.
  - **invariant-sufficient rate** -- steps where the full prediction is *wrong*
    but the model still gets the **security verdict** right (invariant bits-to-correct == 0). The
    decision-relevant payoff: the model can be trusted on the privilege question far more often
    than on
    the whole state.
  - **consult-bits ratio** -- invariant answer size (procs × protected-paths) / full-state size: how
    much cheaper the security consult is.

The H12 reframing: the privilege oracle is *redundant* as a verification signal but a **cheaper,
decision-sufficient** consult for the security question a defender actually asks -- the
tiered-oracle
premise (SPEC-7). Whatever it shows is a datum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from verisim.host.config import DEFAULT_HOST_CONFIG
from verisim.host.delta import apply
from verisim.host.state import HostState
from verisim.hostloop.model import HostModel
from verisim.hostloop.observe import full_bits
from verisim.hostmetrics.bits import bits_to_correct
from verisim.hostoracle.invariant import invariant_bits, invariant_bits_to_correct
from verisim.hostoracle.reference import ReferenceHostOracle

from .eh1 import EH1Config, eval_actions


@dataclass(frozen=True)
class EH6Config:
    base: EH1Config = field(default_factory=EH1Config)
    arm: str = "factored"  # the proposer to evaluate (the more faithful arm, per EH4)
    max_pid: int = 64
    graph_d_model: int = 64
    graph_mp_rounds: int = 3
    graph_iters: int = 800
    graph_batch: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any]) -> EH6Config:
        b = EH6Config()
        return EH6Config(
            base=EH1Config.from_dict(d.get("base", {})),
            arm=d.get("arm", b.arm),
            max_pid=d.get("max_pid", b.max_pid),
            graph_d_model=d.get("graph_d_model", b.graph_d_model),
            graph_mp_rounds=d.get("graph_mp_rounds", b.graph_mp_rounds),
            graph_iters=d.get("graph_iters", b.graph_iters),
            graph_batch=d.get("graph_batch", b.graph_batch),
        )

    @staticmethod
    def from_json_file(path: str | Path) -> EH6Config:
        return EH6Config.from_dict(json.loads(Path(path).read_text()))


def _train_arm(config: EH6Config, oracle: ReferenceHostOracle) -> HostModel:
    from verisim.hostmodel import HostVocab, NeuralHostWorldModel
    from verisim.hostmodel.graph_model import build_host_graph_model
    from verisim.hostmodel.graph_train import build_host_graph_dataset, train_host_graph_model

    from .eh1 import train_model as train_flat

    base = config.base
    host = DEFAULT_HOST_CONFIG
    vocab = HostVocab(host, max_pid=config.max_pid)
    if config.arm == "flat":
        return NeuralHostWorldModel(train_flat(base, vocab, oracle, host), vocab)
    examples = build_host_graph_dataset(
        oracle, vocab, host, driver=base.train_driver, seeds=base.train_seeds,
        n_steps=base.train_steps_per_traj,
    )
    model = build_host_graph_model(
        vocab, host, max_pid=config.max_pid, d_model=config.graph_d_model,
        mp_rounds=config.graph_mp_rounds, seed=base.model_seed,
    )
    train_host_graph_model(
        model, examples, steps=config.graph_iters, lr=base.lr,
        batch_size=config.graph_batch, seed=base.model_seed,
    )
    return model


def run_eh6(
    config: EH6Config | None = None, *, oracle: ReferenceHostOracle | None = None
) -> dict[str, float]:
    """Train the arm; return the H12 metrics (non-redundant / invariant-sufficient / bits ratio)."""
    config = config or EH6Config()
    base = config.base
    oracle = oracle or ReferenceHostOracle()
    host = DEFAULT_HOST_CONFIG
    model = _train_arm(config, oracle)

    total = 0
    non_redundant = 0  # full exact but invariant wrong -- 0 by construction
    invariant_sufficient = 0  # full wrong but invariant right -- the decision-relevant payoff
    full_wrong = 0
    bits_ratios: list[float] = []
    for _difficulty, driver in base.difficulties.items():
        for seed in base.eval_seeds:
            state = HostState.initial()
            for action in eval_actions(oracle, host, driver, seed, base.eval_steps):
                pred_delta = model.predict_delta(state, action)
                result = oracle.step(state, action)
                pred_state = apply(state, pred_delta)
                full_btc = bits_to_correct(pred_delta, result.delta)
                inv_btc = invariant_bits_to_correct(pred_state, result.state)
                total += 1
                if full_btc == 0.0 and inv_btc > 0:
                    non_redundant += 1
                if full_btc > 0.0:
                    full_wrong += 1
                    if inv_btc == 0:
                        invariant_sufficient += 1
                fb = full_bits(result.state)
                bits_ratios.append(invariant_bits(result.state) / fb if fb else 0.0)
                state = result.state  # teacher-forced
    return {
        "non_redundant_rate": non_redundant / total if total else 0.0,
        "invariant_sufficient_rate": invariant_sufficient / full_wrong if full_wrong else 0.0,
        "full_wrong_rate": full_wrong / total if total else 0.0,
        "consult_bits_ratio": fmean(bits_ratios) if bits_ratios else 0.0,
        "n_steps": float(total),
    }


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="EH6 two-oracle (privilege vs full, H12).")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--out", type=str, default="figures/eh6_two_oracle.csv")
    args = parser.parse_args()
    config = EH6Config.from_json_file(args.config) if args.config else EH6Config()
    r = run_eh6(config)
    cols = (
        "non_redundant_rate", "invariant_sufficient_rate", "full_wrong_rate",
        "consult_bits_ratio",
    )
    print(f"EH6 two-oracle ({config.arm} arm), {int(r['n_steps'])} steps:")
    for c in cols:
        print(f"  {c:<26} {r[c]:.4f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("metric,value\n" + "\n".join(f"{c},{r[c]:.6f}" for c in cols) + "\n")
    print(f"wrote {out}")
    _plot(r, out.with_suffix(".png"), config.arm)


def _plot(r: dict[str, float], path: Path, arm: str) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
    rates = ("non_redundant_rate", "invariant_sufficient_rate", "full_wrong_rate")
    labels = (
        "non-redundant\n(H12: ~0)", "invariant-sufficient\n(decision payoff)",
        "full-delta\nwrong",
    )
    ax1.bar(range(len(rates)), [r[m] for m in rates], color=["#c66", "#393", "#9bd"])
    ax1.set_xticks(range(len(rates)))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("privilege second-oracle vs full state")
    ax2.bar([0, 1], [1.0, r["consult_bits_ratio"]], color=["#9bd", "#393"])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["full state", "privilege\ninvariant"], fontsize=8)
    ax2.set_ylabel("consult bits (fraction of full)")
    ax2.set_title("consult cost: the security question is far cheaper")
    fig.suptitle(f"Verisim EH6 / H12 — privilege invariant is redundant but cheaper ({arm})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":  # pragma: no cover
    main()
