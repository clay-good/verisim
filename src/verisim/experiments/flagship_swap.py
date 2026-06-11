"""FL4 -- proposer swap at the flagship: model-invariance of the curve (SPEC-19 §4, H72).

The program's most general claim is that deterministic verification is a *model-agnostic primitive*:
the qualitative shape of `H_ε(ρ)` is governed by the oracle-loop, not the proposer's architecture
(H22, first measured on the EN7 stand-ins). FL4 puts it on the flagship -- run the FL1 four-arm
curve with two materially different proposers on the *same* world and oracle, and ask whether the
*shape* is the same in kind:

  - the **flat** transformer (the frozen FL0 checkpoint, the FL1 headline proposer);
  - the **graph+RSSM** structured arm (the SPEC-10 HS3 / SPEC-14 proposer).

H72 is *supported* when both curves have the same qualitative shape (both floor+cliff, or both with
a knee at the same place) -- the loop governs the shape, competence sets the floor height. It is
*refuted* if a favorable knee appears for one proposer class and not the other at matched per-step
acceptance -- which would narrow the contribution to a fact about one model family, not about
oracle-grounding. Honest caveat, inherited from EN7/H22: the two arms are not competence-matched, so
the evidence is the shared *shape*, not the magnitude (the floor heights differ by construction --
the flat arm's floor dissolved with scale, the graph arm's is pinned at the HS3 wall).

Reuses FL1's `run_flagship_curve` verbatim for both arms (so the comparison is apples-to-apples) and
adds only the graph-arm trainer on the flagship world + the shape verdict. The frozen-LLM proposer
arm (SPEC-5 §7) is the deferred third arm (the LP7/GPU rule). CPU-local; CI runs the smoke instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verisim.net.config import DEFAULT_NET_CONFIG, NetConfig
from verisim.netoracle import ReferenceNetworkOracle
from verisim.netoracle.base import NetOracle

from .flagship import load_checkpoint
from .flagship_curve import (
    CurvePoint,
    FlagshipCurveConfig,
    UncertainNetModel,
    headline_verdict,
    run_flagship_curve,
)


@dataclass(frozen=True)
class SwapConfig:
    """The FL4 instance: the shared curve config + the graph arm's training recipe."""

    curve: FlagshipCurveConfig = field(default_factory=FlagshipCurveConfig)
    graph_d_model: int = 96
    graph_mp_rounds: int = 3
    graph_iters: int = 4000
    graph_train_driver: str = "weighted"
    graph_train_seeds: tuple[int, ...] = (0, 1, 2, 3)
    graph_train_steps_per_traj: int = 60
    model_seed: int = 0

    @staticmethod
    def smoke() -> SwapConfig:
        return SwapConfig(
            curve=FlagshipCurveConfig.smoke(), graph_d_model=32, graph_iters=300,
            graph_train_seeds=(0, 1), graph_train_steps_per_traj=20,
        )


def _train_graph_on(config: SwapConfig, net: NetConfig, oracle: NetOracle) -> UncertainNetModel:
    """Train the graph+RSSM arm on the flagship world (DEFAULT_NET_CONFIG) so the swap is fair."""
    import torch

    from verisim.netmodel import NetVocab
    from verisim.netmodel.graph_model import build_graph_model
    from verisim.netmodel.graph_train import build_graph_dataset, train_graph_model

    torch.set_num_threads(1)
    vocab = NetVocab(net)
    model = build_graph_model(
        vocab, net, d_model=config.graph_d_model, mp_rounds=config.graph_mp_rounds,
        seed=config.model_seed,
    )
    examples = build_graph_dataset(
        oracle, vocab, net, driver=config.graph_train_driver, seeds=config.graph_train_seeds,
        n_steps=config.graph_train_steps_per_traj,
    )
    train_graph_model(model, examples, steps=config.graph_iters, seed=config.model_seed)
    model.net.eval()
    return model


def shape_signature(points: list[CurvePoint]) -> dict[str, Any]:
    """A coarse, magnitude-free descriptor of a curve's *shape* (the H72 comparison unit).

    ``has_knee`` -- does an interior budget (ρ ≤ 0.2) reach ≥80% of the ceiling horizon? (the
    favorable-knee test, SPEC.md §9 H1). ``span`` -- ceiling minus floor (the dynamic range the loop
    has to work with). The shape verdict compares ``has_knee`` across proposers; the floor/ceiling
    magnitudes are reported but deliberately *not* matched (the EN7/H22 caveat).
    """
    v = headline_verdict(points)
    return {
        "floor": v["floor"],
        "ceiling": v["ceiling"],
        "span": v["ceiling"] - v["floor"],
        "has_knee": bool(v["h69_supported"]),
    }


def run_swap(
    config: SwapConfig | None = None, *, checkpoint: str | Path = "runs/flagship/net-l",
    oracle: NetOracle | None = None, flat_model: UncertainNetModel | None = None,
) -> dict[str, list[CurvePoint]]:
    """Run the FL1 curve for both proposers on the flagship world; return per-arm points.

    ``flat_model`` overrides the frozen-checkpoint flat arm (tests pass a freshly-trained smoke arm
    so they need no checkpoint on disk); otherwise the FL0 checkpoint at ``checkpoint`` is loaded.
    """
    config = config or SwapConfig()
    oracle = oracle or ReferenceNetworkOracle()
    net = DEFAULT_NET_CONFIG

    flat = flat_model if flat_model is not None else load_checkpoint(checkpoint).world_model
    graph = _train_graph_on(config, net, oracle)

    return {
        "flat": run_flagship_curve(flat, config.curve, oracle=oracle),
        "graph": run_flagship_curve(graph, config.curve, oracle=oracle),
    }


def h72_verdict(curves: dict[str, list[CurvePoint]]) -> dict[str, Any]:
    """H72: the curve *shape* is the same across proposers (both knee, or both no-knee)."""
    sigs = {arm: shape_signature(pts) for arm, pts in curves.items()}
    knees = {arm: sig["has_knee"] for arm, sig in sigs.items()}
    same_shape = len(set(knees.values())) == 1
    return {
        "shapes": sigs,
        "same_shape": same_shape,
        # H72 supported = the loop governs the shape (same knee/no-knee verdict across proposers)
        "h72_supported": same_shape,
        "knee_per_arm": knees,
    }


def write_csv(curves: dict[str, list[CurvePoint]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        f"{arm},{p.arm},{p.rho:.4f},{p.h_mean:.6f},{p.ci_lo:.6f},{p.ci_hi:.6f},{p.n}"
        for arm, pts in curves.items()
        for p in pts
    ]
    out.write_text("\n".join(["proposer,arm,rho,h_mean,ci_lo,ci_hi,n", *rows]) + "\n")
    return out


def main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="FL4 -- proposer swap at the flagship (SPEC-19).")
    parser.add_argument("--checkpoint", type=str, default="runs/flagship/net-l")
    parser.add_argument("--out", type=str, default="figures/fl4_proposer_swap.csv")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = SwapConfig.smoke() if args.smoke else SwapConfig()
    curves = run_swap(config, checkpoint=args.checkpoint)
    path = write_csv(curves, args.out)
    print(f"wrote curves for {list(curves)} to {path}")
    verdict = h72_verdict(curves)
    for arm, sig in verdict["shapes"].items():
        print(f"  {arm:6s}  floor={sig['floor']:.2f} ceiling={sig['ceiling']:.2f} "
              f"knee={sig['has_knee']}")
    tag = "SUPPORTED" if verdict["h72_supported"] else "not supported"
    print(f"H72 (same shape across proposers): {tag}  {verdict['knee_per_arm']}")


if __name__ == "__main__":  # pragma: no cover
    main()
