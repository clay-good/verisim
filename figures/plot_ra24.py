"""Plot the RA24 figure (SPEC-22 H156): a neural compositional adversary vs the RA18 resolver.

Two panels:
  - **left -- discovery.** Cumulative *true silent misses* vs oracle calls on the pre-RA24 resolver
    (the printf-format-escape soundness bug live). The neural policy's curve is far steeper than
    blind search and the single-transform (RA23) control: a learned compositional adversary
    finds the
    soundness violation the hand-run red team missed, fast.
  - **right -- hardened soundness + frontier.** Cumulative tiered reward vs oracle calls on the
    hardened resolver. Silent misses are 0 for every arm (the invariant holds under the adversary);
    the reward is the folder-incompleteness frontier (string-resolvable forms the resolver ABSTAINs
    on), which the neural policy maps far more efficiently than the controls.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.neural_proposer import RunResult


def plot(d_neural: RunResult, d_blind: RunResult, d_single: RunResult,
         h_neural: RunResult, h_blind: RunResult, h_single: RunResult,
         path: str | Path) -> Path:  # pragma: no cover - needs matplotlib
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: discovery of the soundness bug -----------------------------------------------------
    calls = range(1, d_neural.budget + 1)
    ax_l.plot(calls, d_neural.silent_curve, color="#c33", lw=2,
              label=f"neural (compositional): {d_neural.silent_miss}")
    ax_l.plot(calls, d_blind.silent_curve, color="#777", lw=2, ls="--",
              label=f"blind uniform: {d_blind.silent_miss}")
    ax_l.plot(calls, d_single.silent_curve, color="#39a", lw=2, ls=":",
              label=f"single-transform (RA23): {d_single.silent_miss}")
    ax_l.set_xlabel("oracle calls (= verifiable-reward queries)")
    ax_l.set_ylabel("cumulative true silent misses found")
    ax_l.set_title("Discovery: learned adversary finds the RA18 printf-escape\nsoundness bug fast")
    ax_l.legend(fontsize=9, loc="upper left")

    # --- right: hardened reward (frontier), soundness held ----------------------------------------
    ax_r.plot(calls, h_neural.reward_curve, color="#2a7", lw=2,
              label=f"neural: {h_neural.reward_per_call:.2f}/call")
    ax_r.plot(calls, h_blind.reward_curve, color="#777", lw=2, ls="--",
              label=f"blind uniform: {h_blind.reward_per_call:.2f}/call")
    ax_r.plot(calls, h_single.reward_curve, color="#39a", lw=2, ls=":",
              label=f"single-transform: {h_single.reward_per_call:.2f}/call")
    ax_r.set_xlabel("oracle calls")
    ax_r.set_ylabel("cumulative tiered reward (frontier-mapping)")
    ax_r.set_title("Hardened resolver: silent miss = 0 (sound);\n"
                   "neural maps the frontier efficiently")
    ax_r.legend(fontsize=9, loc="upper left")
    ax_r.text(0.98, 0.05,
              f"silent miss: neural {h_neural.silent_miss} / blind {h_blind.silent_miss} / "
              f"single {h_single.silent_miss}\ndistinct frontier comps: "
              f"neural {h_neural.distinct_frontier_compositions} vs single "
              f"{h_single.distinct_frontier_compositions} (uniform-capped)",
              transform=ax_r.transAxes, ha="right", va="bottom", fontsize=8,
              bbox={"boxstyle": "round", "fc": "#eef", "ec": "#99c"})

    fig.suptitle("RA24: a neural compositional adversary trained by the exact oracle's tiered "
                 "hole-verdict (the two arcs meet, scaled)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
