"""Plot the RA23 figure (SPEC-22 H155): the learned adversarial proposer (oracle-as-reward).

Two panels:
  - **left -- sample efficiency.** Cumulative coverage holes found vs oracle calls, learned vs the
    blind uniform proposer over the same operator space. The learned curve is steeper: the oracle's
    hole-verdict, used as a verifiable reward, concentrates proposals on hole-rich regions.
  - **right -- what the policy learned.** The final per-transform proposal probability. Mass on the
    coverable literal class collapses (the CEGIS repair removed its reward); mass concentrates on
    the indirection transforms that still realize holes -- including quote_splice, absent from RA22.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.learned_proposer import INDIRECTION, RunResult


def plot_ra23(learned: RunResult, rnd: RunResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: hole-discovery efficiency ----------------------------------------------------------
    calls = range(1, learned.budget + 1)
    ax_l.plot(calls, learned.hole_curve, color="#2a7", lw=2,
              label=f"learned (oracle-as-reward): {learned.holes_found}")
    ax_l.plot(calls, rnd.hole_curve, color="#999", lw=2, ls="--",
              label=f"blind uniform: {rnd.holes_found}")
    ax_l.set_xlabel("oracle calls (= verifiable-reward queries)")
    ax_l.set_ylabel("cumulative coverage holes found")
    ax_l.set_title("Sample efficiency: the learned adversary finds more holes per call")
    ax_l.legend(fontsize=9, loc="upper left")

    # --- right: the learned per-transform proposal distribution -----------------------------------
    transforms = list(learned.final_transform_probs)
    probs = [learned.final_transform_probs[t] for t in transforms]
    colors = ["#c44" if t not in INDIRECTION else "#2a7" for t in transforms]
    ax_r.bar(range(len(transforms)), probs, color=colors)
    ax_r.set_xticks(range(len(transforms)))
    ax_r.set_xticklabels(transforms, fontsize=8, rotation=20)
    ax_r.set_ylabel("final proposal probability")
    ax_r.set_ylim(0, max(probs) * 1.25 + 0.02)
    ax_r.set_title("Policy concentrates on indirection (literal class covered -> no reward)")
    for i, p in enumerate(probs):
        ax_r.text(i, p + 0.005, f"{p:.2f}", ha="center", fontsize=8)
    ax_r.text(0.98, 0.95, "red = coverable (literal)\ngreen = irreducible residual",
              transform=ax_r.transAxes, ha="right", va="top", fontsize=8)

    fig.suptitle("RA23: a learned adversarial proposer trained by the exact oracle's hole-verdict "
                 "(verifier-as-reward)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:  # pragma: no cover - local entrypoint
    from verisim.realagent.learned_proposer import LearnedProposer, RandomProposer, run_cegis

    learned = run_cegis(LearnedProposer(seed=0))
    rnd = run_cegis(RandomProposer(seed=0))
    root = Path(__file__).resolve().parent.parent
    out = plot_ra23(learned, rnd, root / "figures" / "ra23_learned_proposer.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
