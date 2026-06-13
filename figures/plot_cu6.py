"""Plot the CU6 figure (SPEC-22 §5, H98): the replanning agent and the capability/safety tension.

Two panels from a :class:`~verisim.acd.closed_loop_replan.CU6Result`:

  - **left -- replanning is capability, and for a free agent capability is danger.** Task success
    (green) and harm (red) vs the consultation budget ρ, with the replanner (solid) far above the
    one-shot agent (dashed) on success -- it recovers the goals the one-shot abandons. But at ρ=0
    the replanner's harm is *above* the one-shot's (the shaded gap, the persistence penalty): its
    retry loop searches the model's blind spots. Both harms fall to 0 as ρ grows -- verification
    closes the amplification gap.
  - **right -- only a verified agent is both capable and safe.** The four agents in (harm, success)
    space: one-shot/free, one-shot/oracle, replanner/free, replanner/oracle. The free replanner is
    the most *capable* free agent and also the most *dangerous*; only the **oracle replanner**
    reaches the capable-and-safe corner (harm 0, success 1). The arrow is the replanner's ρ-path --
    the budget moving it from the dangerous corner to the safe one.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.closed_loop_replan import CU6Result, cells_for


def plot_cu6(result: CU6Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    one = cells_for(result, "one_shot")
    rep = cells_for(result, "replanner")
    rhos = [c.rho for c in rep]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))

    ax_l.plot(rhos, [c.success_rate for c in rep], "-o", color="#2ca02c", lw=2.3, ms=7,
              label="replanner success")
    ax_l.plot(rhos, [c.success_rate for c in one], "--", color="#2ca02c", lw=1.6, alpha=0.7,
              label="one-shot success")
    ax_l.plot(rhos, [c.harm_rate for c in rep], "-s", color="#d62728", lw=2.3, ms=7,
              label="replanner harm")
    ax_l.plot(rhos, [c.harm_rate for c in one], "--", color="#d62728", lw=1.6, alpha=0.7,
              label="one-shot harm")
    ax_l.fill_between(rhos, [c.harm_rate for c in one], [c.harm_rate for c in rep],
                      color="#d62728", alpha=0.10)
    ax_l.set_xlabel("ρ  (the agent's oracle-consultation budget)")
    ax_l.set_ylabel("rate over the goal battery")
    ax_l.set_title(f"replanning is capability — for a free agent, capability is danger  "
                   f"(φ={result.phi})", fontsize=9.2)
    ax_l.set_ylim(-0.03, 1.05)
    ax_l.legend(fontsize=8.2, loc="center right")

    # the four corners in (harm, success) space + the replanner's ρ-path
    ax_r.plot([c.harm_rate for c in rep], [c.success_rate for c in rep], "-", color="#d62728",
              lw=1.4, alpha=0.5, zorder=1)
    corners = [
        (one[0], "one-shot / free", "#9467bd", "v"),
        (one[-1], "one-shot / oracle", "#9467bd", "o"),
        (rep[0], "replanner / free (capable + DANGEROUS)", "#d62728", "v"),
        (rep[-1], "replanner / oracle (capable + safe)", "#2ca02c", "*"),
    ]
    for cell, label, color, marker in corners:
        ax_r.scatter([cell.harm_rate], [cell.success_rate], s=200 if marker == "*" else 120,
                     color=color, marker=marker, zorder=3, edgecolors="white", linewidths=1.0,
                     label=label)
    ax_r.annotate("", xy=(rep[-1].harm_rate, rep[-1].success_rate),
                  xytext=(rep[0].harm_rate, rep[0].success_rate),
                  arrowprops={"arrowstyle": "->", "color": "#d62728", "alpha": 0.6, "lw": 1.5})
    ax_r.text(rep[len(rep) // 2].harm_rate + 0.005, rep[len(rep) // 2].success_rate - 0.07,
              "ρ ↑", color="#d62728", fontsize=9)
    ax_r.set_xlabel("harm rate  (executed a dangerous route)")
    ax_r.set_ylabel("task success rate  (reached the goal)")
    ax_r.set_title("only a verified agent is both capable and safe", fontsize=9.5)
    ax_r.set_ylim(0.0, 1.05)
    ax_r.legend(fontsize=7.8, loc="center right")

    fig.suptitle(
        "CU6 / H98: the replanning agent — persistence amplifies a free agent's harm "
        "(it searches its own gate's blind spots); only verification makes capability safe",
        fontsize=9.4, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
