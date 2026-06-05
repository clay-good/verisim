"""Plot the ED2 equal-dollar-budget figure (SPEC-7 §12, DS7): horizon-vs-oracle-dollar frontiers.

One panel per proposer error mode (`gross` | `subtle`). Each panel plots, per tier policy, the
**faithful-horizon-vs-oracle-dollar frontier** (the Pareto front H17 is really about) -- reading a
vertical line at any dollar budget gives the horizon each policy buys for that budget. The dashed
vertical marks the **quarter budget** ``B/4`` (sub-linear cost) at which the H17 winner and the H18
competitive ratio are read.

The figure makes the mode-dependent verdict visible: in the **gross** panel the cheap metamorphic /
`escalate` frontier sits *above and to the left* of bit-exact (more horizon per dollar -- H17
holds); in the **subtle** panel the cheap tiers are flat at the floor and only bit-exact climbs
(H17's honest negative).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed2 import ED2Result

_COLORS = {
    "metamorphic": "#2ca02c",
    "symbolic": "#ff7f0e",
    "bit_exact": "#d62728",
    "escalate": "#1f77b4",
}


def plot_ed2(result: ED2Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(p["mode"] for p in result.frontier))
    policies = list(dict.fromkeys(p["policy"] for p in result.frontier))
    verdict_by_mode = {v["mode"]: v for v in result.verdict}

    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.8), squeeze=False)
    for ax, mode in zip(axes[0], modes, strict=False):
        for policy in policies:
            pts = sorted(
                (p for p in result.frontier if p["mode"] == mode and p["policy"] == policy),
                key=lambda p: p["rho"],  # ρ-sweep order (a leftward jog = drift contained)
            )
            xs = [p["dollars"] for p in pts]
            ys = [p["h_eps"] for p in pts]
            ax.plot(xs, ys, marker="o", color=_COLORS.get(policy, "#888"),
                    label=policy, linewidth=1.8, markersize=4)
        v = verdict_by_mode.get(mode)
        if v is not None:
            ax.axvline(v["b_quarter"], color="#555", linestyle="--", linewidth=1, alpha=0.7)
            ax.text(v["b_quarter"], ax.get_ylim()[1] * 0.02, "  B/4", fontsize=8, color="#555")
            won = "tiering wins" if v["h17_tiering_wins"] else "bit_exact wins"
            ax.set_title(f"{mode}-error proposer\nH17 @ B/4: {won} ({v['h17_winner']}), "
                         f"H18 ratio={v['competitive_ratio']:.2f}", fontsize=10)
        ax.set_xlabel("cumulative oracle-dollars spent")
        ax.set_ylabel("faithful horizon  H_ε  (steps)")
        ax.legend(title="π_w policy", fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.suptitle("ED2 — equal-dollar-budget: does a cheap/escalate tier buy more horizon per $? "
                 "(H17/H18, SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
