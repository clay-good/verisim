"""Plot the ED2-learned figure (SPEC-7 §12, DS7): the real-model horizon-vs-oracle-dollar frontier.

One panel (one trained model, so no error-mode dial -- contrast ED2's two synthetic panels). It
plots, per tier policy, the **faithful-horizon-vs-oracle-dollar frontier** for the learned flat
`M_θ`: reading a vertical line at any dollar budget gives the horizon each policy buys for that
budget. The dashed vertical marks the **quarter budget** ``B/4`` (sub-linear cost) at which the H17
winner and the H18 competitive ratio are read.

The figure makes the real-model verdict visible -- the **honest inverse of ED2's `gross` panel**:
the constrained decoder removes out-of-vocab errors, so the learned model lives in ED2's `subtle`
regime, where the cheap metamorphic / symbolic frontiers stay flat at the floor and only bit-exact
climbs. Tiering does not win per equal dollar for a grammar-constrained learned model -- you pay the
bit-exact tier its errors actually need.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed2_learned import ED2LearnedResult

_COLORS = {
    "metamorphic": "#2ca02c",
    "symbolic": "#ff7f0e",
    "bit_exact": "#d62728",
    "escalate": "#1f77b4",
}


def plot_ed2_learned(result: ED2LearnedResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = list(dict.fromkeys(p["policy"] for p in result.frontier))
    v = result.verdict

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.8))
    for policy in policies:
        pts = sorted(
            (p for p in result.frontier if p["policy"] == policy),
            key=lambda p: p["rho"],  # ρ-sweep order (a leftward jog = drift contained)
        )
        xs = [p["dollars"] for p in pts]
        ys = [p["h_eps"] for p in pts]
        ax.plot(xs, ys, marker="o", color=_COLORS.get(policy, "#888"),
                label=policy, linewidth=1.8, markersize=4)
    if v:
        ax.axvline(v["b_quarter"], color="#555", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(v["b_quarter"], ax.get_ylim()[1] * 0.02, "  B/4", fontsize=8, color="#555")
        won = "tiering wins" if v["h17_tiering_wins"] else "bit_exact wins"
        ax.set_title(f"learned M_θ (real errors)\nH17 @ B/4: {won} ({v['h17_winner']}), "
                     f"H18 ratio={v['competitive_ratio']:.2f}", fontsize=10)
    ax.set_xlabel("cumulative oracle-dollars spent")
    ax.set_ylabel("faithful horizon  H_ε  (steps)")
    ax.legend(title="π_w policy", fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle("ED2-learned — equal-dollar-budget on the real learned M_θ: "
                 "does a cheap/escalate tier buy more horizon per $? (H17/H18, SPEC-7)")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    return out
