"""Plot the CU21 figure (SPEC-22 H114): the unified target -- one rule across all three worlds.

Two panels from a :class:`~verisim.acd.unified_targeting.UnifiedResult`:

  - **left -- one safe-and-cheap corner across four dangers.** Each world contributes four points in
    (calls, adversarial breach) space: the uniform knee (●, gameable -- high adversarial breach),
    the model self-targeting (×, fails), the full oracle (■, safe but maximal cost), and the unified
    covering target (★, the bottom-left corner: zero adversarial breach at a small fraction of the
    full-oracle cost). Color = world. The message: the *same* rule lands every world's star in the
    same safe-and-cheap corner -- one rule, not four tricks.
  - **right -- the coverage boundary.** For the two worlds where another world's target is carried
    over as a shortcut (segmentation: the CU10 ``connect``; distributed: the genesis ``write``),
    the covering target's adversarial breach (green, ≈0) against the non-covering shortcut's (red,
    leaks). The un-gameability is a theorem of coverage; break coverage and you buy false security.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.unified_targeting import UnifiedResult, _uniform_knee

_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")  # one per world (in arm order)


def plot_cu21(result: UnifiedResult, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 5.1))

    # left: one safe-and-cheap corner across four dangers. Every non-target schedule sits on the
    # gameable line (adv 1.0) or at the full-oracle's maximal cost; only the unified target stars
    # land in the bottom-left corner -- the same corner, in every world.
    for arm, color in zip(result.arms, _COLORS, strict=False):
        knee = _uniform_knee(arm)
        ax_l.plot(knee.mean_calls, knee.adversarial_breach, "o", color=color, ms=10, alpha=0.9,
                  markeredgecolor="black", markeredgewidth=0.4)
        ax_l.plot(arm.model.mean_calls, arm.model.adversarial_breach, "X", color=color, ms=12,
                  alpha=0.9, markeredgecolor="black", markeredgewidth=0.4)
        ax_l.plot(arm.full_oracle.mean_calls, arm.full_oracle.adversarial_breach, "s", color=color,
                  ms=10, alpha=0.9, markeredgecolor="black", markeredgewidth=0.4)
        ax_l.plot(arm.target.mean_calls, arm.target.adversarial_breach, "*", color=color, ms=22,
                  zorder=6, markeredgecolor="black", markeredgewidth=0.6)
    ax_l.axhline(1.0, color="#d62728", lw=1.0, ls=":", alpha=0.5)
    ax_l.text(30, 1.03, "the gameable line (adversarial breach = 1.0)", fontsize=7.6,
              color="#d62728", ha="center")
    ax_l.add_patch(Rectangle((0, -0.05), 7.5, 0.12, color="#2ca02c", alpha=0.08, zorder=0))
    ax_l.annotate("the unified target *\nzero adversarial breach\nat ~3-4 calls (every world)",
                  xy=(4.0, 0.0), xytext=(11.0, 0.28), fontsize=8.6, color="#2ca02c",
                  arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3))
    ax_l.annotate("model self-targeting:\nfails at 0 calls", xy=(0.3, 1.0), xytext=(3.0, 0.78),
                  fontsize=7.8, color="#555",
                  arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    ax_l.annotate("uniform knee:\ngameable at 24 calls", xy=(24, 1.0), xytext=(26, 0.74),
                  fontsize=7.8, color="#555",
                  arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    ax_l.annotate("full oracle:\nsafe at 48 calls", xy=(48, 0.0), xytext=(34, 0.20),
                  fontsize=7.8, color="#555",
                  arrowprops=dict(arrowstyle="->", color="#888", lw=1.0))
    ax_l.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_l.set_ylabel("adversarial breach rate  (worst-case timing)")
    ax_l.set_xlim(-2, 52)
    ax_l.set_ylim(-0.08, 1.12)
    ax_l.set_title("one model-free rule -> one safe-and-cheap corner, in every world", fontsize=9.2)
    schedule_legend = [
        Line2D([], [], marker="*", color="#555", ms=15, ls="", markeredgecolor="black",
               label="unified target (covering)"),
        Line2D([], [], marker="o", color="#555", ms=8, ls="", label="uniform knee (gameable)"),
        Line2D([], [], marker="X", color="#555", ms=9, ls="", label="model self-targeting (fails)"),
        Line2D([], [], marker="s", color="#555", ms=8, ls="", label="full oracle (max cost)"),
    ]
    world_legend = [
        Line2D([], [], marker="o", color=c, ms=8, ls="", label=a.world_name)
        for a, c in zip(result.arms, _COLORS, strict=False)
    ]
    leg1 = ax_l.legend(handles=schedule_legend, fontsize=7.6, loc="upper left")
    ax_l.add_artist(leg1)
    ax_l.legend(handles=world_legend, fontsize=7.6, loc="center right", title="world / danger")

    # right: the coverage boundary -- covering target vs non-covering shortcut
    shortcut_arms = [a for a in result.arms if a.shortcut is not None]
    labels = [a.world_name for a in shortcut_arms]
    x = list(range(len(shortcut_arms)))
    width = 0.36
    cover_adv = [a.target.adversarial_breach for a in shortcut_arms]
    short_adv = [a.shortcut.adversarial_breach if a.shortcut else 0.0 for a in shortcut_arms]
    ax_r.bar([i - width / 2 for i in x], cover_adv, width, color="#2ca02c",
             label="covering target (un-gameable)")
    ax_r.bar([i + width / 2 for i in x], short_adv, width, color="#d62728",
             label="non-covering shortcut (leaks)")
    for i, a in enumerate(shortcut_arms):
        ax_r.scatter([i - width / 2], [0.0], marker="_", s=320, color="#2ca02c", zorder=5)
        ax_r.text(i - width / 2, 0.07, f"{a.target_name}\ncovers=True\n(breach 0.0)",
                  ha="center", va="bottom", fontsize=7.0, color="#1a661a", fontweight="bold")
        ax_r.text(i + width / 2, short_adv[i] / 2, f"{a.shortcut_name}\ncovers=False\n(breach 1.0)",
                  ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, fontsize=9.0)
    ax_r.set_ylabel("adversarial breach rate")
    ax_r.set_ylim(0.0, 1.15)
    ax_r.set_title("the boundary: coverage is what makes the rule safe", fontsize=9.2)
    ax_r.legend(fontsize=8.0, loc="upper center")

    fig.suptitle(
        "CU21 / H114: the unified target — every per-world defense (network exfil, host corrupt, "
        "segmentation, distributed staleness) is ONE\nmodel-free rule: consult iff the action's on "
        "the danger's surface. Un-gameability is a THEOREM of coverage, not a per-world quirk.",
        fontsize=9.0, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
