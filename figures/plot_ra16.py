"""Plot the RA16 figure (SPEC-22 H148): the oracle gate as an AI-control protocol.

Two panels from a :class:`~verisim.realagent.control_eval.ControlEvalResult`:

  - **left -- the safety-usefulness frontier.** Sweeping the audit budget traces each protocol
    through (usefulness, safety) space. The trusted model-monitor against a NON-adaptive attacker
    traces the textbook tradeoff curve (more budget buys safety, costs autonomy); against an
    ADAPTIVE attacker it collapses to the bottom edge and reaches safety=1 only at the far-left
    (usefulness 0, audit everything). The oracle coverage gate is a single point in the top-right
    (safety=1, usefulness=1) corner -- the same point under both attackers.
  - **right -- safety vs audit budget.** The monitor's non-adaptive safety rises with the budget;
    its adaptive safety is pinned near 0 until audit-everything (arXiv 2510.09462). The oracle is a
    flat line at safety=1 from its sparse fixed budget on -- invariant to whether the adversary is
    adaptive, because its verdict is read, not predicted.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.control_eval import ControlEvalResult

_COLORS = {
    "none": "#999",
    "model_monitor": "#c44",
    "permission": "#e80",
    "oracle_coverage": "#2a7",
}


def plot_ra16(result: ControlEvalResult, path: str | Path) -> Path:  # pragma: no cover - local plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mm = sorted(result.by_protocol("model_monitor"), key=lambda p: p.budget)
    oc = result.by_protocol("oracle_coverage")[0]
    perm = result.by_protocol("permission")[0]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- left: safety-usefulness frontier ---------------------------------------------------------
    ax_l.plot([p.usefulness for p in mm], [p.safety_random for p in mm], "-o",
              color=_COLORS["model_monitor"], label="model-monitor vs non-adaptive", ms=4)
    ax_l.plot([p.usefulness for p in mm], [p.safety_adaptive for p in mm], "--s",
              color=_COLORS["model_monitor"], alpha=0.65, label="model-monitor vs ADAPTIVE", ms=4)
    ax_l.scatter([perm.usefulness], [perm.safety_random], s=120, color=_COLORS["permission"],
                 marker="^", zorder=4, label="permission denylist (random)")
    ax_l.scatter([oc.usefulness], [oc.safety_adaptive], s=260, color=_COLORS["oracle_coverage"],
                 marker="*", zorder=5, label="oracle coverage (both attackers)")
    ax_l.annotate("the corner:\nsafety=1, usefulness=1\nat a sparse fixed budget",
                  (oc.usefulness, oc.safety_adaptive), textcoords="offset points",
                  xytext=(-150, -38), fontsize=8,
                  arrowprops=dict(arrowstyle="->", color=_COLORS["oracle_coverage"]))
    ax_l.set_xlabel("usefulness  (autonomy on honest work = 1 - benign audit rate)")
    ax_l.set_ylabel("safety  (1 - P[attack succeeds])")
    ax_l.set_xlim(-0.05, 1.08)
    ax_l.set_ylim(-0.05, 1.08)
    ax_l.set_title("The safety-usefulness frontier")
    ax_l.legend(fontsize=7, loc="lower left")
    ax_l.grid(True, alpha=0.3)

    # --- right: safety vs audit budget ------------------------------------------------------------
    ax_r.plot([p.budget for p in mm], [p.safety_random for p in mm], "-o",
              color=_COLORS["model_monitor"], label="model-monitor vs non-adaptive", ms=4)
    ax_r.plot([p.budget for p in mm], [p.safety_adaptive for p in mm], "--s",
              color=_COLORS["model_monitor"], alpha=0.65, label="model-monitor vs ADAPTIVE", ms=4)
    ax_r.axhline(1.0, color=_COLORS["oracle_coverage"], lw=2.2, label="oracle coverage (both)")
    ax_r.axvline(result.surface_density, color=_COLORS["oracle_coverage"], ls=":", alpha=0.7)
    ax_r.annotate(f"oracle's sparse fixed\nbudget = {result.surface_density:.2f}",
                  (result.surface_density, 0.5), textcoords="offset points", xytext=(10, 0),
                  fontsize=8, color=_COLORS["oracle_coverage"])
    ax_r.set_xlabel("audit budget  (fraction of benign actions escalated)")
    ax_r.set_ylabel("safety  (1 - P[attack succeeds])")
    ax_r.set_ylim(-0.05, 1.08)
    ax_r.set_title("Only the oracle's safety is invariant to an adaptive adversary")
    ax_r.legend(fontsize=7, loc="center right")
    ax_r.grid(True, alpha=0.3)

    fig.suptitle(
        "RA16 / H148 -- the oracle gate as an AI-control protocol: a trusted monitor whose verdict "
        "is read, not predicted, so its frontier does not move under an adaptive attacker",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
