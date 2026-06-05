"""Plot the SY1 system-oracle differential figure (SPEC-11 §3, §5): the agreement table +
the head-to-head H_ε(ρ) overlay -- *the figure that retires W1*.

Two panels from one :class:`~verisim.experiments.sy1.SY1Result`:

  - **left -- per-driver agreement.** Bit-exact agreement rate between the reference oracle
    and a real ``/bin/sh`` along driver trajectories, with bootstrap CIs. The structure-
    building drivers (``structural``/``trivial``) -- the regime v0 is designed to model --
    sit at a flat **1.000**; the destructive drivers sit lower, their gap fully accounted for
    by the named v0 modeling boundaries (the stacked divergence classes).
  - **right -- the curve overlay.** The prime-directive ``H_ε(ρ)`` run with the reference
    oracle and again with the system oracle on the structural grammar. The two curves are
    *indistinguishable* (gap = 0 at every ρ): substituting a real computer for the from-
    scratch model leaves the faithful-horizon curve unchanged where v0 claims fidelity.

The platform is stamped on the figure (the macOS-first principle, SPEC-11 §2.5): the headline
1.000 is platform-independent; the destructive-driver divergence mix is platform-stamped.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.sy1 import SY1Result
from verisim.oracle.differential import BOUNDARY_CLASSES

_CLASS_COLOR = {
    "root_protection": "#d62728",
    "overwrite_policy": "#ff7f0e",
    "permission_enforcement": "#9467bd",
    "self_subtree": "#8c564b",
    "residual": "#000000",
}


def plot_sy1(result: SY1Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))

    # left: per-driver agreement with the divergence-class decomposition stacked below
    drivers = [r["driver"] for r in result.tier2]
    rates = [r["rate"] for r in result.tier2]
    lo = [r["rate"] - r["ci_lo"] for r in result.tier2]
    hi = [r["ci_hi"] - r["rate"] for r in result.tier2]
    x = range(len(drivers))
    ax_l.bar(x, rates, color="#2ca02c", alpha=0.85, label="agree (bit-exact)")
    ax_l.errorbar(x, rates, yerr=[lo, hi], fmt="none", ecolor="#333", capsize=4, lw=1)
    # stack the boundary classes as the remaining (1 - agree) mass
    bottoms = list(rates)
    for cls in (*BOUNDARY_CLASSES, "residual"):
        heights = []
        for r in result.tier2:
            frac = r["classes"].get(cls, 0) / r["n"] if r["n"] else 0.0
            heights.append(frac)
        if any(h > 0 for h in heights):
            ax_l.bar(x, heights, bottom=bottoms, color=_CLASS_COLOR.get(cls, "#777"),
                     alpha=0.8, label=cls.replace("_", " "))
            bottoms = [b + h for b, h in zip(bottoms, heights, strict=True)]
    ax_l.set_xticks(list(x))
    ax_l.set_xticklabels(drivers, rotation=15, fontsize=9)
    ax_l.set_ylabel("fraction of transitions")
    ax_l.set_ylim(0, 1.02)
    ax_l.axhline(1.0, color="#2ca02c", lw=0.8, ls=":")
    ax_l.set_title("SY1: reference vs. real /bin/sh — per-driver agreement (H27)")
    ax_l.legend(loc="lower left", fontsize=7, ncol=2)

    # right: the head-to-head H_eps(rho) overlay
    rhos = [p["rho"] for p in result.tier3]
    h_ref = [p["h_ref"] for p in result.tier3]
    h_sys = [p["h_sys"] for p in result.tier3]
    ax_r.plot(rhos, h_ref, marker="o", color="#1f77b4", lw=2.2, label="reference oracle")
    ax_r.plot(rhos, h_sys, marker="x", color="#d62728", lw=1.4, ls="--",
              label="system oracle (real /bin/sh)")
    ax_r.set_xlabel("oracle-consultation budget  ρ")
    ax_r.set_ylabel("faithful horizon  H_ε(ρ)  (steps)")
    max_gap = max((p["gap"] for p in result.tier3), default=0.0)
    ax_r.set_title(f"SY1 Tier-3: the curve is oracle-invariant (max gap = {max_gap:.3f})")
    ax_r.legend(loc="upper left", fontsize=8)
    ax_r.grid(True, alpha=0.3)

    headline = (f"structure-building agreement = {result.modeled_agreement:.3f}  •  "
                f"residual = {result.residual_fraction:.3f}  •  platform = {result.platform}")
    fig.suptitle(headline, y=1.00, fontsize=9.5)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
