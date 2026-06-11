"""Plot the FL1 flagship faithful-horizon curve (SPEC-19 §3) from its committed points.

One panel, the headline figure: faithful horizon `H_ε` vs oracle budget `ρ`, four arms on the *same*
frozen flagship checkpoint (figures-from-records, SPEC-2 §7.3):

  - **floor** (`ρ=0`) and **ceiling** (`ρ=1`) as horizontal anchors;
  - **fixed-ρ** -- the naive clock baseline;
  - **composed π_c** -- the conformal+speculative policy (the program's best).

The H69 read is whether the *composed* curve reaches the 80%-of-ceiling line (dashed) at `ρ ≤ 0.2`;
the honest-negative read is whether composed and fixed both hug the floor until `ρ → 1`.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.flagship_curve import CurvePoint


def plot_flagship_curve(  # pragma: no cover
    points: list[CurvePoint],
    path: str | Path,
    *,
    title: str = "FL1 — the flagship H_ε(ρ) curve on a real trained M_θ (SPEC-19, H69)",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floor = next((p.h_mean for p in points if p.arm == "floor"), 0.0)
    ceiling = next((p.h_mean for p in points if p.arm == "ceiling"), 0.0)

    def _series(arm: str) -> tuple[list[float], list[float], list[float], list[float]]:
        pts = sorted((p for p in points if p.arm == arm), key=lambda p: p.rho)
        return (
            [p.rho for p in pts], [p.h_mean for p in pts],
            [p.h_mean - p.ci_lo for p in pts], [p.ci_hi - p.h_mean for p in pts],
        )

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.axhline(floor, color="0.6", ls=":", label=f"floor (ρ=0): {floor:.1f}")
    ax.axhline(ceiling, color="0.3", ls="--", label=f"ceiling (ρ=1): {ceiling:.1f}")
    ax.axhline(
        0.8 * ceiling, color="tab:green", ls="--", alpha=0.5, label="80% of ceiling (H69 target)"
    )

    for arm, color, marker in (("fixed", "tab:orange", "s"), ("composed", "tab:blue", "o")):
        rhos, means, lo, hi = _series(arm)
        if rhos:
            ax.errorbar(
                rhos, means, yerr=[lo, hi], color=color, marker=marker, capsize=3, label=arm
            )

    ax.set_xlabel("oracle-consultation budget ρ")
    ax.set_ylabel("faithful horizon H_ε  (steps)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
