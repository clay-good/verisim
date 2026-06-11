"""Plot the HFL1 host flagship `H_ε(ρ)` curve (SPEC-19, H84) from its committed points.

One panel, the host analogue of FL1: faithful horizon `H_ε` vs oracle budget `ρ`, four arms on the
*same* frozen host checkpoint -- floor (`ρ=0`) and ceiling (`ρ=1`) as anchors, the fixed-interval
clock baseline, and the composed decode-entropy-triggered policy (the program's smart scheduler).
The H84 read is whether the composed curve sits *above* the fixed clock at equal budget on the
harder host world (`H_free` ~9 vs the network's ~18) -- the cross-world confirmation of the FL1 win.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.host_flagship_curve import CurvePoint


def plot_hfl1_host_curve(  # pragma: no cover
    points: list[CurvePoint],
    path: str | Path,
    *,
    title: str = "HFL1 — the host flagship H_ε(ρ) curve: smart scheduling, harder world (H84)",
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

    for arm, color, marker in (("fixed", "tab:orange", "s"), ("composed", "tab:blue", "o")):
        rhos, means, lo, hi = _series(arm)
        if rhos:
            ax.errorbar(
                rhos, means, yerr=[lo, hi], color=color, marker=marker, capsize=3, label=arm
            )

    ax.set_xlabel("oracle-consultation budget ρ")
    ax.set_ylabel("faithful horizon H_ε  (steps)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
