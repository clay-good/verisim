"""Plot the UA9 useful-knee curve (SPEC-20 §7, H81) from its committed points.

One panel: the ρ-grounded predictor's content-keyed catch rate vs the oracle-consultation budget ρ,
on the file-integrity task where faithfulness is load-bearing (UA8/H80). The free floor (ρ=0) and
faithful ceiling (ρ=1) are the horizontal anchors; the interior is the curve the program's thesis
predicts and UA2-UA7 never produced (flat, no advantage to recover). The H81 read is whether the
catch rate rises monotonically with ρ (the H76/UA4 mirror) and recovers the ceiling at a budget
markedly below the every-step faithful predictor's (the useful knee, marked).
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ua_host_grounded import KneePoint


def plot_ua9_grounded_knee(  # pragma: no cover
    points: list[KneePoint],
    path: str | Path,
    *,
    knee_frac: float = 0.9,
    title: str = "UA9 — the useful knee: content-keyed faithfulness at budget ρ (SPEC-20, H81)",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = sorted(points, key=lambda p: p.rho)
    rhos = [p.rho for p in pts]
    means = [p.reward for p in pts]
    lo = [p.reward - p.ci_lo for p in pts]
    hi = [p.ci_hi - p.reward for p in pts]
    floor, ceiling = means[0], means[-1]
    ceil_calls = pts[-1].oracle_calls
    knee = next((p for p in pts if p.reward >= knee_frac * ceiling), pts[-1])

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.axhline(floor, color="0.6", ls=":", label=f"free floor (ρ=0): {floor:.2f}")
    ax.axhline(ceiling, color="0.3", ls="--", label=f"faithful ceiling (ρ=1): {ceiling:.2f}")
    ax.errorbar(
        rhos, means, yerr=[lo, hi], color="tab:blue", marker="o", capsize=3,
        label="ρ-grounded predictor",
    )
    ax.scatter(
        [knee.rho], [knee.reward], color="tab:green", zorder=5, s=120, marker="*",
        label=(
            f"useful knee: ρ={knee.rho:.2f}, {knee.oracle_calls:.0f} calls "
            f"(vs {ceil_calls:.0f} for ρ=1)"
        ),
    )

    ax.set_xlabel("oracle-consultation budget ρ")
    ax.set_ylabel("content-keyed catch rate  (true corruptions caught)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
