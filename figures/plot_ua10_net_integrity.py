"""Plot the UA10 network flow-integrity cross-world confirmation (SPEC-20 §7, H82).

Two panels from the committed points:

  - **left -- the content-keyed positive** (the UA8 mirror): the faithful predictor (oracle rollout)
    vs the free predictor (raw network `M_θ` rollout) over the workload horizon. Faithful stays at
    1.0; free collapses as the cumulative flow drift compounds -- the gap widening with horizon.
  - **right -- the useful knee** (the UA9 mirror): the ρ-grounded predictor's catch rate vs the
    oracle-consultation budget ρ, recovering the faithful ceiling at a budget well below the
    every-step faithful predictor's.

Together they confirm the SPEC-20 boundary law and the useful knee are cross-world, not
host-specific.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ua_net_integrity import IntegrityPoint, KneePoint


def plot_ua10_net_integrity(  # pragma: no cover
    horizon_points: list[IntegrityPoint],
    knee_points: list[KneePoint],
    path: str | Path,
    *,
    knee_frac: float = 0.9,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5))

    # --- left: the positive (faithful vs free over horizon) ---
    def _series(name: str) -> tuple[list[int], list[float], list[float], list[float]]:
        pts = sorted((p for p in horizon_points if p.predictor == name), key=lambda p: p.horizon)
        return (
            [p.horizon for p in pts], [p.reward for p in pts],
            [p.reward - p.ci_lo for p in pts], [p.ci_hi - p.reward for p in pts],
        )

    for name, color, marker in (("faithful", "tab:green", "o"), ("free", "tab:red", "s")):
        hs, means, lo, hi = _series(name)
        axl.errorbar(hs, means, yerr=[lo, hi], color=color, marker=marker, capsize=3,
                     label=f"{name} predictor")
    axl.set_xlabel("workload horizon (steps)")
    axl.set_ylabel("flow-integrity catch rate")
    axl.set_ylim(-0.05, 1.05)
    axl.set_title("the content-keyed positive (UA8 mirror):\nfaithful >> free, widens with horizon",
                  fontsize=9)
    axl.legend(fontsize=8, loc="center right")
    axl.grid(alpha=0.3)

    # --- right: the useful knee (catch vs rho) ---
    kpts = sorted(knee_points, key=lambda p: p.rho)
    rhos = [p.rho for p in kpts]
    means = [p.reward for p in kpts]
    lo = [p.reward - p.ci_lo for p in kpts]
    hi = [p.ci_hi - p.reward for p in kpts]
    floor, ceiling = means[0], means[-1]
    ceil_calls = kpts[-1].oracle_calls
    knee = next((p for p in kpts if p.reward >= knee_frac * ceiling), kpts[-1])
    axr.axhline(floor, color="0.6", ls=":", label=f"free floor (ρ=0): {floor:.2f}")
    axr.axhline(ceiling, color="0.3", ls="--", label=f"faithful ceiling (ρ=1): {ceiling:.2f}")
    axr.errorbar(rhos, means, yerr=[lo, hi], color="tab:blue", marker="o", capsize=3,
                 label="ρ-grounded predictor")
    axr.scatter([knee.rho], [knee.reward], color="tab:green", zorder=5, s=120, marker="*",
                label=f"useful knee: ρ={knee.rho:.2f}, {knee.oracle_calls:.0f} calls "
                      f"(vs {ceil_calls:.0f} for ρ=1)")
    axr.set_xlabel("oracle-consultation budget ρ")
    axr.set_ylabel("flow-integrity catch rate")
    axr.set_ylim(-0.05, 1.05)
    axr.set_title("the useful knee (UA9 mirror):\ncatch recovered at sub-linear budget", fontsize=9)
    axr.legend(fontsize=8, loc="lower right")
    axr.grid(alpha=0.3)

    fig.suptitle(
        "UA10 — network flow-integrity: boundary law + useful knee are cross-world (SPEC-20, H82)",
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
