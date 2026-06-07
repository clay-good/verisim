"""Plot the ED12 learned-arm figure (SPEC-7 §5.4): partial-observation projections on the real M_θ.

Two panels from one :class:`~verisim.experiments.ed12_learned.ED12LearnedResult`:

  - **left — free-running horizons.** The trained flat `M_θ`'s bit / observable / consistency
    free-running horizons (ρ=0), with bootstrap-CI whiskers. The structural dominance ``bit ≤
    observable`` holds on every rollout; the absolute values are small (the flat free-runner's low
    floor, inherited from ED1-learned), so the gaps are directional — the clean signal is Panel B.
  - **right — teacher-forced per-step accuracy.** The fraction of steps the model predicts correctly
    under each projection: ``bit`` (every fact), ``observable`` (right at the probe — the in-flight
    medium forgiven), ``consistency`` (right per-object view). The rates order ``bit ≤ observable ≤
    consistency`` and quantify which of the real model's per-step errors each projection forgives.
"""

from __future__ import annotations

from pathlib import Path

from verisim.experiments.ed12_learned import ED12LearnedResult

_COLOR = {"bit": "#d62728", "obs": "#1f77b4", "cons": "#2ca02c"}


def plot_ed12_learned(result: ED12LearnedResult, path: str | Path) -> Path:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    h = result.horizons

    # left: free-running horizons (bit / observable / consistency) with CIs
    labels = ["bit", "observable\n(probe)", "consistency"]
    vals = [h["bit_h"], h["obs_h"], h["cons_h"]]
    los = [h["bit_h"] - h["bit_lo"], h["obs_h"] - h["obs_lo"], h["cons_h"] - h["cons_lo"]]
    his = [h["bit_hi"] - h["bit_h"], h["obs_hi"] - h["obs_h"], h["cons_hi"] - h["cons_h"]]
    colors = [_COLOR["bit"], _COLOR["obs"], _COLOR["cons"]]
    ax_l.bar(labels, vals, yerr=[los, his], capsize=5, color=colors, alpha=0.85, width=0.6)
    for i, v in enumerate(vals):
        ax_l.text(i, v + max(his) * 0.1 + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax_l.set_ylabel("free-running faithful horizon  H_ε  (steps)")
    dom = h["observable_dominates_bit"]
    ax_l.set_title(f"Panel A — free-running horizons (real flat M_θ)\n"
                   f"bit ≤ observable holds every rollout: {dom} (small absolutes)", fontsize=10)
    ax_l.grid(True, axis="y", alpha=0.3)

    # right: teacher-forced per-step accuracy
    a = result.accuracy
    labels2 = ["bit", "observable\n(probe)", "consistency"]
    rates = [a["bit"], a["observable"], a["consistency"]]
    bars = ax_r.bar(labels2, rates, color=colors, alpha=0.85, width=0.6)
    for bar, r in zip(bars, rates, strict=True):
        ax_r.text(bar.get_x() + bar.get_width() / 2, r + 0.01, f"{r:.2f}",
                  ha="center", va="bottom", fontsize=10)
    ax_r.set_ylim(0, max(rates) * 1.25 + 0.05 if rates else 1.0)
    ax_r.set_ylabel("teacher-forced per-step correct rate")
    ax_r.set_title("Panel B — which errors each projection forgives\n"
                   f"bit ≤ observable ≤ consistency (n={a['steps']} steps)", fontsize=10)
    ax_r.grid(True, axis="y", alpha=0.3)

    fig.suptitle("ED12 (learned M_θ) — the probe + consistency projections forgive a real model's "
                 "in-flight / placement errors (SPEC-7 §5.4)", y=1.01, fontsize=9.5)
    fig.tight_layout()
    out = Path(path)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
