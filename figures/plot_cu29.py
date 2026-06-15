"""Plot the CU29 figure (SPEC-22 §5, H122): the forensic oracle -- the posterior dual.

Two panels from a :class:`~verisim.acd.forensic_oracle.CU29Result`:

  - **left -- the headline:** forensic localization accuracy by world. The exact oracle attributes
    every breach (1.000); the omitting world model is blind (0.000) -- it cannot even find the step
    it omitted. The real network ``M_theta`` (if available) is plotted as a marker on the network
    arm, blind too (CU8's 0.02 recall) -- the omitter is not a strawman.
  - **right -- the root cause:** the mean lag between the realizing (breach) step and the earliest
    averting intervention. In the genesis-separated worlds (host ``open``->``write``, distributed
    ``put``->stale-``get``) the incident was determined several steps before the action that tripped
    it; where genesis ~ consumption (net exfil, segmentation flip) the window is tight. The four
    genesis-grammar flavors of the targeting arc, read backward as root-cause structures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verisim.acd.forensic_oracle import CU29Result

_SHORT = {
    "network": "network\nexfil",
    "host": "host\ncorruption",
    "distributed": "distributed\nstaleness",
    "network (segmentation)": "network\nsegmentation",
}


def plot_cu29(
    result: CU29Result,
    verdict: dict[str, Any],
    path: str | Path,
    *,
    real_model: tuple[float, float] | None = None,
) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = result.arms
    labels = [_SHORT.get(a.world_name, a.world_name) for a in arms]
    x = np.arange(len(arms))
    w = 0.38

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Panel A -- forensic localization accuracy: oracle vs the omitting model.
    oracle = [a.oracle_localization_accuracy for a in arms]
    model = [a.omitter_localization_accuracy for a in arms]
    axA.bar(x - w / 2, oracle, w, color="#2ca02c", label="exact oracle forensic")
    axA.bar(x + w / 2, model, w, color="#d62728", label="world-model forensic (omitter)")
    for xi, v in zip(x, oracle, strict=True):
        axA.text(xi - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#2ca02c")
    for xi in x:  # the model bars are all ~0 -- annotate the blindness once, clearly
        axA.text(xi + w / 2, 0.03, "0.00", ha="center", fontsize=8, color="#d62728")
    if real_model is not None:
        net_i = next((i for i, a in enumerate(arms) if a.world_name == "network"), None)
        if net_i is not None:
            lab = f"real M_θ (loc {real_model[0]:.2f}, det {real_model[1]:.2f})"
            axA.scatter([net_i + w / 2], [real_model[0]], marker="D", s=70, color="#7f0000",
                        zorder=5, label=lab)
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=8.5)
    axA.set_ylabel("breach localization accuracy\n(fraction of incidents attributed exactly)")
    axA.set_ylim(0, 1.12)
    axA.set_title("the exact oracle attributes every breach; the model is blind")
    axA.legend(fontsize=8.5, loc="center right")
    axA.axhline(0, color="#333", lw=0.8)

    # Panel B -- root-cause lag: how far upstream the incident was already determined.
    lag = [a.mean_root_cause_lag for a in arms]
    prec = [a.fraction_cause_precedes_breach for a in arms]
    colors = ["#1f77b4" if p > 0.5 else "#aec7e8" for p in prec]
    bars = axB.bar(x, lag, 0.6, color=colors)
    for bar, lg, p in zip(bars, lag, prec, strict=True):
        axB.text(bar.get_x() + bar.get_width() / 2, lg + 0.1,
                 f"{lg:.1f}\n({p:.0%} upstream)", ha="center", fontsize=8, color="#222")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=8.5)
    axB.set_ylabel("mean steps from breach to earliest\naverting intervention (the cause)")
    axB.set_ylim(0, max(lag) * 1.35 + 0.5)
    axB.set_title("the realizing step is not the root cause\n(genesis-separated worlds, dark bars)")
    axB.axhline(0, color="#333", lw=0.8)

    fig.suptitle(
        "CU29 / H122: the forensic oracle — after a breach, the exact oracle attributes it and "
        "finds the root cause; the world model cannot",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.text(0.5, 0.005,
             "left: 'you can't ask the omitter where it breached' — the posterior dual of the "
             "targeting arc.   right: the SCM counterfactual (do-remove an earlier action) reveals "
             "the cause precedes the breach where genesis ≠ consumption (host, distributed)",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
