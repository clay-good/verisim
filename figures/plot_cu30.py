"""Plot the CU30 figure (SPEC-22 §5, H123): the remediation oracle -- the recovery dual.

Two panels from a :class:`~verisim.acd.remediation_oracle.CU30Result`:

  - **left -- the corners:** every (policy, world) on the recovery plane -- mean collateral (the
    mission cost, x) vs avert rate (safety, y). Only ``min_certified`` sits in the all-good corner
    (top-left: averts every incident at minimal collateral) across all four worlds; ``sledgehammer``
    averts but far to the right (mission destroyed); the ``surgical`` undo is cheap but drops low in
    the genesis-separated worlds (a redundant consumer re-triggers the breach); the ``model`` fix
    sits at the origin (empty -- blind to the incident).
  - **right -- the headline:** avert rate per world for the naive ``surgical`` undo vs the oracle's
    ``min_certified`` fix. Undoing the realizing action averts in net/dist but FAILS in host (and
    partially segmentation); the certified fix averts everywhere. The real network ``M_theta`` (if
    available) is a marker on the network arm -- its remediation averts ~0.02 (CU8 omission).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from verisim.acd.remediation_oracle import CU30Result

_SHORT = {
    "network": "network\nexfil",
    "host": "host\ncorruption",
    "distributed": "distributed\nstaleness",
    "network (segmentation)": "network\nsegmentation",
}

_POLICY_STYLE = {
    "model": ("#d62728", "model fix (omitter)"),
    "surgical": ("#ff7f0e", "surgical undo (remove realizing action)"),
    "root_cause": ("#9467bd", "root-cause fix (CU29 genesis)"),
    "min_certified": ("#2ca02c", "min certified (oracle counterfactual)"),
    "sledgehammer": ("#8c564b", "sledgehammer (disable capability)"),
}

_WORLD_MARKER = {
    "network": "o",
    "host": "s",
    "distributed": "^",
    "network (segmentation)": "D",
}


def plot_cu30(
    result: CU30Result,
    verdict: dict[str, Any],
    path: str | Path,
    *,
    real_model: float | None = None,
) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = result.arms

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # Panel A -- the recovery plane: collateral (mission cost) vs avert rate (safety).
    for a in arms:
        marker = _WORLD_MARKER.get(a.world_name, "o")
        for p in a.policies:
            color, _ = _POLICY_STYLE[p.name]
            # jitter overlapping (0,*) points a hair so the origin cluster is legible
            axA.scatter(p.mean_collateral, p.avert_rate, marker=marker, s=95, color=color,
                        edgecolor="#333", linewidth=0.5, zorder=4, alpha=0.9)
    axA.axhspan(0.995, 1.06, xmin=0.0, xmax=0.30, color="#2ca02c", alpha=0.06, zorder=0)
    axA.text(0.12, 1.045, "all-good corner\n(averts, low collateral)", fontsize=8,
             color="#2ca02c", ha="left", va="top")
    axA.text(max(p.mean_collateral for a in arms for p in a.policies) * 0.50, 0.90,
             "sledgehammer:\naverts, mission destroyed", fontsize=8, color="#8c564b", ha="left")
    axA.text(0.18, 0.42, "surgical undo fails\n(redundant consumer)", fontsize=8,
             color="#ff7f0e", ha="left")
    # legends: colors = policies, markers = worlds
    from matplotlib.lines import Line2D

    pol_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor="#333",
               markersize=9, label=lab)
        for c, lab in _POLICY_STYLE.values()
    ]
    world_handles = [
        Line2D([0], [0], marker=m, color="w", markerfacecolor="#999", markeredgecolor="#333",
               markersize=9, label=_SHORT[w].replace("\n", " "))
        for w, m in _WORLD_MARKER.items()
    ]
    leg1 = axA.legend(handles=pol_handles, fontsize=7.8, loc="lower center", framealpha=0.9)
    axA.add_artist(leg1)
    axA.legend(handles=world_handles, fontsize=7.8, loc="center right", framealpha=0.9)
    axA.set_xlabel("mean collateral  (benign mission actions sacrificed)")
    axA.set_ylabel("avert rate  (fraction of incidents the fix undoes)")
    axA.set_ylim(-0.06, 1.10)
    axA.set_xlim(-0.25, None)
    axA.set_title("the recovery plane: only the oracle's certified fix\naverts at low collateral")
    axA.grid(True, alpha=0.25)

    # Panel B -- the headline: surgical undo vs min_certified avert rate, per world.
    labels = [_SHORT.get(a.world_name, a.world_name) for a in arms]
    x = np.arange(len(arms))
    w = 0.38
    surg = [a.by_name("surgical").avert_rate for a in arms]
    mini = [a.by_name("min_certified").avert_rate for a in arms]
    axB.bar(x - w / 2, surg, w, color="#ff7f0e", label="surgical undo (remove realizing action)")
    axB.bar(x + w / 2, mini, w, color="#2ca02c", label="min certified (oracle counterfactual)")
    for xi, v in zip(x, surg, strict=True):
        axB.text(xi - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#cc6600")
    for xi, v in zip(x, mini, strict=True):
        axB.text(xi + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#2ca02c")
    if real_model is not None:
        net_i = next((i for i, a in enumerate(arms) if a.world_name == "network"), None)
        if net_i is not None:
            axB.scatter([net_i], [real_model], marker="D", s=75, color="#7f0000", zorder=6,
                        label=f"real M_θ model fix ({real_model:.2f})")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=8.5)
    axB.set_ylabel("avert rate  (fraction of incidents the fix undoes)")
    axB.set_ylim(0, 1.14)
    axB.set_title("undoing the realizing action does not undo the breach\n"
                  "(host: a redundant consumer re-triggers it)")
    axB.legend(fontsize=8.0, loc="lower right")
    axB.axhline(1.0, color="#2ca02c", lw=0.7, ls=":", alpha=0.6)

    fig.suptitle(
        "CU30 / H123: the remediation oracle — the exact oracle computes and certifies a fix that "
        "averts the breach and preserves the mission; the model cannot",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.text(0.5, 0.005,
             "the recovery dual of CU29: the realizing step is not the cause, so undoing it does "
             "not fix the incident — only the oracle's counterfactual finds the minimal certified "
             "fix; the model's fix is empty (blind to the breach it omitted)",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
