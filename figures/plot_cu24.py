"""Plot the CU24 figure (SPEC-22 H117): the composite defense -- the whole threat model at once.

Two panels from a :class:`~verisim.acd.composite_targeting.CU24Result` -- defending three coexisting
network dangers (exfil / exposure / outage) at once:

  - **left -- the defender's coverage matrix.** Rows = the schedules a defender might deploy (each
    single point defense, each leave-one-out pair, the union); columns = the three danger legs;
    cell color = adversarial breach (green = un-gameable, red = leaks). Only the union (and the full
    oracle) is green across the whole row; every partial is red in *exactly* the leg(s) it omits --
    and the ``covers`` ✓/✗ column predicts each row a priori, before any deployment runs.
  - **right -- the cost of defense in depth.** Mean oracle calls per schedule: the union target is
    the *sum of the three rare per-leg surfaces* (a disjoint, additive stack), still far below the
    full oracle's price of total safety. Defense in depth is the union of the surfaces, not more
    verification.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.composite_targeting import CU24Result

_LEG_COLS = ("exfil", "exposure", "outage")
_LEG_HEAD = ("exfil\n(confid.)", "exposure\n(segment.)", "outage\n(availab.)")
_ROW_ORDER = (
    "exfil_only", "exposure_only", "outage_only",
    "no_exfil", "no_exposure", "no_outage", "composite",
)
_ROW_LABEL = {
    "exfil_only": "exfil only (CU10 point defense)",
    "exposure_only": "exposure only (CU17)",
    "outage_only": "availability only (CU22)",
    "no_exfil": "exposure + outage (no exfil cover)",
    "no_exposure": "exfil + outage (no segment. cover)",
    "no_outage": "exfil + exposure (forgets availab.)",
    "composite": "UNION (defense in depth)",
}


def plot_cu24(result: CU24Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    by = {c.name: c for c in result.candidates}
    rows = [by[n] for n in _ROW_ORDER]
    full = result.full_oracle

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.6, 5.3),
                                     gridspec_kw={"width_ratios": [1.18, 1.0]})

    # left: the coverage matrix (adversarial breach per leg)
    mat = np.array([[c.per_leg_adversarial[leg] for leg in _LEG_COLS] for c in rows])
    cmap = LinearSegmentedColormap.from_list("safe_leak", ["#2ca02c", "#f5f0a0", "#d62728"])
    ax_l.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax_l.set_xticks(range(len(_LEG_COLS)))
    ax_l.set_xticklabels(_LEG_HEAD, fontsize=8.4)
    ax_l.set_yticks(range(len(rows)))
    ax_l.set_yticklabels([_ROW_LABEL[c.name] for c in rows], fontsize=8.2)
    for i, c in enumerate(rows):
        for j, leg in enumerate(_LEG_COLS):
            b = c.per_leg_adversarial[leg]
            ax_l.text(j, i, f"{b:.2f}", ha="center", va="center", fontsize=8.0,
                      color="white" if b > 0.55 or b < 0.05 else "#222", fontweight="bold")
        # the a-priori covers prediction + cost, to the right of each row
        mark = "✓" if c.covers_composite else "✗"
        mc = "#2ca02c" if c.covers_composite else "#d62728"
        ax_l.text(len(_LEG_COLS) - 0.30, i, f"  covers {mark}  ·  {c.mean_calls:.1f} calls",
                  ha="left", va="center", fontsize=8.0, color=mc,
                  fontweight="bold" if c.covers_composite else "normal")
    ax_l.axhline(len(rows) - 1.5, color="#222", lw=1.6)  # separate the union row
    ax_l.set_xlim(-0.5, len(_LEG_COLS) + 2.4)
    ax_l.set_title("the defender's coverage matrix: adversarial breach per danger leg\n"
                   "(green = un-gameable, red = leaks; covers ✓/✗ predicts each row a priori)",
                   fontsize=8.6)

    # right: the cost of defense in depth -- the union is the sum of the rare per-leg surfaces
    comp = by["composite"]
    leg_calls = result.single_leg_calls
    colors = {"exfil": "#1f77b4", "exposure": "#9467bd", "outage": "#ff7f0e"}
    bottom = 0.0
    for leg in _LEG_COLS:
        ax_r.bar(0, leg_calls[leg], 0.5, bottom=bottom, color=colors[leg],
                 label=f"{leg} surface ({leg_calls[leg]:.1f})")
        bottom += leg_calls[leg]
    ax_r.bar(1, comp.mean_calls, 0.5, color="#2ca02c",
             label=f"union target ({comp.mean_calls:.1f})")
    ax_r.bar(2, full.mean_calls, 0.5, color="#888", label=f"full oracle ({full.mean_calls:.0f})")
    ax_r.axhline(full.mean_calls, color="#888", ls="--", lw=1.0)
    saving = full.mean_calls / comp.mean_calls if comp.mean_calls > 0 else float("inf")
    ax_r.annotate(
        f"defend ALL THREE\nsafe + un-gameable\n{saving:.1f}x cheaper than\nverifying everything",
        xy=(1, comp.mean_calls), xytext=(1.45, full.mean_calls * 0.55), fontsize=8.4,
        color="#2ca02c", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.3),
    )
    ax_r.set_xticks([0, 1, 2])
    ax_r.set_xticklabels(["sum of\nper-leg surfaces", "union\ntarget", "full\noracle"],
                         fontsize=8.4)
    ax_r.set_ylabel("mean oracle calls / deployment")
    ax_r.set_title("defense in depth = the union of the rare per-leg surfaces\n"
                   "(disjoint action classes, so additive), still far below total verification",
                   fontsize=8.6)
    ax_r.legend(fontsize=7.6, loc="upper center", ncol=1)
    ax_r.set_ylim(0, full.mean_calls * 1.12)

    fig.suptitle(
        "CU24 / H117: the unified target COMPOSES — one model-free union target defends the whole "
        "threat model at once (confidentiality + segmentation\n+ availability), safe + un-gameable "
        "+ cheap; every partial leaks EXACTLY its omitted leg, and covers() predicts all of it "
        f"a priori  ({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=8.6, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
