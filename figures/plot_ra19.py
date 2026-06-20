"""Plot the RA19 figure (SPEC-22 H151, recast): the framing-transfer trap and layer saturation.

Three panels, honest after the adversarial review:

  - **A -- the residual heatmap (subset x harm, ambient).** The triad columns are red for every
    oracle-absent subset and green the instant the oracle is present. On the triad the action-
    readers are flat by SATURATION (both marginals ~1.0 under ambient), so stacking them does
    nothing; only the oracle reaches 0.
  - **B -- the framing-transfer trap is a SENSITIVITY, not a point.** The deployer's over-estimate
    factor (true ambient residual / naive estimate from the monitor's measured direct recall) vs the
    monitor's assumed ambient triad miss. ~20x at 1.0, degrading ~linearly; the >10x claim survives
    only for ambient miss >= ~0.55 (marked). The point is the conditionality, not a single number.
  - **C -- correlation has teeth only OFF the triad.** residual_true vs rho for alignment+monitor:
    on disguised-ops it rises product -> min (real correlation); on the triad it is FLAT (rho inert,
    saturated marginals); the oracle-present curve is flat at 0.
"""

from __future__ import annotations

from pathlib import Path

from verisim.realagent.ra19_correlated_layers import (
    HARM_CLASSES,
    Layer,
    RA19Result,
    residual_true,
    subsets,
    trap_sensitivity,
)


def plot_ra19(result: RA19Result, path: str | Path) -> Path:  # pragma: no cover - local plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16.5, 5.0))
    subs = subsets()
    labels = ["+".join(x[:4] for x in s) if s else "none" for s in subs]

    # --- A: residual heatmap (ambient) ------------------------------------------------------------
    grid = [[residual_true(s, h, 1.0, "ambient") for h in HARM_CLASSES] for s in subs]
    im = ax_a.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax_a.set_xticks(range(len(HARM_CLASSES)))
    ax_a.set_xticklabels([h.replace("_", "\n") for h in HARM_CLASSES], fontsize=7)
    ax_a.set_yticks(range(len(subs)))
    ax_a.set_yticklabels(labels, fontsize=7)
    for i in range(len(subs)):
        for j in range(len(HARM_CLASSES)):
            ax_a.text(j, i, f"{grid[i][j]:.2f}", ha="center", va="center", fontsize=6.5,
                      color="white" if grid[i][j] > 0.5 else "black")
    ax_a.set_title("A. residual: triad flat by\nsaturation; only oracle -> 0", fontsize=9)
    fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)

    # --- B: the framing-transfer trap sensitivity -------------------------------------------------
    miss = [i / 50 for i in range(51)]
    factor = [trap_sensitivity(m) for m in miss]
    ax_b.plot(miss, factor, "-", color="#c44", lw=2.2)
    ax_b.axhline(10.0, color="#888", ls=":", lw=1.2)
    ax_b.axvline(0.55, color="#2a7", ls="--", lw=1.2)
    ax_b.annotate(">10x survives\nfor miss >= ~0.55", (0.56, 13), fontsize=7, color="#2a7")
    ax_b.scatter([1.0], [trap_sensitivity(1.0)], s=60, color="#c44", zorder=3)
    ax_b.annotate("20x (worst case,\nmonitor ambient = alignment)", (1.0, 20),
                  textcoords="offset points", xytext=(-150, -6), fontsize=7, color="#c44")
    ax_b.set_xlabel("assumed monitor AMBIENT triad miss (the load-bearing prior)")
    ax_b.set_ylabel("framing-transfer trap factor  (true ambient / naive direct)")
    ax_b.set_title("B. the trap is a SENSITIVITY\n(direct recall used in ambient)", fontsize=9)
    ax_b.set_xlim(0, 1.05)
    ax_b.grid(True, alpha=0.3)

    # --- C: correlation has teeth only off the triad ----------------------------------------------
    s_am: tuple[Layer, ...] = ("alignment", "monitor")
    s_amo: tuple[Layer, ...] = ("alignment", "monitor", "oracle")
    rhos = [i / 20 for i in range(21)]
    dis = [residual_true(s_am, "disguised_ops", r, "ambient") for r in rhos]
    rel = [residual_true(s_am, "relational", r, "ambient") for r in rhos]
    orc = [residual_true(s_amo, "disguised_ops", r, "ambient") for r in rhos]
    ax_c.plot(rhos, dis, "-o", color="#e80", ms=3,
              label="align+monitor (disguised-ops): rho has teeth")
    ax_c.plot(rhos, rel, "--s", color="#c44", ms=3,
              label="align+monitor (relational): rho INERT (sat.)")
    ax_c.plot(rhos, orc, "-", color="#2a7", lw=2.4, label="+ oracle: flat at 0")
    ax_c.set_xlabel("correlation rho  (0 = independent salts, 1 = shared blind spot)")
    ax_c.set_ylabel("residual missed-harm")
    ax_c.set_ylim(-0.05, 1.1)
    ax_c.set_title("C. correlation has teeth only OFF the triad\n(product -> min on disguised-ops)",
                   fontsize=9)
    ax_c.legend(fontsize=6.5, loc="center right")
    ax_c.grid(True, alpha=0.3)

    fig.suptitle(
        "RA19 / H151 (recast) -- framing-transfer trap & layer saturation: action-readers don't "
        "compose on effect-harms; the oracle is the unique framing-robust closer",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
