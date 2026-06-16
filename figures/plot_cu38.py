"""Plot the CU38 figure (SPEC-22 H131): the heterogeneous verifier ensemble -- the dual of CU24.

One decisive panel from a :class:`~verisim.acd.verifier_ensemble.CU38Result`: the adversarial-breach
grid over the verifier panel (rows) x the three CIA legs + the composite (columns). Cell color is
the breach (green = 0.000 = as safe as the perfect oracle, red = leaks). The story reads off the
grid:

  - every SINGLE cheap monitor (state-diff, structure, read-audit) has a red cell -- it is blind on
    at least one leg, so it leaks the composite (the rightmost column red);
  - the ENSEMBLE {state-diff, read-audit} row is all green -- its members' faithful surfaces tile
    CIA, so it is exactly as safe as the exact-oracle row above it;
  - dropping a member (the two ensemble-minus rows) re-opens exactly the leg that member was the
    only one faithful on -- the composition theorem read backward.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.verifier_ensemble import _LEGS, _PANEL, CU38Result, _cell


def plot_cu38(result: CU38Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    scopes = (*_LEGS, "composite")
    rows = list(_PANEL)
    grid = [[_cell(result, v, s).adversarial_breach for s in scopes] for v in rows]

    cmap = LinearSegmentedColormap.from_list("safe_leak", ["#2a8a4a", "#f4d35e", "#c2362f"])

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(scopes)))
    ax.set_xticklabels(["integrity\n(content)", "availability\n(proc table)",
                        "confidentiality\n(disclosure)", "COMPOSITE\n(CIA)"], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)

    for i, _v in enumerate(rows):
        for j, _s in enumerate(scopes):
            val = grid[i][j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.45 else "#10301a",
                    fontsize=9, fontweight="bold")

    # mark the headline row (the safe ensemble) and the composite column
    ax.axhline(4 - 0.5, color="#222", lw=1.2)
    ax.axhline(4 + 0.5, color="#222", lw=1.2)
    ax.axvline(len(scopes) - 1 - 0.5, color="#222", lw=1.2)

    ax.set_title(
        "A panel of cheap partial monitors = a perfect oracle iff faithful surfaces tile CIA\n"
        "(CU38/H131: the verifier-side dual of CU24 -- adversarial breach; 0 = as safe as oracle)",
        fontsize=10.5,
    )
    ax.set_xlabel(f"danger scope   ({result.n_episodes} deployments)", fontsize=9)
    fig.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
