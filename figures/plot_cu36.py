"""Plot the CU36 figure (SPEC-22 H129): the grounded verifier -- CU35's fidelity law, real models.

Two panels from a :class:`~verisim.acd.grounded_verifier.CU36Result`:

  - **left -- the localization grid.** Adversarial breach for every (verifier x CIA leg) cell, from
    the perfect oracle (top) through two REAL partial verifiers to no gate (bottom). Green (breach
    0) = exactly as safe as the perfect oracle; red (breach high) = exactly as blind as no gate.
    Each cell is annotated with CU35's dual condition ``faithful_on_surface`` (check / cross). The
    state-diff verifier is green on integrity + availability and red on confidentiality (the
    footprintless leg); the structure verifier is green only on availability. A globally partial
    verifier is locally exact wherever it is faithful on the surface.
  - **right -- the structural predictor.** The grammar fact behind every cell: which host state
    channel each danger mutates (rows) and which channels each verifier observes (columns). A cell
    is safe iff the danger's channel is one the verifier observes -- so a defender reads the cheap
    verifier's safety off the danger grammar, without ever measuring its fidelity.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.grounded_verifier import CU36Result

_VERIFIERS = ["exact oracle", "state-diff", "structure", "no gate"]
_LEGS = ["availability", "integrity", "confidentiality"]
_LEG_LABEL = {
    "availability": "availability\n(kill daemon)",
    "integrity": "integrity\n(write file)",
    "confidentiality": "confidentiality\n(read secret)",
}
_CHANNELS = ["process table", "file content", "(output only)"]
_OBSERVES = {  # which channels each verifier observes (for the right-panel predictor)
    "exact oracle": {"process table", "file content", "(output only)"},
    "state-diff": {"process table", "file content"},
    "structure": {"process table"},
    "no gate": set(),
}
_MUTATES = {  # which channel each danger mutates
    "availability": "process table",
    "integrity": "file content",
    "confidentiality": "(output only)",
}


def plot_cu36(result: CU36Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    cell = {(c.verifier, c.leg): c for c in result.cells}
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # --- left: the breach heatmap with the faithful-on-surface annotation ------------------------
    cmap = LinearSegmentedColormap.from_list("safe_leak", ["#2a7", "#f2e36b", "#c44"])
    grid = [[cell[(v, leg)].adversarial_breach for leg in _LEGS] for v in _VERIFIERS]
    ax_l.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax_l.set_xticks(range(len(_LEGS)))
    ax_l.set_xticklabels([_LEG_LABEL[leg] for leg in _LEGS], fontsize=9)
    ax_l.set_yticks(range(len(_VERIFIERS)))
    ax_l.set_yticklabels(_VERIFIERS, fontsize=10)
    for i, v in enumerate(_VERIFIERS):
        for j, leg in enumerate(_LEGS):
            c = cell[(v, leg)]
            mark = "✓" if c.faithful_on_surface else "✗"
            txt = f"{c.adversarial_breach:.2f}\n{mark} faithful" if c.faithful_on_surface \
                else f"{c.adversarial_breach:.2f}\n{mark} blind"
            ax_l.text(j, i, txt, ha="center", va="center", fontsize=8.4,
                      color="white" if c.adversarial_breach > 0.55 or c.adversarial_breach < 1e-9
                      else "#222", fontweight="bold")
    ax_l.set_title("Adversarial breach: green = as safe as the oracle, red = as blind as no gate",
                   fontsize=10.5)

    # --- right: the structural predictor (danger channel x verifier-observed channels) -----------
    ax_r.set_xlim(0, len(_VERIFIERS))
    ax_r.set_ylim(0, len(_LEGS))
    ax_r.set_xticks([i + 0.5 for i in range(len(_VERIFIERS))])
    ax_r.set_xticklabels(_VERIFIERS, fontsize=9.5)
    ax_r.set_yticks([i + 0.5 for i in range(len(_LEGS))])
    ax_r.set_yticklabels(
        [f"{leg}\nmutates: {_MUTATES[leg]}" for leg in reversed(_LEGS)], fontsize=8.6
    )
    for i, v in enumerate(_VERIFIERS):
        for j, leg in enumerate(reversed(_LEGS)):
            observes = _OBSERVES[v]
            safe = _MUTATES[leg] in observes
            ax_r.add_patch(Rectangle((i, j), 1, 1, facecolor="#2a7" if safe else "#c44",
                                     alpha=0.32, edgecolor="white"))
            ax_r.text(i + 0.5, j + 0.5, "safe" if safe else "blind", ha="center", va="center",
                      fontsize=8.6, color="#175" if safe else "#822", fontweight="bold")
    ax_r.set_title("...read off the grammar: safe iff the verifier observes\n"
                   "the channel the danger mutates", fontsize=10.0)
    ax_r.set_aspect("auto")

    fig.suptitle(
        "CU36 / H129 -- the grounded verifier: a real verifier is exactly as safe as the oracle on "
        "the danger surface it observes",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out
