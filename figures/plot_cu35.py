"""Plot the CU35 figure (SPEC-22 H128): the verifier-fidelity condition (the dual coverage law).

Three panels from a :class:`~verisim.acd.verifier_fidelity.CU35Result`:

  - **left -- the on-surface fidelity dial (the load-bearing knob).** Adversarial and random breach
    vs the verifier's fidelity ON the danger surface (``phi``), under a covering target. Random
    breach falls smoothly with ``phi`` -- the verifier's faithful horizon (safety improves gradually
    vs nature) -- but adversarial breach is a CLIFF: every partial fidelity leaks, only the exact
    verifier (``phi=1``) is safe. The verifier dial is sloped vs nature, a mirage vs the adversary
    (the verifier analogue of CU33's budget dial / CU11 un-gameability).
  - **middle -- the off-surface fidelity dial (irrelevant to safety).** Vs the verifier's fidelity
    OFF the surface (``psi``), with on-surface fidelity held exact. Breach is flat at zero for all
    ``psi`` -- a verifier globally wrong but exact on the danger grammar is as safe as a perfect
    oracle, because a covering target consults it only on the surface. Off-surface drift buys only
    rising false blocks (a utility cost on the twin axis), never a missed danger.
  - **right -- the 2x2: both coverage conditions are independently necessary.** Adversarial breach
    across {target covers?} x {verifier faithful on surface?}. Only the (covers AND faithful) corner
    is safe; a covering target with a blind verifier leaks (CU35), and a faithful verifier does not
    save a non-covering target (CU21).
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.verifier_fidelity import CU35Result


def plot_cu35(result: CU35Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax_l, ax_m, ax_r) = plt.subplots(1, 3, figsize=(16.2, 4.8))
    colors = {"network": "#2a6", "host": "#26a"}

    # --- left: on-surface fidelity dial (random sloped horizon vs adversarial cliff) --------------
    for arm in result.arms:
        c = colors.get(arm.world_name, "#888")
        phi = [p.fidelity for p in arm.on_surface]
        ax_l.plot(phi, [p.random_breach for p in arm.on_surface], "-o", color=c, markersize=4,
                  label=f"{arm.world_name}: random (vs nature)")
        ax_l.plot(phi, [p.adversarial_breach for p in arm.on_surface], "--s", color=c, markersize=4,
                  alpha=0.75, label=f"{arm.world_name}: adversarial (worst case)")
    ax_l.set_xlabel("verifier fidelity ON the danger surface  (φ)")
    ax_l.set_ylabel("breach rate")
    ax_l.set_ylim(-0.04, 1.08)
    ax_l.set_title("On-surface fidelity is load-bearing\n"
                   "(sloped vs nature, a cliff vs the adversary)")
    ax_l.annotate("partial fidelity is a mirage:\nonly the exact verifier is safe",
                  xy=(0.5, 1.0), xytext=(0.12, 0.62), fontsize=8.5, color="#a33",
                  arrowprops=dict(arrowstyle="->", color="#a33"))
    ax_l.legend(loc="lower left", fontsize=7.2, framealpha=0.9)

    # --- middle: off-surface fidelity dial (breach flat, false blocks rise) -----------------------
    arm = result.arms[0]  # the network arm carries the utility signal (target broader than danger)
    psi = [p.fidelity for p in arm.off_surface]
    ax_m.plot(psi, [p.adversarial_breach for p in arm.off_surface], "-o", color="#2a6",
              markersize=4, label="adversarial breach")
    ax_m.set_xlabel("verifier fidelity OFF the danger surface  (ψ)")
    ax_m.set_ylabel("breach rate", color="#2a6")
    ax_m.set_ylim(-0.04, 1.08)
    ax_m.set_title("Off-surface fidelity is irrelevant to safety\n"
                   "(breach flat at 0; it buys only false blocks)")
    ax_m.annotate("breach flat at 0 for all ψ:\nnever consulted off-surface",
                  xy=(0.5, 0.0), xytext=(0.18, 0.30), fontsize=8.5, color="#2a6",
                  arrowprops=dict(arrowstyle="->", color="#2a6"))
    ax_tw = ax_m.twinx()
    ax_tw.plot(psi, [p.mean_false_blocks for p in arm.off_surface], "-^", color="#d80",
               markersize=4, label="false blocks (utility cost)")
    ax_tw.set_ylabel("mean false blocks / deployment", color="#d80")
    ax_tw.set_ylim(bottom=0)
    lines = ax_m.get_lines() + ax_tw.get_lines()
    ax_m.legend(lines, [ln.get_label() for ln in lines], loc="center right", fontsize=7.6,
                framealpha=0.9)

    # --- right: the 2x2 -- both conditions independently necessary --------------------------------
    g = result.grid
    grid = np.array([[g.covers_exact, g.leak_exact], [g.covers_blind, g.leak_blind]])
    ax_r.imshow(grid, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax_r.set_xticks([0, 1])
    ax_r.set_xticklabels(["target\ncovers", "target\nnon-covering"])
    ax_r.set_yticks([0, 1])
    ax_r.set_yticklabels(["verifier\nexact on surface", "verifier\nblind on surface"])
    for i in range(2):
        for j in range(2):
            val = grid[i, j]
            ax_r.text(j, i, f"{val:.2f}\n{'SAFE' if val <= 1e-9 else 'LEAKS'}",
                      ha="center", va="center", fontsize=10,
                      color="white" if val > 0.5 else "black", fontweight="bold")
    ax_r.set_title(f"Both conditions necessary ({g.world_name})\n"
                   "only covers AND faithful is safe")

    fig.suptitle("CU35 / H128 -- the verifier must be faithful on the danger surface "
                 "(the dual of CU21 target coverage)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(path)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
