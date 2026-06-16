"""Plot the CU39 figure (SPEC-22 H132): the redundant verifier -- defense in depth via independence.

Two panels from a :class:`~verisim.acd.verifier_redundancy.CU39Result` (network arm):

  - LEFT -- the contrast that names the principle: adversarial breach vs stack height ``m`` at a
    fixed member fidelity ``phi``. HOMOGENEOUS redundancy (copies of one monitor, a shared spot) is
    flat -- running the same scanner twice buys nothing. HETEROGENEOUS redundancy (independent blind
    spots) falls to the oracle's 0 -- the members' faithful surfaces tile the leg. Independence is
    load-bearing.
  - RIGHT -- the defense-in-depth knee: heterogeneous adversarial breach vs ``m`` over several
    ``phi``.
    The knee ``m* ~ ln A / ln(1/(1-phi))`` moves left as each member gets more faithful -- a weaker
    monitor just needs a taller stack.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.verifier_redundancy import CU39Result


def plot_cu39(result: CU39Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm = result.arms[0]  # network
    phi = min(arm.heterogeneous.keys(), key=lambda k: abs(k - result.headline_phi))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.4, 5.0))

    # --- LEFT: homogeneous vs heterogeneous at the headline phi --------------------------------
    homo = arm.homogeneous[phi]
    hetero = arm.heterogeneous[phi]
    ms = [p.m for p in hetero]
    ax_l.plot(ms, [p.adversarial_breach for p in homo], "o--", color="#c2362f", lw=2.2,
              ms=7, label="homogeneous (copies of one monitor)")
    ax_l.plot(ms, [p.adversarial_breach for p in hetero], "o-", color="#2a8a4a", lw=2.2,
              ms=7, label="heterogeneous (independent blind spots)")
    ax_l.axhline(0.0, color="#222", lw=0.8, ls=":")
    ax_l.set_ylim(-0.05, 1.05)
    ax_l.set_xlabel(f"stack height m  (number of redundant monitors, phi={phi})")
    ax_l.set_ylabel("adversarial breach   (0 = as safe as a perfect oracle)")
    ax_l.set_title("Defense in depth requires failure INDEPENDENCE\n"
                   "copies of one monitor are flat; diverse monitors tile the leg", fontsize=10.5)
    ax_l.legend(loc="center right", fontsize=9)
    ax_l.grid(alpha=0.25)

    # --- RIGHT: the defense-in-depth knee vs phi (heterogeneous) -------------------------------
    colors = ["#7b3294", "#2c7fb8", "#d95f0e"]
    for color, ph in zip(colors, sorted(arm.heterogeneous.keys()), strict=False):
        pts = arm.heterogeneous[ph]
        ax_r.plot([p.m for p in pts], [p.adversarial_breach for p in pts], "o-",
                  color=color, lw=2.2, ms=6, label=f"phi = {ph:.1f}")
    ax_r.axhline(0.0, color="#222", lw=0.8, ls=":")
    ax_r.set_ylim(-0.05, 1.05)
    ax_r.set_xlabel("stack height m  (heterogeneous)")
    ax_r.set_ylabel("adversarial breach")
    ax_r.set_title("The defense-in-depth knee m* ~ ln A / ln(1/(1-phi))\n"
                   "a weaker monitor just needs a taller stack", fontsize=10.5)
    ax_r.legend(loc="upper right", fontsize=9, title="member fidelity")
    ax_r.grid(alpha=0.25)

    fig.suptitle(
        f"CU39/H132 -- the redundant verifier: stack diverse monitors, not copies   "
        f"({arm.world_name} arm, {arm.n_scenarios} scenarios)",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
