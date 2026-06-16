"""Plot the CU33 figure (SPEC-22 H126): the value of the oracle -- the cost-optimal policy.

Two panels from a :class:`~verisim.acd.oracle_value.CU33Result`:

  - **left -- the dial is real vs nature, flat vs an adversary.** The uniform schedule's breach as a
    function of the verification budget. The RANDOM breach (blue) slopes smoothly down with the
    budget -- the tuning dial a practitioner reaches for. The ADVERSARIAL breach (red) is FLAT at
    1.0 across every sub-oracle budget, dropping only at the full-oracle cliff -- the dial does
    nothing.
    The structure target (green star) is safe against BOTH at a small fraction of the budget: it
    skips the whole curve.
  - **right -- the value of the oracle: when to verify.** Expected operational loss (in units of the
    call cost ``c``) vs the stake ratio ``C/c``, under the adversary. ``free`` (accept the loss)
    rises as ``C/c``; ``structure`` is flat at ``calls_structure``; ``full_oracle`` is flat higher
    and dashed (Pareto-dominated -- never optimal). They cross at the critical ratio
    ``C/c = calls_structure``: below it accept the loss, above it verify the structure surface.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.oracle_value import CU33Result


def plot_cu33(result: CU33Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.4, 4.9))
    econ = result.arm(result.demo_world)
    struct = econ.point("structure")
    full = econ.point("full_oracle")
    crit = econ.critical_ratio_units
    horizon = result.horizon

    # --- left: the dial (uniform breach vs budget), nature vs adversary ---------------------------
    rhos = [d.rho for d in result.dial_curve]
    ax_l.plot(rhos, [d.random_breach for d in result.dial_curve], "o-", color="#2c7fb8",
              lw=2.2, label="vs nature (random workload)")
    ax_l.plot(rhos, [d.adversarial_breach for d in result.dial_curve], "s--", color="#d62728",
              lw=2.2, label="vs an adversary (worst-case timing)")
    # structure's operating point: breach 0 on both, at calls/horizon of the budget
    ax_l.scatter([struct.calls / horizon], [0.0], marker="*", s=320, color="#2ca02c",
                 zorder=6, edgecolor="black", linewidth=0.6,
                 label=f"structure target ({struct.calls:.1f} calls, breach 0)")
    ax_l.annotate("the dial you tuned\nagainst nature is FLAT\nagainst the adversary",
                  xy=(0.5, 1.0), xytext=(0.18, 0.55), fontsize=8.5, color="#d62728",
                  arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.2})
    ax_l.set_xlabel("uniform verification budget  $\\rho$  (fraction of steps verified)")
    ax_l.set_ylabel("breach rate")
    ax_l.set_ylim(-0.05, 1.08)
    ax_l.set_title("The safety/cost dial: real vs nature, an illusion vs an adversary")
    ax_l.legend(loc="center right", fontsize=8.3, framealpha=0.95)
    ax_l.grid(True, alpha=0.3)

    # --- right: expected loss vs stakes (the value of the oracle) ---------------------------------
    ratios = [lp.ratio for lp in result.loss_curve]
    ax_r.plot(ratios, [lp.free_loss for lp in result.loss_curve], "o-", color="#7f7f7f",
              lw=2.2, label="free  (accept the loss)")
    ax_r.plot(ratios, [lp.structure_loss for lp in result.loss_curve], "*-", color="#2ca02c",
              lw=2.4, markersize=11, label=f"structure  ({struct.calls:.1f} calls, breach 0)")
    ax_r.plot(ratios, [lp.full_loss for lp in result.loss_curve], "s--", color="#9467bd",
              lw=1.8, label=f"full oracle  ({full.calls:.0f} calls -- dominated)")
    ax_r.axvline(crit, color="black", ls=":", lw=1.4)
    ax_r.annotate(f"deploy threshold\n$C/c = {crit:.1f}$ calls", xy=(crit, crit),
                  xytext=(crit * 1.5, crit * 4.5), fontsize=8.6,
                  arrowprops={"arrowstyle": "->", "lw": 1.0})
    ymax = max(lp.full_loss for lp in result.loss_curve) * 1.3
    ax_r.axvspan(min(ratios), crit, color="#d62728", alpha=0.06)
    ax_r.axvspan(crit, max(ratios), color="#2ca02c", alpha=0.06)
    ax_r.text(crit * 0.45, ymax * 0.86, "accept\nthe loss", ha="center", fontsize=8.5,
              color="#a11")
    ax_r.text(crit * 4.0, ymax * 0.86, "verify the structure surface", ha="center", fontsize=8.5,
              color="#161")
    ax_r.set_xscale("log")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("stake ratio  $C/c$  (breach cost / oracle-call cost)")
    ax_r.set_ylabel("expected loss  (units of $c$, worst case)")
    ax_r.set_title("The value of the oracle: verify iff a breach costs > a handful of calls")
    ax_r.legend(loc="lower right", fontsize=8.3, framealpha=0.95)
    ax_r.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "CU33 / H126 -- the coverage theorem turns the safety/cost dial into a binary "
        "'cover or accept the loss'; coverage is cheap",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
