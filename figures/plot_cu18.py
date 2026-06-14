"""Plot the CU18 figure (SPEC-22 H111): the asynchronous danger -- target the medium, not action.

Two panels from a :class:`~verisim.acd.dist_targeting.CU18Result`, the distributed boundary on the
targeting arc:

  - **left -- the cost/safety frontier (random timing).** Breach rate vs calls. Uniform is the blind
    curve that only reaches zero breach at the full oracle (top-right); model self-targeting and
    write_target (the CU10-16 genesis-action transfer) are *failed* points -- high breach, and
    write_target even spends calls (false security); the medium target is the green star in the
    safe-and-cheap corner (zero breach at a small fraction of the full-oracle calls -- and cheaper
    than the failing write_target).
  - **right -- the knee is a mirage (adversarial timing).** Uniform breach vs ρ: the random-timing
    curve (falling) against the adversarial curve (flat near 1.0 until the full oracle), with the
    medium target's un-gameable 0 line. The distributed lesson: the danger is consumed at a read
    whose staleness the attacker controls by *timing*, so only verifying the medium condition at
    consumption survives.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.dist_targeting import CU18Result


def plot_cu18(result: CU18Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))
    full = result.uniform[-1]
    medium = result.medium
    write = result.write_target
    model = result.model

    # left: cost/safety frontier (random timing) -- uniform curve, failed points, medium star
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.random_breach for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_l.plot(model.mean_calls, model.random_breach, "X", color="#d62728", ms=13,
              label="model self-targeting (fails)")
    ax_l.plot(write.mean_calls, write.random_breach, "P", color="#ff7f0e", ms=13,
              label="write-to-key = genesis action (does not transfer)")
    ax_l.plot(medium.mean_calls, medium.random_breach, "*", color="#2ca02c", ms=20,
              label="medium = stale-read closure", zorder=5)
    ax_l.annotate(
        f"medium: 0 breach\n{medium.mean_calls:.1f} calls "
        f"({full.mean_calls / medium.mean_calls:.0f}x cheaper)",
        xy=(medium.mean_calls, medium.random_breach),
        xytext=(medium.mean_calls + 8, 0.20), fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_l.annotate(
        f"genesis-action target:\nspends {write.mean_calls:.1f} calls,\nstill breaches "
        f"{write.random_breach:.2f}",
        xy=(write.mean_calls, write.random_breach),
        xytext=(write.mean_calls + 7, 0.80), fontsize=8.0, color="#ff7f0e",
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.0),
    )
    ax_l.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.random_breach), xytext=(full.mean_calls - 16, 0.16),
        fontsize=8.2, color="#555",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_l.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_l.set_ylabel("breach rate  (the safety axis)")
    ax_l.set_ylim(-0.05, 1.10)
    ax_l.set_title("the genesis-action target does not transfer; target the medium at consumption",
                   fontsize=9.0)
    ax_l.legend(fontsize=7.6, loc="center right")

    # right: the knee is a mirage -- random vs adversarial breach vs ρ
    rx = [c.rho for c in result.uniform]
    rand = [c.random_breach for c in result.uniform]
    adv = [c.adversarial_breach for c in result.uniform]
    ax_r.plot(rx, rand, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform, random timing")
    ax_r.plot(rx, adv, "-X", color="#d62728", lw=2.0, ms=8, label="uniform, adversarial timing")
    ax_r.axhline(medium.adversarial_breach, color="#2ca02c", lw=2.0, ls="--",
                 label="medium, adversarial (un-gameable)")
    ax_r.fill_between(rx, rand, adv, color="#d62728", alpha=0.10)
    ax_r.set_xlabel("verification budget ρ  (uniform schedule)")
    ax_r.set_ylabel("breach rate over the deployment")
    ax_r.set_ylim(-0.03, 1.07)
    ax_r.set_title("the uniform knee is a mirage: adversarial timing erases it", fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center left")

    fig.suptitle(
        "CU18 / H111: the asynchronous danger — a stale read's genesis (a write under partition) "
        "is separated from its\nconsumption (the read), so the genesis-action target fails; only "
        f"the medium-condition target is safe AND cheap  ({result.n_episodes} deployments, horizon "
        f"{result.horizon})",
        fontsize=9.0, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
