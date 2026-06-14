"""Plot the CU16 figure (SPEC-22 H109): cross-world targeting on the host world.

Two panels from a :class:`~verisim.acd.host_targeting.CU16Result`, the host mirror of CU10+CU11:

  - **left -- the cost/safety frontier (random timing, the CU10 axis).** Breach rate vs calls.
    Uniform is the blind curve that only reaches zero breach at the full oracle (top-right); model
    self-targeting is a failed point (high breach, ~0 calls); structure is the green star in the
    safe-and-cheap corner (zero breach at a small fraction of the full-oracle calls).
  - **right -- the knee is a mirage (adversarial timing, the CU11 axis).** Uniform breach vs ρ: the
    random-timing curve (falling) against the adversarial curve (flat near 1.0 until the full
    oracle), with structure's un-gameable 0 line. The same cross-world lesson the network showed:
    the danger surface (a ``write`` to a protected path) is grammar-fixed, so the attacker cannot
    time around the structure schedule.
"""

from __future__ import annotations

from pathlib import Path

from verisim.acd.host_targeting import CU16Result


def plot_cu16(result: CU16Result, path: str | Path) -> Path:  # pragma: no cover - local plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 4.9))
    full = result.uniform[-1]
    structure = result.structure
    model = result.model

    # left: cost/safety frontier (random timing) -- uniform curve, model X, structure star
    ux = [c.mean_calls for c in result.uniform]
    uy = [c.random_breach for c in result.uniform]
    ax_l.plot(ux, uy, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform (blind clock)")
    ax_l.plot(model.mean_calls, model.random_breach, "X", color="#d62728", ms=13,
              label="model self-targeting (fails)")
    ax_l.plot(structure.mean_calls, structure.random_breach, "*", color="#2ca02c", ms=20,
              label="structure (write-to-jewel)", zorder=5)
    ax_l.annotate(
        f"structure: 0 breach\n{structure.mean_calls:.1f} calls "
        f"({full.mean_calls / structure.mean_calls:.0f}x cheaper)",
        xy=(structure.mean_calls, structure.random_breach),
        xytext=(structure.mean_calls + 8, 0.18), fontsize=8.2, color="#2ca02c",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.2),
    )
    ax_l.annotate(
        f"full oracle\n{full.mean_calls:.0f} calls",
        xy=(full.mean_calls, full.random_breach), xytext=(full.mean_calls - 16, 0.16),
        fontsize=8.2, color="#555",
        arrowprops=dict(arrowstyle="->", color="#888", lw=1.0),
    )
    ax_l.set_xlabel("mean oracle calls / deployment  (the cost axis)")
    ax_l.set_ylabel("breach rate  (the safety axis)")
    ax_l.set_ylim(-0.05, 1.08)
    ax_l.set_title("danger is cheap to defend if you target the write-to-jewel surface",
                   fontsize=9.0)
    ax_l.legend(fontsize=8.0, loc="upper center")

    # right: the knee is a mirage -- random vs adversarial breach vs ρ
    rx = [c.rho for c in result.uniform]
    rand = [c.random_breach for c in result.uniform]
    adv = [c.adversarial_breach for c in result.uniform]
    ax_r.plot(rx, rand, "-o", color="#7f7fbf", lw=2.0, ms=6, label="uniform, random timing")
    ax_r.plot(rx, adv, "-X", color="#d62728", lw=2.0, ms=8, label="uniform, adversarial timing")
    ax_r.axhline(structure.adversarial_breach, color="#2ca02c", lw=2.0, ls="--",
                 label="structure, adversarial (un-gameable)")
    ax_r.fill_between(rx, rand, adv, color="#d62728", alpha=0.10)
    ax_r.set_xlabel("verification budget ρ  (uniform schedule)")
    ax_r.set_ylabel("breach rate over the deployment")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.set_title("the uniform knee is a mirage: adversarial timing erases it", fontsize=9.0)
    ax_r.legend(fontsize=8.0, loc="center left")

    fig.suptitle(
        "CU16 / H109: the targeting result generalizes to the host world — a /passwd corruption is "
        "born only by a write to\nan fd bound to it, so structure targeting is cheap AND "
        f"un-gameable there too  ({result.n_episodes} deployments, horizon {result.horizon})",
        fontsize=9.2, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
